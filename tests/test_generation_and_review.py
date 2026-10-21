import csv
import json

import pytest

from kamiknows import chunking, generation, retrieval, review
from kamiknows.common import read_jsonl, read_manifest

PASSAGES = [{"label": "C1", "chunk_id": "a1"}, {"label": "C2", "chunk_id": "b1"}]


def test_parse_response_json_wrapped_and_invalid():
    answer, warnings = generation.parse_response(
        '<think>hm</think>Sure: {"answer_status": "answered", "answer": "GAN [C1]", "citations": ["[C1]", "C1"],'
        ' "confidence": "HIGH"} done')
    assert answer["citations"] == ["C1"] and answer["confidence"] == "high"
    assert warnings == ["json_extracted_from_response"]

    answer, warnings = generation.parse_response("not json, see C2")
    assert answer["answer"] == "not json, see C2" and answer["citations"] == ["C2"]
    assert "json_parse_error" in warnings


def test_validate_citations():
    def answer(status, text, citations):
        return {"answer_status": status, "answer": text, "evidence": "", "limitations": "", "citations": citations}

    assert generation.validate_citations(answer("answered", "x [C1]", ["C2"]), PASSAGES)["cited_chunk_ids"] == ["a1", "b1"]
    assert generation.validate_citations(answer("answered", "x [C7]", []), PASSAGES)["status"] == "invented_citation"
    assert generation.validate_citations(answer("answered", "x", []), PASSAGES)["status"] == "answered_without_citation"
    assert generation.validate_citations(answer("insufficient_evidence", "", []), PASSAGES)["status"] == "valid"


def test_build_prompt_lists_passages_and_requested_sources():
    context = {"question": "Q?", "source_constraint": {"requested_arxiv_ids": ["1.1"], "status": "ok"},
               "passages": [{"label": "C1", "chunk_id": "a1", "paper_id": "p", "arxiv_id": "1.1",
                             "title": "T", "section_heading": "S", "text": "TEXT"}]}
    prompt = generation.build_prompt(context, "grounded")
    assert "[C1]\nchunk_id: a1" in prompt and "Requested sources: 1.1" in prompt and prompt.endswith("TEXT")
    assert "<NO_CONTEXT>" in generation.build_prompt({**context, "passages": []}, "synthesis")


@pytest.fixture
def retrieval_dir(parsed_dir, questions_file, tmp_path):
    chunking.run_chunking(parsed_dir=parsed_dir, output_dir=tmp_path / "chunks", chunker_name="word",
                          chunker=chunking.make_word_chunker(target_words=100, overlap_words=0),
                          exclude_backmatter=False, params={})
    retrieval.run_retrieval(dataset_dir=tmp_path / "chunks", questions_file=questions_file,
                            output_dir=tmp_path / "retrieval", retriever="tfidf", embedding_model="",
                            device="cpu", top_k=2, source_policy="explicit_arxiv")
    return tmp_path / "retrieval"


def fake_ollama(monkeypatch, replies):
    monkeypatch.setattr(generation, "check_model", lambda model, base_url: None)

    def generate(prompt, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"format": "json", "done_reason": "stop", "eval_count": 10, "prompt_eval_count": 100}

    monkeypatch.setattr(generation, "ollama_generate", generate)


def test_generation_and_review_end_to_end(retrieval_dir, tmp_path, monkeypatch):
    fake_ollama(monkeypatch, [
        json.dumps({"answer_status": "answered", "answer": "A GAN [C1].", "citations": ["C1"]}),
        RuntimeError("timeout"),
        json.dumps({"answer_status": "insufficient_evidence", "answer": "Not in the context."}),
    ])
    summary = generation.run_generation(
        retrieval_dir=retrieval_dir, output_dir=tmp_path / "qa", model="m", prompt="grounded", question_ids=None,
        temperature=0.0, top_p=0.9, num_predict=64, timeout_seconds=5, base_url="http://x")
    assert summary["answer_status"] == {"answered": 1, "backend_error": 1, "insufficient_evidence": 1}
    assert summary["answer_status_by_expected_behavior"]["insufficient_evidence"] == {"insufficient_evidence": 1}
    assert read_manifest(tmp_path / "qa", "generate")["params"]["retriever"] == "tfidf"

    review.run_prepare(qa_dirs=[tmp_path / "qa"], output_dir=tmp_path / "review")
    with (tmp_path / "review" / "review.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["reference_answer"] == "A GAN." and rows[0]["required_answer_points"] == "Name the GAN."
    assert rows[0]["reference_evidence"] == "1111.00001v1 / Introduction: We simulate calorimeter showers"
    assert rows[0]["cited_passages"].startswith("[C1] 1111.00001v1")
    assert all(row["review_status"] == "pending" for row in rows)

    rows[0].update(review_status="done", reviewer_correctness="correct", reviewer_abstention="bad_label")
    with (tmp_path / "filled.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=review.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    result = review.summarize_review(tmp_path / "filled.csv")
    assert result["runs"]["qa"]["reviewed"] == 1 and result["runs"]["qa"]["pending"] == 2
    assert result["invalid_labels"] == [{"run": "qa", "question_id": "q1", "reviewer_abstention": "bad_label"}]


def test_generation_question_subset(retrieval_dir, tmp_path, monkeypatch):
    fake_ollama(monkeypatch, [json.dumps({"answer_status": "insufficient_evidence", "answer": "No."})])
    generation.run_generation(
        retrieval_dir=retrieval_dir, output_dir=tmp_path / "qa", model="m", prompt="synthesis",
        question_ids=["q3"], temperature=0.0, top_p=0.9, num_predict=64, timeout_seconds=5, base_url="http://x")
    assert [a["question_id"] for a in read_jsonl(tmp_path / "qa" / "answers.jsonl")] == ["q3"]
    with pytest.raises(ValueError, match="not in"):
        generation.run_generation(
            retrieval_dir=retrieval_dir, output_dir=tmp_path / "qa2", model="m", prompt="grounded",
            question_ids=["nope"], temperature=0.0, top_p=0.9, num_predict=64, timeout_seconds=5,
            base_url="http://x")
