import csv
import json

import pytest

from kamiknows import chunking, generation, retrieval, review
from kamiknows.common import read_jsonl, read_manifest

PASSAGES = [{"label": "C1", "chunk_id": "a1"}, {"label": "C2", "chunk_id": "b1"}]


def test_parse_response_json_wrapped_and_invalid():
    answer, warnings = generation.parse_response(
        '<think>hm</think>Sure: {"answer_status": "answered", "answer": "GAN [C1]", "citations": ["[C1]", "C1"],'
        ' "confidence": "HIGH"} done'
    )
    assert answer["citations"] == ["C1"] and answer["confidence"] == "high"
    assert warnings == ["json_extracted_from_response"]

    answer, warnings = generation.parse_response("not json, see C2")
    assert answer["answer"] == "not json, see C2" and answer["citations"] == ["C2"]
    assert "json_parse_error" in warnings


def test_validate_citations():
    def answer(status, text, citations):
        return {
            "answer_status": status,
            "answer": text,
            "evidence": "",
            "limitations": "",
            "citations": citations,
        }

    assert generation.validate_citations(answer("answered", "x [C1]", ["C2"]), PASSAGES)[
        "cited_chunk_ids"
    ] == ["a1", "b1"]
    assert (
        generation.validate_citations(answer("answered", "x [C7]", []), PASSAGES)["status"]
        == "invented_citation"
    )
    assert (
        generation.validate_citations(answer("answered", "x", []), PASSAGES)["status"]
        == "answered_without_citation"
    )
    assert (
        generation.validate_citations(answer("insufficient_evidence", "", []), PASSAGES)["status"] == "valid"
    )


def test_build_prompt_lists_passages_and_requested_sources():
    context = {
        "question": "Q?",
        "source_constraint": {"requested_arxiv_ids": ["1.1"], "status": "ok"},
        "passages": [
            {
                "label": "C1",
                "chunk_id": "a1",
                "paper_id": "p",
                "arxiv_id": "1.1",
                "title": "T",
                "section_heading": "S",
                "text": "TEXT",
            }
        ],
    }
    prompt = generation.build_prompt(context, "grounded")
    assert "[C1]\nchunk_id: a1" in prompt and "Requested sources: 1.1" in prompt
    # The question is restated after the passages, where long tables cannot bury it.
    assert prompt.index("TEXT") < prompt.rindex("Q?") and prompt.endswith("specified above:\nQ?")
    assert "<NO_CONTEXT>" in generation.build_prompt({**context, "passages": []}, "synthesis")


@pytest.fixture
def retrieval_dir(parsed_dir, questions_file, tmp_path):
    chunking.run_chunking(
        parsed_dir=parsed_dir,
        output_dir=tmp_path / "chunks",
        chunker_name="word",
        chunker=chunking.make_word_chunker(target_words=100, overlap_words=0),
        exclude_backmatter=False,
        params={},
    )
    retrieval.run_retrieval(
        dataset_dir=tmp_path / "chunks",
        questions_file=questions_file,
        output_dir=tmp_path / "retrieval",
        retriever="tfidf",
        embedding_model="",
        device="cpu",
        top_k=2,
        source_policy="explicit_arxiv",
    )
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
    fake_ollama(
        monkeypatch,
        [
            json.dumps({"answer_status": "answered", "answer": "A GAN [C1].", "citations": ["C1"]}),
            RuntimeError("timeout"),
            json.dumps({"answer_status": "insufficient_evidence", "answer": "Not in the context."}),
        ],
    )
    summary = generation.run_generation(
        retrieval_dir=retrieval_dir,
        output_dir=tmp_path / "qa",
        model="m",
        prompt="grounded",
        question_ids=None,
        temperature=0.0,
        top_p=0.9,
        num_predict=64,
        timeout_seconds=5,
        base_url="http://x",
    )
    assert summary["answer_status"] == {"answered": 1, "backend_error": 1, "insufficient_evidence": 1}
    assert summary["answer_status_by_expected_behavior"]["insufficient_evidence"] == {
        "insufficient_evidence": 1
    }
    params = read_manifest(tmp_path / "qa", "generate")["params"]
    assert params["retriever"] == "tfidf" and params["num_ctx"] == 32768

    review.run_prepare(qa_dirs=[tmp_path / "qa"], output_dir=tmp_path / "review")
    with (tmp_path / "review" / "review.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["reference_answer"] == "A GAN." and rows[0]["required_answer_points"] == "Name the GAN."
    assert rows[0]["reference_evidence"] == "1111.00001v1 / Introduction: We simulate calorimeter showers"
    assert rows[0]["cited_passages"].startswith("[C1] 1111.00001v1")
    assert all(row["review_status"] == "pending" for row in rows)
    assert "Human review pending" in (tmp_path / "review" / "review.html").read_text()

    rows[0].update(review_status="done", reviewer_correctness="correct", reviewer_abstention="bad_label")
    with (tmp_path / "filled.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=review.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    result = review.summarize_review(tmp_path / "filled.csv")
    assert result["runs"]["qa"]["reviewed"] == 1 and result["runs"]["qa"]["pending"] == 2
    assert result["invalid_labels"] == [
        {"run": "qa", "question_id": "q1", "reviewer_abstention": "bad_label"}
    ]


def test_generation_question_subset(retrieval_dir, tmp_path, monkeypatch):
    fake_ollama(monkeypatch, [json.dumps({"answer_status": "insufficient_evidence", "answer": "No."})])
    generation.run_generation(
        retrieval_dir=retrieval_dir,
        output_dir=tmp_path / "qa",
        model="m",
        prompt="synthesis",
        question_ids=["q3"],
        temperature=0.0,
        top_p=0.9,
        num_predict=64,
        timeout_seconds=5,
        base_url="http://x",
    )
    assert [a["question_id"] for a in read_jsonl(tmp_path / "qa" / "answers.jsonl")] == ["q3"]
    with pytest.raises(ValueError, match="not in"):
        generation.run_generation(
            retrieval_dir=retrieval_dir,
            output_dir=tmp_path / "qa2",
            model="m",
            prompt="grounded",
            question_ids=["nope"],
            temperature=0.0,
            top_p=0.9,
            num_predict=64,
            timeout_seconds=5,
            base_url="http://x",
        )


def test_prompt_filling_the_context_is_flagged(monkeypatch):
    context = {
        "question_id": "q",
        "question": "Q?",
        "source_constraint": {"requested_arxiv_ids": [], "status": "none"},
        "passages": [{"label": "C1", "chunk_id": "a", "paper_id": "p", "arxiv_id": "1.1",
                      "title": "T", "section_heading": "S", "text": "TEXT"}],
    }
    reply = json.dumps({"answer_status": "answered", "answer": "A [C1].", "citations": ["C1"]})
    evaluated = []
    monkeypatch.setattr(generation, "ollama_generate", lambda prompt, **kwargs: (
        reply, {"done_reason": "stop", "eval_count": 5, "prompt_eval_count": evaluated.pop(0)}))
    # Fewer evaluated tokens than a quarter of the prompt characters: Ollama cut the prompt.
    evaluated[:] = [10, 100_000, 10]
    small = {"num_ctx": 100}
    answer = generation.answer_question(context, model="m", prompt="grounded", options=small,
                                        base_url="http://x", timeout_seconds=5)
    assert "prompt_truncated_at_context_limit" in answer["warnings"]
    answer = generation.answer_question(context, model="m", prompt="grounded", options=small,
                                        base_url="http://x", timeout_seconds=5)
    assert "prompt_truncated_at_context_limit" not in answer["warnings"]
    # A prompt shorter than the context cannot be cut; a low count is Ollama's prefix cache.
    answer = generation.answer_question(context, model="m", prompt="grounded", options={"num_ctx": 32768},
                                        base_url="http://x", timeout_seconds=5)
    assert "prompt_truncated_at_context_limit" not in answer["warnings"]


def _solver_context():
    from kamiknows.compute import Axis, Catalog, Entry, Series, Unit

    def histogram(name, values):
        return Series(name, (Axis("x", Unit.parse("GeV")),), Axis("y", Unit.parse("A.U.")),
                      centers=[[5.0], [15.0], [25.0]], values=values,
                      lower=[[0.0], [10.0], [20.0]], upper=[[10.0], [20.0], [30.0]])

    link = {"asset_id": "p::a1", "figure_id": "p::f1", "image": "1 of 1", "position": None,
            "extraction_status": "success", "digitized_data_sha256": "h",
            "series": [{"handle": "p::a1/blue", "series_id": "blue", "label": "blue curve", "usable": True},
                       {"handle": "p::a1/red", "series_id": "red", "label": "red curve", "usable": True}]}
    context = {
        "question_id": "q", "question": "What is the ratio of the blue to the red peak?",
        "source_constraint": {"requested_arxiv_ids": [], "status": "ok"},
        "passages": [
            {"label": "C1", "chunk_id": "text", "paper_id": "p", "arxiv_id": "1.1", "title": "T",
             "section_heading": "Results", "text": "We show jets."},
            {"label": "C2", "chunk_id": "plot", "paper_id": "p", "arxiv_id": "1.1", "title": "T",
             "section_heading": "Figure 1, single image", "text": "Plot: Figure 1.", "evidence_type": "plot",
             "plot_link": link},
        ],
    }
    catalog = Catalog([Entry("p::a1/blue", histogram("p::a1/blue", [2.0, 6.0, 1.0])),
                       Entry("p::a1/red", histogram("p::a1/red", [1.0, 3.0, 2.0])),
                       Entry("p::a9/other", histogram("p::a9/other", [9.0, 9.0, 9.0]))])
    return context, catalog


def _fake_backend(monkeypatch, plan_reply):
    prompts = []
    replies = plan_reply if isinstance(plan_reply, list) else [plan_reply]

    def fake(prompt, **kwargs):
        prompts.append(prompt)
        if prompt.startswith("You plan computations"):
            planned = sum(p.startswith("You plan computations") for p in prompts)
            return replies[min(planned, len(replies)) - 1], {"done_reason": "stop"}
        answer = {"answer_status": "answered", "answer": "The ratio is 2 [C3] [C2].", "citations": ["C3", "C2"],
                  "confidence": "high"}
        return json.dumps(answer), {"done_reason": "stop", "prompt_eval_count": 10}

    monkeypatch.setattr(generation, "ollama_generate", fake)
    return prompts


def _answer(context, catalog):
    return generation.answer_question(context, model="m", prompt="grounded", options={"num_ctx": 32768},
                                      base_url="u", timeout_seconds=1, catalog=catalog)


def test_solver_computes_and_adds_a_citable_passage(monkeypatch):
    context, catalog = _solver_context()
    plan = {"steps": [{"id": "b", "op": "reduce", "args": {"series": "p::a1/blue", "how": "max"}},
                      {"id": "r", "op": "reduce", "args": {"series": "p::a1/red", "how": "max"}},
                      {"id": "q", "op": "combine", "args": {"left": "$b", "right": "$r", "op": "div"}}],
            "result": ["$q"]}
    prompts = _fake_backend(monkeypatch, json.dumps(plan))
    answer = _answer(context, catalog)
    assert answer["solver"]["status"] == "computed"
    passage = answer["computed_passage"]
    assert passage["label"] == "C3" and passage["evidence_type"] == "computed"
    assert "1. 2 (computed as div(max(arXiv:1.1 Figure 1, image 1 of 1, blue curve)" in passage["text"]
    assert "p::a1" not in passage["text"]
    assert ("Series handles:\n- blue curve: p::a1/blue (domain axes: x in GeV; value axis: y in A.U.)"
            in prompts[0] and "argextremum(" in prompts[0])
    assert len(answer["solver"]["attempts"]) == 1
    assert "[C3]" in prompts[1] and 'A passage of type "computed"' in prompts[1]
    assert answer["citation_validation"]["status"] == "valid"


@pytest.mark.parametrize("reply, status, passage", [
    ("I would compute the maximum.", "invalid_plan", False),
    ('{"no_computation": "the axes are stated"}', "no_computation", False),
    ('{"not_computable": "event counts are not drawn"}', "declined", True),
    (json.dumps({"steps": [{"id": "a", "op": "reduce", "args": {"series": "p::a9/other", "how": "max"}}],
                 "result": ["$a"]}), "not_computable", True),
])
def test_solver_outcomes_without_numbers(monkeypatch, reply, status, passage):
    context, catalog = _solver_context()
    _fake_backend(monkeypatch, reply)
    answer = _answer(context, catalog)
    assert answer["solver"]["status"] == status
    assert (answer["computed_passage"] is not None) == passage
    if passage:
        assert "could not be computed" in answer["computed_passage"]["text"]


def test_solver_is_skipped_without_plot_passages(monkeypatch):
    context, catalog = _solver_context()
    context["passages"] = context["passages"][:1]
    prompts = _fake_backend(monkeypatch, "{}")
    answer = _answer(context, catalog)
    assert answer["solver"] == {"status": "not_applicable"} and len(prompts) == 1


def test_solver_catalog_needs_plot_data(tmp_path):
    from kamiknows.common import write_manifest

    chunks = tmp_path / "chunks"
    chunks.mkdir()
    write_manifest(chunks, step="chunk", params={"chunker": "sentence", "digitized_dir": None}, inputs={}, summary={})
    with pytest.raises(ValueError, match="--digitized-dir"):
        generation.solver_catalog({"params": {"dataset_dir": str(chunks)}}, [])


def test_solver_returns_the_error_to_the_planner_and_retries(monkeypatch):
    context, catalog = _solver_context()
    wrong = {"steps": [{"id": "m", "op": "argextremum", "args": {"series": "p::a1/blue", "kind": "localmax"}}],
             "result": ["$m[0]"]}
    right = {"steps": [{"id": "m", "op": "argextremum", "args": {"series": "p::a1/blue", "kind": "max"}}],
             "result": ["$m[0]"]}
    prompts = _fake_backend(monkeypatch, [json.dumps(wrong), json.dumps(right)])
    answer = _answer(context, catalog)
    assert answer["solver"]["status"] == "computed"
    assert [a["outcome"]["status"] for a in answer["solver"]["attempts"]] == ["invalid_plan", "computed"]
    assert "It failed: argextremum: kind must be" in prompts[1] and '"localmax"' in prompts[1]
    assert "scope?: global|local" in prompts[1] and "or decline" not in prompts[1]
    assert "6 A.U. at x=15 GeV [10, 20]" in answer["computed_passage"]["text"]


def test_solver_stops_after_the_last_attempt(monkeypatch):
    context, catalog = _solver_context()
    prompts = _fake_backend(monkeypatch, "not json")
    answer = _answer(context, catalog)
    assert answer["solver"]["status"] == "invalid_plan" and len(answer["solver"]["attempts"]) == 3
    assert len(prompts) == 4


def test_solver_passage_accepts_plain_record_fields(monkeypatch):
    context, catalog = _solver_context()
    plan = {"steps": [{"id": "d", "op": "describe", "args": {"series": "p::a1/blue"}}],
            "result": ["$d.elements_are", "$d.axes"]}
    _fake_backend(monkeypatch, json.dumps(plan))
    answer = _answer(context, catalog)
    assert answer["solver"]["status"] == "computed"
    text = answer["computed_passage"]["text"]
    assert '1. "bins or cells"' in text and '"unit": "GeV"' in text


def test_a_correction_cannot_decline(monkeypatch):
    context, catalog = _solver_context()
    wrong = {"steps": [{"id": "m", "op": "argextremum", "args": {"series": "p::a1/blue", "kind": "localmax"}}],
             "result": ["$m[0]"]}
    right = {"steps": [{"id": "m", "op": "argextremum", "args": {"series": "p::a1/blue", "kind": "max"}}],
             "result": ["$m[0]"]}
    _fake_backend(monkeypatch, [json.dumps(wrong), '{"not_computable": "too hard"}', json.dumps(right)])
    answer = _answer(context, catalog)
    assert [a["outcome"]["status"] for a in answer["solver"]["attempts"]] == ["invalid_plan", "invalid_plan", "computed"]
    assert answer["solver"]["status"] == "computed"
