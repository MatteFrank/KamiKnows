import pytest

from kamiknows import chunking, retrieval
from kamiknows.common import read_jsonl, read_manifest


def chunk(chunk_id, arxiv_id, text):
    return {"chunk_id": chunk_id, "paper_id": arxiv_id.replace(".", "_"), "arxiv_id": arxiv_id,
            "title": "", "section_heading": "", "text": text}


CHUNKS = [
    chunk("a1", "1111.00001v1", "generative adversarial network calorimeter showers"),
    chunk("a2", "1111.00001v1", "training details and learning rate"),
    chunk("b1", "2222.00002v1", "normalizing flows invertible likelihood"),
    chunk("b2", "2222.00002v1", "calorimeter showers with flows"),
]


def test_tfidf_ranks_the_matching_chunk_first():
    passages, constraint = retrieval.retrieve(retrieval.TfidfIndex(CHUNKS), CHUNKS,
                                              "adversarial network for showers", top_k=2, source_policy="none")
    assert [p["chunk_id"] for p in passages] == ["a1", "b2"]
    assert [p["label"] for p in passages] == ["C1", "C2"] and constraint["status"] == "ok"
    assert passages[0]["text"] == CHUNKS[0]["text"]


def test_extract_arxiv_ids():
    text = "Compare arXiv:1705.02355v2, https://arxiv.org/abs/2106.05285 and 2106.05285v3 (not 3.14)."
    assert retrieval.extract_arxiv_ids(text) == ["1705.02355v2", "2106.05285"]
    assert retrieval.matches_source("2106.05285v3", "2106.05285")
    assert not retrieval.matches_source("2106.05285v3", "2106.05285v2")


def test_explicit_sources_keep_one_chunk_per_named_paper():
    index = retrieval.TfidfIndex(CHUNKS)
    question = "Compare 1111.00001v1 and 2222.00002v1 on calorimeter showers with flows"
    passages, _ = retrieval.retrieve(index, CHUNKS, question, top_k=2, source_policy="explicit_arxiv")
    assert {p["arxiv_id"] for p in passages} == {"1111.00001v1", "2222.00002v1"}

    passages, constraint = retrieval.retrieve(index, CHUNKS, "What is in 3333.00003?", top_k=2,
                                              source_policy="explicit_arxiv")
    assert passages == [] and constraint["status"] == "requested_source_unavailable"


def test_evaluate_any_and_all_expected_sources():
    passages = [{"rank": 1, "arxiv_id": "2222.00002v1"}, {"rank": 2, "arxiv_id": "1111.00001v1"}]
    any_hit = retrieval.evaluate(passages, {"expected_arxiv_ids": ["1111.00001v1", "9.9"]})
    assert any_hit["hit"] and any_hit["hit_rank"] == 2
    all_hit = retrieval.evaluate(passages, {"expected_arxiv_ids": ["1111.00001v1", "2222.00002v1"],
                                            "required_all_expected_sources": True})
    assert all_hit["hit"] and all_hit["hit_rank"] == 2 and all_hit["reciprocal_rank"] == 0.5
    assert retrieval.evaluate(passages, {}) == {"has_expected": False}


def test_run_retrieval(parsed_dir, questions_file, tmp_path):
    chunking.run_chunking(parsed_dir=parsed_dir, output_dir=tmp_path / "chunks", chunker_name="word",
                          chunker=chunking.make_word_chunker(target_words=100, overlap_words=0),
                          exclude_backmatter=False, params={})
    summary = retrieval.run_retrieval(dataset_dir=tmp_path / "chunks", questions_file=questions_file,
                                      output_dir=tmp_path / "retrieval", retriever="tfidf",
                                      embedding_model="all-MiniLM-L6-v2", device="cpu", top_k=2,
                                      source_policy="explicit_arxiv")
    assert summary["hits"] == summary["questions_with_expected_sources"] == 2
    contexts = read_jsonl(tmp_path / "retrieval" / "contexts.jsonl")
    assert contexts[2]["passages"] == [] and contexts[2]["annotations"]["expected_behavior"] == "insufficient_evidence"
    assert read_manifest(tmp_path / "retrieval", "retrieve")["params"]["chunker"] == "word"
    with pytest.raises(ValueError, match="'retrieve' run"):
        retrieval.run_retrieval(dataset_dir=tmp_path / "retrieval", questions_file=questions_file,
                                output_dir=tmp_path / "x", retriever="tfidf", embedding_model="",
                                device="cpu", top_k=2, source_policy="none")
