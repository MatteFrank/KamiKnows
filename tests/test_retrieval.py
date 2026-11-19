import pytest

from kamiknows import chunking, retrieval
from kamiknows.common import read_jsonl, read_manifest


def chunk(chunk_id, arxiv_id, text):
    return {
        "chunk_id": chunk_id,
        "paper_id": arxiv_id.replace(".", "_"),
        "arxiv_id": arxiv_id,
        "title": "",
        "section_heading": "",
        "text": text,
    }


CHUNKS = [
    chunk("a1", "1111.00001v1", "generative adversarial network calorimeter showers"),
    chunk("a2", "1111.00001v1", "training details and learning rate"),
    chunk("b1", "2222.00002v1", "normalizing flows invertible likelihood"),
    chunk("b2", "2222.00002v1", "calorimeter showers with flows"),
]


def test_tfidf_ranks_the_matching_chunk_first():
    passages, constraint = retrieval.retrieve(
        retrieval.TfidfIndex(CHUNKS), CHUNKS, "adversarial network for showers", top_k=2, source_policy="none"
    )
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

    passages, constraint = retrieval.retrieve(
        index, CHUNKS, "What is in 3333.00003?", top_k=2, source_policy="explicit_arxiv"
    )
    assert passages == [] and constraint["status"] == "requested_source_unavailable"


def test_one_panel_per_figure_leaves_room_for_other_evidence():
    def panel(chunk_id, figure_id):
        return {**chunk(chunk_id, "1111.00001v1", "jet momentum distribution panel"),
                "plot_link": {"figure_id": figure_id}}

    chunks = [panel("f12a", "f12"), panel("f12b", "f12"), panel("f12c", "f12"),
              panel("f1", "f1"), chunk("t1", "1111.00001v1", "jet momentum")]
    scores = {"f12a": 0.9, "f12b": 0.89, "f12c": 0.88, "f1": 0.7, "t1": 0.5}

    class Index:
        def scores(self, query):
            return [scores[c["chunk_id"]] for c in chunks]

    def ranked(**options):
        passages, _ = retrieval.retrieve(Index(), chunks, "jet momentum", top_k=3, source_policy="none", **options)
        return [p["chunk_id"] for p in passages]

    assert ranked() == ranked(max_plots_per_figure=1) == ["f12a", "f1", "t1"]
    assert ranked(max_plots_per_figure=None) == ["f12a", "f12b", "f12c"]
    assert ranked(max_plots_per_figure=2) == ["f12a", "f12b", "f1"]


def test_evaluate_any_and_all_expected_sources():
    passages = [{"rank": 1, "arxiv_id": "2222.00002v1"}, {"rank": 2, "arxiv_id": "1111.00001v1"}]
    any_hit = retrieval.evaluate(passages, {"expected_arxiv_ids": ["1111.00001v1", "9.9"]})
    assert any_hit["hit"] and any_hit["hit_rank"] == 2
    all_hit = retrieval.evaluate(
        passages,
        {"expected_arxiv_ids": ["1111.00001v1", "2222.00002v1"], "required_all_expected_sources": True},
    )
    assert all_hit["hit"] and all_hit["hit_rank"] == 2 and all_hit["reciprocal_rank"] == 0.5
    assert retrieval.evaluate(passages, {}) == {"has_expected": False}


def test_run_retrieval(parsed_dir, questions_file, tmp_path):
    chunking.run_chunking(
        parsed_dir=parsed_dir,
        output_dir=tmp_path / "chunks",
        chunker_name="word",
        chunker=chunking.make_word_chunker(target_words=100, overlap_words=0),
        exclude_backmatter=False,
        params={},
    )
    summary = retrieval.run_retrieval(
        dataset_dir=tmp_path / "chunks",
        questions_file=questions_file,
        output_dir=tmp_path / "retrieval",
        retriever="tfidf",
        embedding_model="all-MiniLM-L6-v2",
        device="cpu",
        top_k=2,
        source_policy="explicit_arxiv",
    )
    assert summary["hits"] == summary["questions_with_expected_sources"] == 2
    contexts = read_jsonl(tmp_path / "retrieval" / "contexts.jsonl")
    assert (
        contexts[2]["passages"] == []
        and contexts[2]["annotations"]["expected_behavior"] == "insufficient_evidence"
    )
    assert read_manifest(tmp_path / "retrieval", "retrieve")["params"]["chunker"] == "word"
    with pytest.raises(ValueError, match="'retrieve' run"):
        retrieval.run_retrieval(
            dataset_dir=tmp_path / "retrieval",
            questions_file=questions_file,
            output_dir=tmp_path / "x",
            retriever="tfidf",
            embedding_model="",
            device="cpu",
            top_k=2,
            source_policy="none",
        )


def test_plot_evidence_is_scored_from_the_linked_series_without_ranking_annotations(tmp_path):
    from kamiknows.common import write_jsonl, write_manifest

    dataset = tmp_path / "dataset"
    record = chunk("plot", "1111.00001v1", "calorimeter peak energy")
    link = {"asset_id": "asset", "figure_id": "fig", "series": [{"handle": "asset/blue", "series_id": "blue", "label": "blue curve",
                                             "usable": True}]}
    record.update(evidence_type="plot", plot_link=link)
    write_jsonl(dataset / "chunks.jsonl", [record])
    write_manifest(dataset, step="chunk", params={"chunker": "sentence"}, inputs={}, summary={})
    questions = tmp_path / "questions.jsonl"
    question = {
        "question_id": "q",
        "question": "What is the calorimeter peak energy?",
        "question_style": "natural",
        "expected_plot_sources": [{"asset_id": "asset", "series_id": "blue"}],
    }
    write_jsonl(questions, [question])

    def run(name):
        return retrieval.run_retrieval(
            dataset_dir=dataset,
            questions_file=questions,
            output_dir=tmp_path / name,
            retriever="tfidf",
            embedding_model="",
            device="cpu",
            top_k=1,
            source_policy="none",
        )

    summary = run("found")
    assert summary["plot_evidence_by_style"]["natural"]["all_expected_retrieved"] == 1
    question["expected_plot_sources"] = [{"asset_id": "missing", "series_id": "red"}]
    questions = tmp_path / "questions_missing.jsonl"
    write_jsonl(questions, [question])
    assert run("missing")["plot_evidence"]["all_expected_retrieved"] == 0
    passages = read_jsonl(tmp_path / "found" / "contexts.jsonl")[0]["passages"]
    assert passages == read_jsonl(tmp_path / "missing" / "contexts.jsonl")[0]["passages"]
    assert passages[0]["plot_link"] == link and passages[0]["evidence_type"] == "plot"


def test_ranking_can_ignore_presentation_labels_without_removing_them_from_evidence():
    a = chunk('a', '1111.00001v1', 'display: top-left panel; a long incidental location label')
    a.update(retrieval_text='jet momentum peak', retrieval_section_heading='Numerical summary')
    b = {**a, 'text': 'display: bottom-right panel; another location', 'section_heading': 'Different layout'}
    assert retrieval.TfidfIndex([a]).scores('jet momentum peak') == retrieval.TfidfIndex([b]).scores('jet momentum peak')
    class Encoder:
        def encode(self, texts, kind):
            self.texts = texts
            return [[1., 0.] for _ in texts]
    enc = Encoder()
    retrieval.DenseIndex([a, b], enc)
    assert enc.texts[0] == enc.texts[1]
    passages, _ = retrieval.retrieve(retrieval.TfidfIndex([a]), [a], 'jet momentum peak', top_k=1, source_policy='none')
    assert 'top-left' in passages[0]['text']


def test_repository_form_of_arxiv_ids_is_a_source_request():
    assert retrieval.extract_arxiv_ids("In the paper arXiv 1705_02355v2: what?") == ["1705.02355v2"]
    assert retrieval.extract_arxiv_ids("figs/0_0.pdf and run 12345_678") == []


def test_a_retrieved_panel_reaches_the_series_of_its_sibling_panels(tmp_path):
    from kamiknows.common import write_jsonl, write_manifest

    dataset = tmp_path / "dataset"
    record = chunk("panel-b", "1111.00001v1", "average shower energy map")
    sibling = {"asset_id": "panel-a", "series": [{"series_id": "cells"}]}
    record.update(evidence_type="plot", plot_link={"asset_id": "panel-b", "figure_id": "fig", "series": [{"series_id": "cells"}],
                                                   "figure_panels": [sibling]})
    write_jsonl(dataset / "chunks.jsonl", [record])
    write_manifest(dataset, step="chunk", params={"chunker": "sentence"}, inputs={}, summary={})
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [{"question_id": "q", "question": "Brightest cell of the average shower?",
                             "expected_plot_sources": [{"asset_id": "panel-a", "series_id": "cells"}]}])
    summary = retrieval.run_retrieval(dataset_dir=dataset, questions_file=questions, output_dir=tmp_path / "run",
                                      retriever="tfidf", embedding_model="", device="cpu", top_k=1,
                                      source_policy="none")
    assert summary["plot_evidence"]["all_expected_retrieved"] == 1
