import pytest

from kamiknows import chunking
from kamiknows.common import read_jsonl, read_manifest

PAPER = {"paper_id": "p", "arxiv_id": "1.1", "title": "T"}


def section(text, section_id="sec_001"):
    return {"section_id": section_id, "heading": "H", "text": text}


def test_word_chunker_windows_overlap_and_short_tail():
    chunker = chunking.make_word_chunker(target_words=100, overlap_words=20)
    chunks = chunker(PAPER, [section(" ".join(f"w{i}" for i in range(260)))])
    assert [c["words"] for c in chunks] == [100, 100, 100]
    assert [c["cut_reason"] for c in chunks] == ["length_limit", "length_limit", "section_end"]
    assert chunks[1]["text"].split()[0] == "w80"
    assert [c["chunk_id"] for c in chunks] == ["p::word::0001", "p::word::0002", "p::word::0003"]
    # A tail shorter than 80 words is absorbed by the previous window.
    chunks = chunker(PAPER, [section(" ".join(f"w{i}" for i in range(250)))])
    assert [c["words"] for c in chunks] == [100, 170]


def test_split_sentences_keeps_abbreviations():
    text = "See Fig. 3 for details. The value is 3.5 GeV, e.g. small! Is it? Yes."
    assert chunking.split_sentences(text) == [
        "See Fig. 3 for details.", "The value is 3.5 GeV, e.g. small!", "Is it?", "Yes."]


def test_sentence_chunker_packs_whole_sentences_with_overlap():
    sentences = [" ".join(["word"] * 30) + "." for _ in range(10)]
    chunks = chunking.make_sentence_chunker(target_words=100, overlap_words=40)(PAPER, [section(" ".join(sentences))])
    assert [c["sentence_span"] for c in chunks] == [[0, 3], [2, 5], [4, 7], [6, 9], [8, 10]]
    assert all(c["words"] <= 100 for c in chunks)


def test_semantic_chunker_cuts_at_topic_change():
    # Two topics of 6 sentences each; the encoder maps them to orthogonal vectors.
    topic_a = [f"alpha {' '.join(['aa'] * 25)}." for _ in range(6)]
    topic_b = [f"beta {' '.join(['bb'] * 25)}." for _ in range(6)]
    chunker = chunking.SemanticChunker(
        lambda sentences: [[1.0, 0.0] if s.startswith("alpha") else [0.0, 1.0] for s in sentences],
        target_words=400, overlap_words=0, min_words=50, window_sentences=1,
        threshold_quantile=0.25, cut_policy="first_drop", min_score_drop=0.1)
    chunks = chunker(PAPER, [section(" ".join(topic_a + topic_b))])
    assert [c["sentence_span"] for c in chunks] == [[0, 6], [6, 12]]
    assert chunks[0]["cut_reason"] == "semantic_drop" and chunks[1]["cut_reason"] == "section_end"
    # Only the topic change is selected; the section end is not a scored boundary.
    assert [row["boundary"] for row in chunker.boundaries if row["selected"]] == [6]


def test_filter_backmatter():
    kept, audit = chunking.filter_backmatter([
        section("Physics text refs", "sec_001"),
        {"section_id": "sec_002", "heading": "Acknowledgements", "text": "Thanks."},
    ])
    assert [s["text"] for s in kept] == ["Physics text"]
    assert [row["action"] for row in audit] == ["remove_trailing_marker", "exclude_section"]


def test_run_chunking(parsed_dir, tmp_path):
    summary = chunking.run_chunking(
        parsed_dir=parsed_dir, output_dir=tmp_path / "chunks", chunker_name="sentence",
        chunker=chunking.make_sentence_chunker(target_words=100, overlap_words=0),
        exclude_backmatter=True, params={"target_words": 100, "overlap_words": 0})
    chunks = read_jsonl(tmp_path / "chunks" / "chunks.jsonl")
    assert summary["chunks"] == len(chunks) == 2 and summary["excluded_sections"] == 1
    assert chunks[0]["equation_ids"] == ["1111_00001v1::eq_001"]
    assert summary["cut_reasons"] == {"section_end": 2}
    equations = read_jsonl(tmp_path / "chunks" / "equations.jsonl")
    assert [(e["equation_id"], e["arxiv_id"], e["raw_latex"]) for e in equations] == [
        ("1111_00001v1::eq_001", "1111.00001v1", "E=mc^2")]
    assert read_manifest(tmp_path / "chunks", "chunk")["params"]["chunker"] == "sentence"
    with pytest.raises(FileExistsError):
        chunking.run_chunking(parsed_dir=parsed_dir, output_dir=tmp_path / "chunks", chunker_name="sentence",
                              chunker=chunking.make_sentence_chunker(target_words=100, overlap_words=0),
                              exclude_backmatter=False, params={})
