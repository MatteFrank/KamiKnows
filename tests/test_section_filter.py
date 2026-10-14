import json

from kamiknows.fulltext.section_filter import filter_sections
from kamiknows.rag_ready.sentence_dataset import build_sentence_dataset


def test_filter_preserves_scientific_prose_appendices_and_sources():
    sections = [
        {"section_id": "m", "heading": "Method", "text": "See references for the method."},
        {"section_id": "a", "heading": "8. Acknowledgements", "text": "Thanks. Embedded bibliography."},
        {"section_id": "r", "heading": "References", "text": "A cited paper."},
        {"section_id": "c", "heading": "Conflicts of interest", "text": "No conflict. biblio spphys"},
        {"section_id": "p", "heading": "Appendix", "text": "Scientific derivation."},
    ]
    before = json.dumps(sections)
    kept, audit = filter_sections(sections)
    assert [s["section_id"] for s in kept] == ["m", "c", "p"]
    assert kept[0]["text"] == "See references for the method."
    assert kept[1]["text"] == "No conflict."
    assert len(audit) == 3
    assert audit[0]["removed_text"] == "Thanks. Embedded bibliography."
    assert json.dumps(sections) == before


def test_builder_filter_is_opt_in_and_coverage_applies_to_retained_sections(tmp_path):
    reference, paper_dir = tmp_path / "reference", tmp_path / "paper"
    reference.mkdir()
    paper_dir.mkdir()
    sections = [{"section_id": "m", "heading": "Method", "text": "Scientific evidence."},
                {"section_id": "a", "heading": "Acknowledgments", "text": "Thanks. Reference list."}]
    source = paper_dir / "sections.json"
    source.write_text(json.dumps(sections))
    original = source.read_bytes()
    paper = {"paper_id": "p", "arxiv_id": "a", "title": "Title",
             "paper_dir": str(paper_dir), "parsing_status": "success", "files": {}}
    for name, rows in [("papers", [paper]), ("chunks", [{"text": "text", "metadata": {"domain": "test"}}]),
                       ("equations", []), ("eval_questions_v0", [])]:
        (reference / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (reference / "rag_manifest_v0.json").write_text("{}")
    baseline = build_sentence_dataset(reference_dir=reference, output_dir=tmp_path / "baseline")
    filtered_dir = tmp_path / "filtered"
    filtered = build_sentence_dataset(reference_dir=reference, output_dir=filtered_dir,
                                      exclude_backmatter=True)
    assert baseline["chunks_v1_count"] == 2
    assert "section_filter" not in baseline
    assert filtered["chunks_v1_count"] == 1
    assert filtered["section_filter"]["excluded_sections"] == 1
    assert filtered["validation"]["sentence_coverage"] == "PASS"
    chunk = json.loads((filtered_dir / "chunks_v1.jsonl").read_text())
    assert chunk["section_id"] == "m"
    assert source.read_bytes() == original
    assert json.loads((filtered_dir / "section_exclusions_v1.jsonl").read_text())["paper_id"] == "p"
