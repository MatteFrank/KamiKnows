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
        "See Fig. 3 for details.",
        "The value is 3.5 GeV, e.g. small!",
        "Is it?",
        "Yes.",
    ]


def test_sentence_chunker_packs_whole_sentences_with_overlap():
    sentences = [" ".join(["word"] * 30) + "." for _ in range(10)]
    chunks = chunking.make_sentence_chunker(target_words=100, overlap_words=40)(
        PAPER, [section(" ".join(sentences))]
    )
    assert [c["sentence_span"] for c in chunks] == [[0, 3], [2, 5], [4, 7], [6, 9], [8, 10]]
    assert all(c["words"] <= 100 for c in chunks)


def test_semantic_chunker_cuts_at_topic_change():
    # Two topics of 6 sentences each; the encoder maps them to orthogonal vectors.
    topic_a = [f"alpha {' '.join(['aa'] * 25)}." for _ in range(6)]
    topic_b = [f"beta {' '.join(['bb'] * 25)}." for _ in range(6)]
    chunker = chunking.SemanticChunker(
        lambda sentences: [[1.0, 0.0] if s.startswith("alpha") else [0.0, 1.0] for s in sentences],
        target_words=400,
        overlap_words=0,
        min_words=50,
        window_sentences=1,
        threshold_quantile=0.25,
        cut_policy="first_drop",
        min_score_drop=0.1,
    )
    chunks = chunker(PAPER, [section(" ".join(topic_a + topic_b))])
    assert [c["sentence_span"] for c in chunks] == [[0, 6], [6, 12]]
    assert chunks[0]["cut_reason"] == "semantic_drop" and chunks[1]["cut_reason"] == "section_end"
    # Only the topic change is selected; the section end is not a scored boundary.
    assert [row["boundary"] for row in chunker.boundaries if row["selected"]] == [6]


def test_filter_backmatter():
    kept, audit = chunking.filter_backmatter(
        [
            section("Physics text refs", "sec_001"),
            {"section_id": "sec_002", "heading": "Acknowledgements", "text": "Thanks."},
        ]
    )
    assert [s["text"] for s in kept] == ["Physics text"]
    assert [row["action"] for row in audit] == ["remove_trailing_marker", "exclude_section"]


def test_run_chunking(parsed_dir, tmp_path):
    summary = chunking.run_chunking(
        parsed_dir=parsed_dir,
        output_dir=tmp_path / "chunks",
        chunker_name="sentence",
        chunker=chunking.make_sentence_chunker(target_words=100, overlap_words=0),
        exclude_backmatter=True,
        params={"target_words": 100, "overlap_words": 0},
    )
    chunks = read_jsonl(tmp_path / "chunks" / "chunks.jsonl")
    assert summary["chunks"] == len(chunks) == 2 and summary["excluded_sections"] == 1
    assert chunks[0]["equation_ids"] == ["1111_00001v1::eq_001"]
    assert summary["cut_reasons"] == {"section_end": 2}
    equations = read_jsonl(tmp_path / "chunks" / "equations.jsonl")
    assert [(e["equation_id"], e["arxiv_id"], e["raw_latex"]) for e in equations] == [
        ("1111_00001v1::eq_001", "1111.00001v1", "E=mc^2")
    ]
    assert read_manifest(tmp_path / "chunks", "chunk")["params"]["chunker"] == "sentence"
    with pytest.raises(FileExistsError):
        chunking.run_chunking(
            parsed_dir=parsed_dir,
            output_dir=tmp_path / "chunks",
            chunker_name="sentence",
            chunker=chunking.make_sentence_chunker(target_words=100, overlap_words=0),
            exclude_backmatter=False,
            params={},
        )


def test_digitized_evidence_is_complete_and_keeps_provenance(tmp_path):
    from kamiknows.common import write_jsonl, write_manifest, sha256_file
    from kamiknows.retrieval import TfidfIndex, retrieve

    run = tmp_path / "digitized"
    figure_file = tmp_path / "figures.jsonl"
    write_jsonl(
        figure_file,
        [
            {
                "figure_id": "p::figure::0001",
                "caption": "A test curve.",
                "source_assets": [{"asset_id": "a", "sha256": "source_hash"}],
            }
        ],
    )
    record = {
        "paper_id": "p",
        "figure_id": "p::figure::0001",
        "data_id": "d",
        "extraction_status": "success",
        "quality": {"structural_validity": "passed"},
        "provenance": {"asset_id": "a", "asset_sha256": "source_hash", "source_path": "curve.pdf"},
        "calibration": {"x": {"scale": "linear"}, "y": {"scale": "log10"}},
        "series": [
            {
                "series_id": "s",
                "geometry": "rendered_curve_samples",
                "units": {"x": "GeV"},
                "x": list(range(30)),
                "y": [i + 1.0 for i in range(30)],
            }
        ],
    }
    data = run / "digitized_data.jsonl"
    write_jsonl(data, [record, {"extraction_status": "needs_calibration"}])
    write_manifest(
        run,
        step="extract_plots",
        params={"mode": "digitize"},
        inputs={"classified_figures": figure_file},
        summary={"data_sha256": sha256_file(data)},
    )
    chunks, inputs = chunking.digitized_chunks([run], [PAPER])
    assert len(chunks) == 3
    assert "x=29; y=30" in chunks[0]["text"]
    assert "Y scale: log10" in chunks[0]["text"]
    assert "29, 30" in chunks[-1]["text"]
    assert "Scientific validation pending" in chunks[0]["text"]
    assert chunks[0]["plot_provenance"]["asset_sha256"] == "source_hash"
    passages, _ = retrieve(TfidfIndex(chunks), chunks, "What is the maximum?", top_k=1, source_policy="none")
    assert passages[0]["evidence_type"] == "digitized_plot"
    assert passages[0]["plot_provenance"]["data_id"] == "d"
    assert "::" not in chunks[0]["text"]
    assert chunks[0]["plot_reference"]["figure"] == "Figure 1"
    from copy import deepcopy

    grouped = deepcopy(record)
    grouped["series"].append({**deepcopy(record["series"][0]), "series_id": "second"})
    second = tmp_path / "grouped"
    second_data = second / "digitized_data.jsonl"
    write_jsonl(second_data, [grouped])
    write_manifest(
        second,
        step="extract_plots",
        params={"mode": "digitize"},
        inputs={"classified_figures": figure_file},
        summary={"data_sha256": sha256_file(second_data)},
    )
    overview_chunks, _ = chunking.digitized_chunks([second], [PAPER])
    overview = next(c for c in overview_chunks if c["cut_reason"] == "plot_overview")
    assert {s["series_id"] for s in overview["plot_sources"]} == {"s", "second"}
    assert all(s["axis_metadata"]["x"]["unit"] == "GeV" for s in overview["plot_sources"])
    without, _ = chunking.digitized_chunks([second], [PAPER], include_overviews=False)
    assert not any(c["cut_reason"] == "plot_overview" for c in without)
    with pytest.raises(ValueError, match="duplicate plot asset"):
        chunking.digitized_chunks([run, run], [PAPER])
    with pytest.raises(ValueError, match="outside the parsed corpus"):
        chunking.digitized_chunks([run], [])
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="intact digitization"):
        chunking.digitized_chunks([run], [PAPER])


def test_axis_metadata_distinguishes_units_scales_and_missing_values():
    from kamiknows.plot_extractor.records import series_axis_metadata

    record = {
        "calibration": {
            "x": {"unit": "GeV", "scale": "linear"},
            "y": {"unit": "MeV", "scale": "log10", "display_multiplier": 1000},
        }
    }
    series = {"geometry": "rendered_curve_samples", "units": {"x": "GeV", "y": "MeV"}}
    axes = series_axis_metadata(record, series)
    assert axes["y"]["unit"] == "MeV" and axes["y"]["scale"] == "log10"
    assert axes["y"]["display_multiplier"] == 1000
    assert axes["y"]["values_include_display_multiplier"]
    with pytest.raises(ValueError, match="conflicting Y units"):
        series_axis_metadata(record, {**series, "units": {"y": "GeV"}})
    missing = series_axis_metadata({"calibration": {"x": {"unit": "", "unit_status": "missing"}}}, {})
    assert missing["x"]["unit"] == "unspecified"
    assert missing["y"]["unit_status"] == "unspecified"


def test_explicit_source_layout_does_not_guess_automatic_wrapping():
    tex = r'\includegraphics{a} \includegraphics{b} \\ \includegraphics{c} \includegraphics{d}'
    import re
    assets = [{"asset_id": name, "flat_span": list(m.span())} for name, m in zip('abcd', re.finditer(r'\\includegraphics\{[^}]+\}', tex))]
    positions = chunking._figure_layout({"source_assets": assets}, tex)
    assert positions['a'].startswith('top-left') and positions['d'].startswith('bottom-right')
    one_row = tex.replace(r'\\', '')
    assets = [{"asset_id": name, "flat_span": list(m.span())} for name, m in zip('abcd', re.finditer(r'\\includegraphics\{[^}]+\}', one_row))]
    assert chunking._figure_layout({"source_assets": assets}, one_row) == {}


def test_obsolete_text_classifiers_are_not_supported():
    from kamiknows.plot_extractor.classification import FigureClassifier
    for method in ['text_rules_v1', 'text_context_v2', 'text_context_v3', 'text_context_v4', 'text_context_v5']:
        with pytest.raises(ValueError, match='unsupported classification method'):
            FigureClassifier(method)
