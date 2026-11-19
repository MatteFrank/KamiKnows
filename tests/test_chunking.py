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


FIGURE_TEX = (
    "\\begin{figure}\n\\includegraphics{curves}\\\\\n\\includegraphics{map}\n"
    "\\caption{Jet momentum curves (top) and an energy map (bottom).}\\label{fig:jets}\n\\end{figure}\n\n"
    "As Fig.~\\ref{fig:jets} shows, the weighted curve peaks at low momentum.\n"
)
ASSET_A, ASSET_B = "p::figure::0001::asset::01", "p::figure::0001::asset::02"


def _curve(series_id, colour, offset):
    return {"series_id": series_id, "geometry": "rendered_curve_samples", "units": {"x": "GeV", "y": "A.U."},
            "x": [float(i) for i in range(30)], "y": [i * 1.5 + offset for i in range(30)],
            "raster_color_rgb": colour, "value_states": ["visible"] * 30}


def _plot_record(asset, status="success", series=()):
    return {"paper_id": "p", "figure_id": "p::figure::0001", "data_id": f"d-{asset[-2:]}",
            "extraction_status": status, "quality": {"structural_validity": "passed"},
            "provenance": {"asset_id": asset, "asset_sha256": f"hash-{asset[-2:]}", "source_path": "f.pdf"},
            "calibration": {"x": {"scale": "linear"}, "y": {"scale": "log10"}}, "series": list(series)}


def _digitization_run(tmp_path, name, records):
    from kamiknows.common import sha256_file, write_jsonl, write_manifest

    flat = tmp_path / f"{name}_flat.tex"
    flat.write_text(FIGURE_TEX)
    figures = tmp_path / f"{name}_figures.jsonl"
    spans = [[FIGURE_TEX.index(n), FIGURE_TEX.index(n) + len(n)] for n in ("\\includegraphics{curves}", "\\includegraphics{map}")]
    write_jsonl(figures, [{
        "figure_id": "p::figure::0001", "caption": "Jet momentum curves (top) and an energy map (bottom).",
        "latex_label": "fig:jets",
        "source_assets": [{"asset_id": a, "sha256": f"hash-{a[-2:]}", "flat_span": s}
                          for a, s in zip((ASSET_A, ASSET_B), spans)],
    }])
    run = tmp_path / name
    data = run / "digitized_data.jsonl"
    write_jsonl(data, records)
    write_manifest(run, step="extract_plots", params={"mode": "digitize"},
                   inputs={"classified_figures": figures, "p:flat": flat}, summary={"data_sha256": sha256_file(data)})
    return run


def test_plot_chunks_describe_each_image_and_link_its_data(tmp_path):
    run = _digitization_run(tmp_path, "d", [
        _plot_record(ASSET_A, series=[_curve("s", [0.1, 0.1, 0.9], 0.25), _curve("second", [0.9, 0.1, 0.1], 1.0)]),
        _plot_record(ASSET_B, status="needs_calibration"),
    ])
    chunks, inputs = chunking.plot_chunks(run, [PAPER])
    assert [c["cut_reason"] for c in chunks] == ["plot", "plot"]
    curves, energy_map = chunks
    text = curves["text"]
    assert "Plot: Figure 1 of arXiv:1.1, image 1 of 2." in text
    assert "Caption: Jet momentum curves" in text
    assert "Paper discussion of this figure:" in text and "weighted curve peaks" in text
    assert "Axes: X in GeV (linear), Y in A.U. (log10)." in text
    assert "- blue curve: 30 points; X from 0 to 29 GeV; maximum 43.75 A.U. at X = 29 GeV." in text
    assert "- red curve:" in text
    assert "10.75" not in text and "::" not in text          # no extracted values, no storage ids
    assert "Panel position: top panel" in text and "top panel" not in curves["retrieval_text"]
    assert curves["plot_link"]["series"] == [
        {"handle": f"{ASSET_A}/s", "series_id": "s", "label": "blue curve", "usable": True},
        {"handle": f"{ASSET_A}/second", "series_id": "second", "label": "red curve", "usable": True},
    ]
    assert curves["plot_link"]["data_id"] == "d-01"
    assert curves["plot_link"]["position"] == "top panel" and curves["plot_link"]["image"] == "1 of 2"
    panels = curves["plot_link"]["figure_panels"]
    assert [(p["asset_id"], p["position"], p["extraction_status"]) for p in panels] == [
        (ASSET_A, "top panel", "success"), (ASSET_B, "bottom panel", "needs_calibration")]
    assert energy_map["plot_link"]["figure_panels"] == panels
    assert "Data status: plot not digitized (extraction status: needs_calibration)" in energy_map["text"]
    assert energy_map["flags"] == ["values_not_extracted"] and energy_map["plot_link"]["series"] == []
    assert inputs["plot_context_p"].name == "d_flat.tex"


def test_plot_chunks_need_one_intact_record_per_image(tmp_path):
    assert chunking.plot_chunks(None, [PAPER]) == ([], {})
    twice = _digitization_run(tmp_path, "twice", [_plot_record(ASSET_A, status="needs_calibration"),
                                                  _plot_record(ASSET_A, series=[_curve("s", None, 0)])])
    with pytest.raises(ValueError, match="recorded more than once"):
        chunking.plot_chunks(twice, [PAPER])
    digitized = _digitization_run(tmp_path, "digitized", [_plot_record(ASSET_A, series=[_curve("s", None, 0)])])
    with pytest.raises(ValueError, match="outside the parsed corpus"):
        chunking.plot_chunks(digitized, [])
    data = digitized / "digitized_data.jsonl"
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="intact digitization"):
        chunking.plot_chunks(digitized, [PAPER])


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
