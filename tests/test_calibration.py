"""Independent synthetic axes exercise proposals without model calls or paper-specific labels."""

import json
from pathlib import Path

import pytest

from kamiknows.plot_extractor.calibration import propose_pdf, run_calibration
from kamiknows.plot_extractor.validation import run_calibration_review
from kamiknows.plot_extractor.digitization import Calibration, vector_series, read_pdf


def make_pdf(path, log=False, units=True, second=False, tick_count=3, data=None):
    pdf = pytest.importorskip("pymupdf")
    with pdf.open() as doc:
        page = doc.new_page(width=320, height=240)
        page.draw_rect(pdf.Rect(60, 30, 270, 180), color=(0, 0, 0))
        if second:
            page.draw_rect(pdf.Rect(80, 50, 160, 110), color=(0, 0, 0))
        for position, value in list(zip([65, 165, 265], ["0", "5", "10"]))[:tick_count]:
            page.draw_line((position, 180), (position, 184), color=(0, 0, 0))
            width = pdf.get_text_length(value, fontsize=10)
            page.insert_text((position - width / 2, 195), value, fontsize=10)
        values = ["1e-1", "1e-2", "1e-3"] if log else ["10", "5", "0"]
        for position, value in zip([30, 105, 180], values):
            page.draw_line((60, position), (56, position), color=(0, 0, 0))
            width = pdf.get_text_length(value, fontsize=10)
            page.insert_text((53 - width, position + 3), value, fontsize=10)
        if units:
            page.insert_text((150, 218), "(m)", fontsize=10)
            page.insert_text((8, 110), "(s)", fontsize=10)
        shape = page.new_shape()
        if data is None:
            shape.draw_polyline([(65, 190), (65, 130), (130, 130), (130, 80), (265, 80), (265, 190)])
            shape.finish(color=(0, 0, 1), closePath=False)
        else:
            data(shape)
        shape.commit()
        page.draw_rect(pdf.Rect(175, 40, 255, 60), color=(0, 0, 0), fill=(1, 1, 1))
        doc.save(path)


@pytest.mark.parametrize("log", [False, True])
def test_pdf_proposal_recovers_independent_ticks_and_excludes_legend(tmp_path, log):
    path = tmp_path / "synthetic.pdf"
    make_pdf(path, log=log)
    original = path.read_bytes()
    config, evidence = propose_pdf(path)
    assert config is not None, evidence
    assert config["x"]["scale"] == "linear" and config["x"]["unit"] == "m"
    assert config["y"]["scale"] == ("log10" if log else "linear")
    assert config["y"]["unit"] == "s"
    assert len(config["series"]) == 1 and config["series"][0]["geometry"] == "step_histogram"
    cal = Calibration(config)
    paths, _, _ = read_pdf(path, cal, 72)
    series = vector_series(paths, cal)[0]
    assert series["rendered_intervals"][0] == pytest.approx([0, 3.25])
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"second": True}, "no_unique_single_panel_frame"),
        ({"tick_count": 2}, "axes_need_review"),
    ],
)
def test_ambiguous_axes_are_not_promoted(tmp_path, kwargs, reason):
    path = tmp_path / "synthetic.pdf"
    make_pdf(path, **kwargs)
    config, evidence = propose_pdf(path)
    assert config is None and evidence["reason"] == reason


def test_calibration_run_provenance_and_consumable_config(tmp_path):
    from kamiknows import parsing
    from kamiknows.common import write_manifest, write_json, write_jsonl
    from kamiknows.plot_extractor import run_inventory, run_classification, run_digitization

    parsed = tmp_path / "parsed"
    root = parsed / "papers" / "synthetic" / "source"
    root.mkdir(parents=True)
    make_pdf(root / "plot.pdf")
    tex = r"\section{Results}\begin{figure}\includegraphics{plot.pdf}\caption{One-dimensional histogram.}\end{figure}"
    (root / "main.tex").write_text(tex)
    (root.parent / "flat.tex").write_text(tex)
    sections, _, _ = parsing.parse_latex(tex)
    write_json(root.parent / "sections.json", sections)
    write_jsonl(parsed / "papers.jsonl", [{"paper_id": "synthetic", "status": "success"}])
    write_manifest(parsed, step="parse", params={}, inputs={}, summary={})
    inv = tmp_path / "inv"
    classified = tmp_path / "classified"
    run_inventory(parsed_dir=parsed, output_dir=inv)
    run_classification(inventory_dir=inv, output_dir=classified)
    result = run_calibration(classified_dir=classified, parsed_dir=parsed, output_dir=tmp_path / "cal")
    assert result["status_counts"] == {"proposed": 1}
    assert result["decision_mode"] == "automatic"
    config = tmp_path / "cal" / "calibration_proposals.json"
    summary = run_digitization(
        classified_dir=classified,
        parsed_dir=parsed,
        output_dir=tmp_path / "digitized",
        backend="pymupdf",
        calibration_file=config,
    )
    assert summary["extraction_status_counts"] == {"success": 1}
    before = (tmp_path / "cal" / "calibration_diagnostics.jsonl").read_bytes()
    review = run_calibration_review(calibration_dir=tmp_path / "cal", output_dir=tmp_path / "review")
    assert review["rendered"] == 1 and review["render_errors"] == 0
    page = (tmp_path / "review" / "review.html").read_text()
    assert "data:image/jpeg;base64," in page and "axis X" in page and "X1" in page
    assert "synthetic::figure::0001" in page
    assert (tmp_path / "cal" / "calibration_diagnostics.jsonl").read_bytes() == before
    (root / "plot.pdf").write_bytes(b"changed")
    review = run_calibration_review(calibration_dir=tmp_path / "cal", output_dir=tmp_path / "review_changed")
    assert review["rendered"] == 0 and review["render_errors"] == 1
    result = run_calibration(classified_dir=classified, parsed_dir=parsed, output_dir=tmp_path / "bad")
    assert result["status_counts"] == {"error": 1} and result["proposals"] == 0


def test_numeric_axes_without_units_are_calibrated_as_unknown(tmp_path):
    path = tmp_path / "no_units.pdf"
    make_pdf(path, units=False)
    config, evidence = propose_pdf(path)
    assert config is not None, evidence
    for axis in ("x", "y"):
        assert config[axis]["unit"] == "unknown" and config[axis]["unit_status"] == "missing"
        assert evidence[axis]["unit_status"] == "missing"
    Calibration(config)


@pytest.mark.parametrize(
    "state,status", [("readable", None), ("absent", "excluded_no_numbers"), ("unclear", "needs_review_axes")]
)
def test_visual_numeric_gate_requires_each_axis_and_ignores_units(state, status):
    from kamiknows.plot_extractor.calibration import visual_numeric_gate

    region = {
        "analysis_type": "histogram_2d",
        "axes": {
            name: {"presence": "present", "numeric_ticks": "readable", "unit_label": "absent"}
            for name in ("x", "y", "colorbar")
        },
    }
    region["axes"]["x"]["numeric_ticks"] = state
    outcome, reasons = visual_numeric_gate([region], "histogram_2d")
    assert outcome == status
    assert bool(reasons) == (status is not None)
    other = __import__("copy").deepcopy(region)
    other["axes"]["colorbar"]["numeric_ticks"] = "absent"
    assert visual_numeric_gate([region, other], "histogram_2d")[0] == "excluded_no_numbers"
    assert visual_numeric_gate([], "histogram_2d")[0] == "needs_review_axes"


@pytest.mark.parametrize(
    "numbers,ticks,status",
    [
        ("readable", 3, "proposed"),
        ("absent", 3, "proposed"),
        ("unclear", 3, "proposed"),
        ("absent", 0, "needs_review_axes"),
        ("readable", 0, "needs_review_axes"),
    ],
)
@pytest.mark.parametrize("role", ["plot", "mixed", "visual_comparison"])
def test_calibration_compares_pdf_with_gemma(tmp_path, monkeypatch, numbers, ticks, status, role):
    from kamiknows import parsing
    from kamiknows.common import write_manifest, write_json, write_jsonl
    from kamiknows.plot_extractor import run_inventory, run_classification

    parsed = tmp_path / "parsed"
    root = parsed / "papers" / "synthetic" / "source"
    root.mkdir(parents=True)
    make_pdf(root / "plot.pdf", units=False, tick_count=ticks)
    tex = r"\section{Results}\begin{figure}\includegraphics{plot.pdf}\caption{Histogram.}\end{figure}"
    (root / "main.tex").write_text(tex)
    (root.parent / "flat.tex").write_text(tex)
    sections, _, _ = parsing.parse_latex(tex)
    write_json(root.parent / "sections.json", sections)
    write_jsonl(parsed / "papers.jsonl", [{"paper_id": "synthetic", "status": "success"}])
    write_manifest(parsed, step="parse", params={}, inputs={}, summary={})
    inv = tmp_path / "inv"
    classified = tmp_path / "classified"
    run_inventory(parsed_dir=parsed, output_dir=inv)
    run_classification(inventory_dir=inv, output_dir=classified)
    axes = {
        a: {"presence": "present", "numeric_ticks": "readable", "unit_label": "absent"}
        for a in ("x", "y", "colorbar")
    }
    axes["x"]["numeric_ticks"] = numbers
    hints = tmp_path / "gemma.jsonl"
    write_jsonl(
        hints,
        [
            {
                "figure_id": "synthetic::figure::0001",
                "prediction": {
                    "asset_role": role,
                    "analysis_type": "histogram_1d",
                    "regions": [{"asset_number": 1, "analysis_type": "histogram_1d", "axes": axes}],
                },
            }
        ],
    )
    out = tmp_path / "cal"
    summary = run_calibration(
        classified_dir=classified, parsed_dir=parsed, output_dir=out, visual_predictions=hints
    )
    effective_status = "skipped_class" if role == "visual_comparison" and status != "proposed" else status
    assert summary["status_counts"] == {effective_status: 1}
    proposals = json.loads((out / "calibration_proposals.json").read_text())["plots"]
    assert len(proposals) == (1 if status == "proposed" else 0)
    page = (out / "review.html").read_text()
    assert "no-numbers" in page and "uncertain-numbers" in page
    if proposals:
        assert proposals[0]["x"]["unit"] == "unknown"
        assert proposals[0].get("classification_review_required", False) == (
            role in {"mixed", "visual_comparison"}
        )


def test_axis_calibration_does_not_require_series_selection(tmp_path, monkeypatch):
    path = tmp_path / "known_axes.pdf"
    make_pdf(path)
    monkeypatch.setattr("kamiknows.plot_extractor.calibration.infer_series", lambda *a: [])
    config, evidence = propose_pdf(path)
    assert config is not None and config["series"] == []
    assert config["digitization_readiness"] == "series_selection_required"
    assert evidence["reason"] == "axes_calibrated_series_selection_required"
    assert config["x"]["unit"] == "m" and config["y"]["unit"] == "s"


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_display_orientation_preserves_rotated_crop_and_source(tmp_path, rotation):
    import io
    import numpy as np
    from PIL import Image

    pdf = pytest.importorskip("pymupdf")
    path = tmp_path / "rotated.pdf"
    with pdf.open() as doc:
        page = doc.new_page(width=420, height=360)
        page.draw_rect(pdf.Rect(30, 40, 150, 170), fill=(1, 0, 0), color=None)
        page.draw_rect(pdf.Rect(210, 150, 330, 270), fill=(0, 0, 1), color=None)
        page.set_cropbox(pdf.Rect(20, 20, 350, 300))
        page.set_rotation(rotation)
        original = page.get_pixmap()
        expected = (
            np.frombuffer(original.samples, np.uint8).reshape(original.height, original.width, 3).copy()
        )
        width, height = page.rect.width, page.rect.height
        doc.save(path)
    before = path.read_bytes()
    config = {
        "roi": [0, 0, width, height],
        "coordinate_system": "pdf_points",
        "pdf_orientation": "display",
        "page": 1,
        "x": {
            "scale": "linear",
            "unit": "unknown",
            "anchors": [{"position": 0, "value": 0}, {"position": width, "value": 1}],
        },
        "y": {
            "scale": "linear",
            "unit": "unknown",
            "anchors": [{"position": 0, "value": 1}, {"position": height, "value": 0}],
        },
    }
    _, metadata, preview = read_pdf(path, Calibration(config), 72)
    actual = np.asarray(Image.open(io.BytesIO(preview)))
    assert actual.shape == expected.shape
    assert np.mean(np.abs(actual.astype(float) - expected.astype(float))) < 0.5
    assert metadata["render_orientation"] == "display" and metadata["pdf_rotation"] == rotation
    assert path.read_bytes() == before


@pytest.mark.parametrize("minor_ticks", [True, False])
def test_two_log_labels_require_minor_tick_evidence_and_axis_multiplier(tmp_path, minor_ticks):
    import math

    pdf = pytest.importorskip("pymupdf")
    path = tmp_path / "scientific_axes.pdf"
    with pdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.draw_rect(pdf.Rect(80, 40, 280, 220), color=(0, 0, 0))
        for x, text in [(80, "0"), (180, "50"), (280, "100")]:
            page.draw_line((x, 220), (x, 224), color=(0, 0, 0))
            page.insert_text((x - pdf.get_text_length(text, fontsize=10) / 2, 238), text, fontsize=10)
        page.insert_text((290, 235), "×10", fontsize=10)
        page.insert_text((308, 231), "3", fontsize=7)
        for y in (40, 220):
            page.draw_line((80, y), (75, y), color=(0, 0, 0))
        page.insert_text((50, 43), "10", fontsize=10)
        page.insert_text((61, 39), "2", fontsize=7)
        page.insert_text((50, 223), "10", fontsize=10)
        if minor_ticks:
            for value in range(20, 100, 10):
                y = 220 - 180 * math.log10(value / 10)
                page.draw_line((80, y), (77, y), color=(0, 0, 0))
        doc.save(path)
    config, evidence = propose_pdf(path)
    if minor_ticks:
        assert config is not None, evidence
        assert config["y"]["scale"] == "log10" and evidence["y"]["log_minor_tick_support"] == 8
        assert config["x"]["display_multiplier"] == 1000 and config["x"]["anchors"][-1]["value"] == 100000
    else:
        assert config is None


def test_unit_superscript_is_not_a_tick_label():
    from kamiknows.plot_extractor.calibration import _numeric_labels

    words = [(10, 20, 30, 40, "[mm", 1, 0, 0), (30, 15, 35, 22, "2", 1, 0, 1), (50, 15, 60, 30, "2", 2, 0, 0)]
    labels = _numeric_labels(words, [])
    assert len(labels) == 1 and labels[0]["box"] == [50, 15, 60, 30]


def test_split_minus_and_same_block_axis_units_are_not_exponents():
    from kamiknows.plot_extractor.calibration import _numeric_labels

    words = [
        (10, 100, 22, 130, "−", 0, 0, 0),
        (22, 100, 48, 132, "20", 0, 0, 1),
        (100, 100, 126, 132, "40", 0, 0, 2),
        (150, 120, 220, 158, "[GeV]", 0, 1, 0),
    ]
    labels = _numeric_labels(words, [])
    assert [v["value"] for v in labels] == [-20.0, 40.0]
    assert labels[0]["box"][0] == 10


def test_numeric_label_matches_nearest_tick_not_first_nearby_tick():
    pdf = pytest.importorskip("pymupdf")
    from kamiknows.plot_extractor.calibration import infer_axis

    roi = [80, 20, 280, 220]
    paths = []
    words = []
    for y, value in [(20, "5"), (70, "4"), (120, "3"), (170, "2"), (220, "1")]:
        for position in [y - 10, y]:
            if position < 20:
                continue
            paths.append(
                {"color": (0, 0, 0), "items": [("l", pdf.Point(80, position), pdf.Point(85, position))]}
            )
        words.append((50, y - 11, 70, y + 9, value, 0, 0, 0))
    config, evidence = infer_axis(paths, words, roi, "y")
    assert config is not None, evidence
    assert [a["position"] for a in config["anchors"]] == [20.0, 70.0, 120.0, 170.0, 220.0]


def test_filled_minus_glyph_is_retained_in_decimal_label():
    pdf = pytest.importorskip("pymupdf")
    from kamiknows.plot_extractor.calibration import _numeric_labels

    words = [(32, 90, 55, 110, "0.1", 0, 0, 0)]
    paths = [{"rect": pdf.Rect(20, 99, 30, 101), "fill": (0, 0, 0), "items": []}]
    labels = _numeric_labels(words, paths)
    assert labels[0]["value"] == -0.1 and labels[0]["box"][0] == 20


def test_power_label_alignment_uses_exponent_width():
    from kamiknows.plot_extractor.calibration import _numeric_labels

    labels = _numeric_labels([(0, 0, 10, 12, "10", 0, 0, 0), (10, 0, 16, 8, "-3", 0, 0, 1)], [])
    assert len(labels) == 1 and labels[0]["value"] == 0.001
    assert labels[0]["box"][2] == 16


def test_major_tick_geometry_wins_over_closer_minor_ticks():
    pdf = pytest.importorskip("pymupdf")
    from kamiknows.plot_extractor.calibration import infer_axis

    paths = []
    words = []
    for y, value in [(20, "3"), (70, "2"), (120, "1")]:
        for position, length in [(y, 8), (y + 5, 4)]:
            paths.append(
                {
                    "color": (0, 0, 0),
                    "items": [("l", pdf.Point(80, position), pdf.Point(80 + length, position))],
                }
            )
        words.append((50, y - 6, 70, y + 14, value, 0, 0, 0))
    config, evidence = infer_axis(paths, words, [80, 20, 280, 220], "y")
    assert config is not None, evidence
    assert [a["position"] for a in config["anchors"]] == [20.0, 70.0, 120.0]


STEPS = [(70, 200), (70, 100), (110, 100), (110, 120), (150, 120), (150, 200)]
LATER = [(190, 200), (190, 60), (230, 60), (230, 200)]


def draw_steps(filled):
    """A histogram that drops below the frame (bottom at y=180): in two pieces, or filled with its outline."""

    def draw(shape):
        if filled:
            shape.draw_polyline(STEPS + LATER + [(70, 200)])
            shape.finish(fill=(0.8, 0.2, 0.2), color=None, closePath=False)
            shape.draw_polyline(STEPS + LATER)
            shape.finish(color=(0.8, 0.2, 0.2), closePath=False)
        else:
            shape.draw_polyline(STEPS)
            shape.draw_polyline(LATER)
            shape.finish(color=(0, 0.5, 0), closePath=False)

    return draw


@pytest.mark.parametrize("filled", [False, True])
def test_histogram_in_pieces_or_filled_is_selected_once(tmp_path, filled):
    path = tmp_path / "steps.pdf"
    make_pdf(path, data=draw_steps(filled))
    config, evidence = propose_pdf(path)
    assert config is not None, evidence
    assert len(config["series"]) == 1 and config["series"][0]["geometry"] == "step_histogram"
    cal = Calibration(config)
    paths, _, _ = read_pdf(path, cal, 72)
    assert paths[config["series"][0]["path_indices"][0]].get("fill") is None  # the outline, not the fill
    series = vector_series(paths, cal)[0]
    flat = [v for interval in series["rendered_intervals"] for v in interval]
    assert flat == pytest.approx([0.25, 2.25, 2.25, 4.25, 6.25, 8.25])
    below = [v for interval in series["below_range_intervals"] for v in interval]
    assert below == (pytest.approx([4.25, 6.25]) if filled else [])


def test_ocr_axes_still_read_vector_curves(tmp_path):
    from kamiknows.plot_extractor.calibration import _with_vector_series

    path = tmp_path / "synthetic.pdf"
    make_pdf(path)
    config, _ = propose_pdf(path)
    ocr = {**config, "series": [], "calibration_source": "ocr_axis_ticks",
           "raster_extraction": {"kind": "colored_traces", "method": "pixel_geometry_v1"}}
    native = _with_vector_series(path, ocr)
    assert "raster_extraction" not in native and native["series"][0]["geometry"] == "step_histogram"
    assert native["calibration_source"] == "ocr_axis_ticks"
    pixels = {**ocr, "coordinate_system": "pixels"}
    assert _with_vector_series(path, pixels) is pixels


def test_false_2d_suggestion_can_use_verified_1d_series(tmp_path):
    path = tmp_path / "steps.pdf"
    make_pdf(path)
    config, evidence = propose_pdf(path, "histogram_2d")
    assert config is not None and config["series"]
    assert evidence["calibration_type"] == "histogram_1d"
    assert evidence["suggested_type"] == "histogram_2d"


@pytest.mark.parametrize("partial", [False, True])
def test_disjoint_1d_panels_have_independent_calibrations(tmp_path, partial):
    pdf = pytest.importorskip("pymupdf")
    left = tmp_path / "left.pdf"
    right = tmp_path / "right.pdf"
    make_pdf(left)
    make_pdf(right, tick_count=2 if partial else 3)
    target = tmp_path / "panels.pdf"
    with pdf.open(left) as a, pdf.open(right) as b, pdf.open() as doc:
        page = doc.new_page(width=700, height=260)
        page.show_pdf_page(pdf.Rect(0, 0, 320, 240), a, 0)
        page.show_pdf_page(pdf.Rect(350, 0, 670, 240), b, 0)
        doc.save(target)
    config, evidence = propose_pdf(target, "histogram_1d")
    assert config is not None, evidence
    assert config["panel_count"] == 2 and len(config["panels_1d"]) == (1 if partial else 2)
    assert config["digitization_readiness"] == "panel_digitization_required"
    assert config["calibration_completeness"] == ("partial" if partial else "complete")
    if partial:
        assert config["uncalibrated_panels"][0]["panel_index"] == 2
    else:
        assert config["panels_1d"][1]["roi"][0] > config["panels_1d"][0]["roi"][2]


def test_positioned_glyph_gap_splits_adjacent_axis_labels(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    from kamiknows.plot_extractor.calibration import _axis_words, _numeric_labels

    with pdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((80, 100), "10", fontsize=20)
        page.insert_text((81 + pdf.get_text_length("10", fontsize=20), 100), "11", fontsize=20)
        page.insert_text((80, 150), "1000", fontsize=20)
        words = _axis_words(page)
        assert [w[4] for w in words if w[1] < 100] == ["10", "11"]
        assert "1000" in [w[4] for w in words]
        labels = _numeric_labels(words, [])
        assert {v["value"] for v in labels} == {10.0, 11.0, 1000.0}
        assert sum(v.get("text_segmentation") is not None for v in labels) == 2


@pytest.mark.parametrize("numeric", [True, False])
def test_model_diagram_skip_requires_independent_pdf_audit(tmp_path, numeric):
    from kamiknows import parsing
    from kamiknows.common import write_manifest, write_json, write_jsonl
    from kamiknows.plot_extractor import run_inventory, run_classification, run_digitization

    parsed = tmp_path / "parsed"
    root = parsed / "papers" / "synthetic" / "source"
    root.mkdir(parents=True)
    if numeric:
        make_pdf(root / "plot.pdf")
    else:
        pdf = pytest.importorskip("pymupdf")
        with pdf.open() as doc:
            page = doc.new_page(width=320, height=240)
            page.draw_rect(pdf.Rect(60, 30, 270, 180), color=(0, 0, 0))
            page.insert_text((100, 100), "Input to output")
            doc.save(root / "plot.pdf")
    tex = r"\section{Results}\begin{figure}\includegraphics{plot.pdf}\caption{Histogram.}\end{figure}"
    (root / "main.tex").write_text(tex)
    (root.parent / "flat.tex").write_text(tex)
    sections, _, _ = parsing.parse_latex(tex)
    write_json(root.parent / "sections.json", sections)
    write_jsonl(parsed / "papers.jsonl", [{"paper_id": "synthetic", "status": "success"}])
    write_manifest(parsed, step="parse", params={}, inputs={}, summary={})
    inv = tmp_path / "inv"
    classified = tmp_path / "classified"
    run_inventory(parsed_dir=parsed, output_dir=inv)
    run_classification(inventory_dir=inv, output_dir=classified)
    hints = tmp_path / "gemma.jsonl"
    write_jsonl(
        hints,
        [
            {
                "figure_id": "synthetic::figure::0001",
                "prediction": {"asset_role": "diagram", "analysis_type": "unknown", "regions": []},
            }
        ],
    )
    out = tmp_path / "cal"
    summary = run_calibration(
        classified_dir=classified, parsed_dir=parsed, output_dir=out, visual_predictions=hints
    )
    expected = "proposed_class_conflict" if numeric else "needs_review_classification"
    assert summary["status_counts"] == {expected: 1}
    proposals = json.loads((out / "calibration_proposals.json").read_text())["plots"]
    if numeric:
        assert proposals[0]["classification_review_required"] is True
        result = run_digitization(
            classified_dir=classified,
            parsed_dir=parsed,
            output_dir=tmp_path / "digitized",
            backend="pymupdf",
            calibration_file=out / "calibration_proposals.json",
        )
        assert result["extraction_status_counts"] == {"needs_manual_review": 1}
    else:
        assert proposals == []


def test_overprinted_numeric_labels_are_one_observation():
    from kamiknows.plot_extractor.calibration import _numeric_labels

    words = [(10, 20, 20, 30, "100", 0, 0, 0)] * 4
    assert len(_numeric_labels(words, [])) == 1
    # Conflicting text at the same location must remain ambiguous.
    assert len(_numeric_labels(words + [(10, 20, 20, 30, "200", 0, 0, 0)], [])) == 2


def test_ocr_keeps_text_unverified_without_publishing_anchors(tmp_path, monkeypatch):
    from PIL import Image
    from kamiknows.plot_extractor import calibration

    pytest.importorskip("cv2")
    path = tmp_path / "pixels.png"
    Image.new("RGB", (120, 100), "white").save(path)
    monkeypatch.setattr(
        calibration,
        "_ocr_engine",
        lambda: lambda image: ([([[1, 2], [10, 2], [10, 12], [1, 12]], "-0.1", 0.95)], None),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    evidence = calibration.ocr_axis_candidates(path)
    assert evidence["observations"][0]["text"] == "-0.1"
    assert evidence["numeric_verified"] is False
    assert evidence["observations"][0]["verified"] is False
    assert evidence["coordinate_system"] == "pixels"
    assert not evidence["panels"]


def test_ocr_superscript_requires_raised_glyphs():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from kamiknows.plot_extractor.calibration import _ocr_power_notation, _numeric_labels

    raised = np.full((90, 120, 3), 255, dtype=np.uint8)
    cv2.putText(raised, "10", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
    cv2.putText(raised, "3", (65, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    assert _ocr_power_notation(raised, [0, 0, 120, 90], "103") == "10^3"
    plain = np.full_like(raised, 255)
    cv2.putText(plain, "103", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
    assert _ocr_power_notation(plain, [0, 0, 120, 90], "103") == "103"
    labels = _numeric_labels([(0, 0, 20, 15, "10^-3", 0, 0, 0)], [])
    assert labels[0]["value"] == pytest.approx(0.001) and labels[0]["power10"]


def test_raster_tick_centers_survive_thick_spines():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from kamiknows.plot_extractor.calibration import _raster_axis_ticks, _ticks

    mask = np.zeros((200, 240), dtype=np.uint8)
    cv2.rectangle(mask, (30, 20), (200, 160), 255, 3)
    for x in [60, 110, 160]:
        cv2.line(mask, (x, 160), (x, 168), 255, 3)
    boxes, paths = _raster_axis_ticks(mask, [[28, 18, 202, 162], [32, 22, 198, 158], [28, 18, 202, 170]])
    assert len(boxes) == 1
    assert _ticks(paths, boxes[0], "x") == pytest.approx([60, 110, 160], abs=1)


def test_ocr_unit_label_does_not_suppress_unrelated_integer_ticks(tmp_path, monkeypatch):
    from kamiknows.plot_extractor import calibration

    pdf = pytest.importorskip("pymupdf")
    pytest.importorskip("cv2")
    path = tmp_path / "axis.pdf"
    make_pdf(path, units=False)
    with pdf.open(path) as document:
        words = document[0].get_text("words")
    observations = []
    for x0, y0, x1, y1, text, *rest in words:
        observations.append(
            ([[2 * x0, 2 * y0], [2 * x1, 2 * y0], [2 * x1, 2 * y1], [2 * x0, 2 * y1]], text, 0.99)
        )
    observations.append(([[0, 0], [20, 0], [20, 150], [0, 150]], "[MeV]", 0.99))
    monkeypatch.setattr(
        calibration,
        "_ocr_engine",
        lambda: lambda image: (observations if image.shape[1] == 640 else [], None),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    result = calibration.ocr_axis_candidates(path)
    assert result["panels"][0]["x"]["scale"] == "linear"
    assert len(result["panels"][0]["x"]["anchors"]) == 3


def test_raster_recipe_requires_complete_three_label_axis_fits():
    from kamiknows.plot_extractor.calibration import raster_calibration_proposal

    axis = {"scale": "linear", "unit": "", "anchors": [{"position": v, "value": v} for v in [10, 50, 90]]}
    info = {"reason": "axis_ticks_consistent", "anchors": axis["anchors"]}
    panel = {"roi": [10, 10, 90, 90], "x": info, "y": info, "axis_configs": {"x": axis, "y": axis}}
    evidence = {"coordinate_system": "pixels", "panels": [panel]}
    assert raster_calibration_proposal(evidence, "line_plot")["raster_extraction"]["kind"] == "colored_traces"
    assert raster_calibration_proposal(evidence, "histogram_2d") is None
    panel["x"] = {**info, "anchors": axis["anchors"][:2]}
    assert raster_calibration_proposal(evidence, "line_plot") is None
    panel["x"] = info
    evidence["panels"] = [panel, panel]
    assert raster_calibration_proposal(evidence, "line_plot") is None


def test_ocr_detects_axes_in_bitmap_only_pdf(tmp_path, monkeypatch):
    from kamiknows.plot_extractor import calibration

    pdf = pytest.importorskip("pymupdf")
    pytest.importorskip("cv2")
    original = tmp_path / "vector.pdf"
    make_pdf(original, units=False)
    pixels = tmp_path / "pixels.png"
    with pdf.open(original) as doc:
        doc[0].get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False).save(pixels)
        words = doc[0].get_text("words")
    bitmap = tmp_path / "bitmap.pdf"
    with pdf.open() as doc:
        page = doc.new_page(width=320, height=240)
        page.insert_image(page.rect, filename=str(pixels))
        doc.save(bitmap)
    observations = [
        ([[2 * a, 2 * b], [2 * c, 2 * b], [2 * c, 2 * d], [2 * a, 2 * d]], text, 0.99)
        for a, b, c, d, text, *rest in words
    ]
    monkeypatch.setattr(
        calibration,
        "_ocr_engine",
        lambda: lambda image: (observations if image.shape[1] == 640 else [], None),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    evidence = calibration.ocr_axis_candidates(bitmap)
    assert evidence["coordinate_system"] == "pdf_points"
    proposal = calibration.raster_calibration_proposal(evidence, "histogram_1d")
    assert proposal is not None, evidence
    assert proposal["x"]["scale"] == "linear" and proposal["y"]["scale"] == "linear"
