"""Offline visual fixtures; optional installed-engine tests are identified separately."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kamiknows.common import read_json, read_jsonl, sha256_file, write_json, write_jsonl, write_manifest
from kamiknows.plot_extractor.digitization import (
    Axis,
    Backend,
    Calibration,
    ExtractionError,
    vector_series,
)
from kamiknows.plot_extractor.records import PlotDataRecord
from kamiknows.plot_extractor import run_benchmark, run_classification, run_digitization, run_inventory
from kamiknows.plot_extractor.validation import compare_series


def config(system="pdf_points"):
    return {
        "figure_id": "fig",
        "panel_id": None,
        "asset_id": "asset",
        "page": 1,
        "roi": [0, 0, 100, 100],
        "coordinate_system": system,
        "calibration_source": "axis_ticks",
        "manual_interventions": ["selected axes and paths"],
        "x": {
            "scale": "linear",
            "anchors": [{"position": 0, "value": 0}, {"position": 100, "value": 10}],
            "unit": "GeV",
        },
        "y": {
            "scale": "linear",
            "anchors": [{"position": 0, "value": 10}, {"position": 100, "value": 0}],
            "unit": "events",
        },
        "series": [
            {
                "series_id": "s",
                "observable": "yield",
                "coordinate_definition": "x energy, y yield",
                "geometry": "line",
                "path_indices": [0],
            }
        ],
    }


def line(points):
    return {
        "items": [["l", a, b] for a, b in zip(points, points[1:])],
        "rect": [0, 0, 100, 100],
        "closePath": False,
    }


def rectangle(box):
    return {"items": [["re", box, 1]], "rect": box, "closePath": True}


def test_schema_preserves_12():
    r = PlotDataRecord(None, None, None, "d", "unavailable", {}, {}, {})
    assert "calibration" not in r.to_dict()
    r.schema_version = "1.3"
    assert r.to_dict()["calibration"] == {}


def test_calibration_linear_log_inversion_and_residuals():
    c = config()
    c["x"]["anchors"].append({"position": 50, "value": 5})
    c["y"].update(scale="log10", anchors=[{"position": 0, "value": 10000}, {"position": 100, "value": 1}])
    cal = Calibration(c)
    assert cal.convert([[50, 50]]) == ([5], [100])
    assert cal.y.position(100) == 50
    assert cal.metadata()["fit"]["x"]["residuals"] == [0, 0, 0]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["x"].update(anchors=[{"position": 0, "value": 0}]),
        lambda c: c["x"]["anchors"][1].update(position=0),
        lambda c: c["x"]["anchors"][1].update(value=0),
        lambda c: c["x"].update(scale="log10"),
        lambda c: c["x"]["anchors"][1].update(position=101),
        lambda c: c["x"]["anchors"][1].update(value=float("nan")),
        lambda c: c["y"].update(direction="increasing"),
        lambda c: c["x"]["anchors"].append({"position": 50, "value": 6}),
        lambda c: c.update(page=0),
        lambda c: c.update(roi=[0, 0, 0, 100]),
    ],
)
def test_bad_calibration(mutation):
    c = config()
    mutation(c)
    with pytest.raises(ValueError):
        Calibration(c)


def test_vector_line_multiseries_and_bezier():
    c = config()
    c["series"].append({**c["series"][0], "series_id": "t", "path_indices": [1]})
    bez = {"items": [["c", [0, 100], [20, 80], [80, 20], [100, 0]]], "rect": [0, 0, 100, 100]}
    result = vector_series([line([[10, 90], [50, 50], [90, 10]]), bez], Calibration(c))
    assert result[0]["x"] == [1, 5, 9] and result[0]["y"] == [1, 5, 9]
    assert len(result[1]["x"]) == 17
    assert result[1]["sampling"]["original_measurement_points"] is False


def test_scatter_log_errors_and_histogram_band():
    c = config()
    c["series"][0].update(geometry="scatter", path_indices=[0], error_path_indices=[1])
    c["y"].update(scale="log10", anchors=[{"position": 0, "value": 10000}, {"position": 100, "value": 1}])
    result = vector_series([rectangle([48, 48, 52, 52]), line([[50, 25], [50, 75]])], Calibration(c))[0]
    assert result["y"] == [100]
    assert result["displayed_uncertainties"][0]["lower"] == 10
    assert result["displayed_uncertainties"][0]["upper"] == 1000
    c = config()
    c["series"][0].update(geometry="histogram", path_indices=[0, 1], baseline_position=100)
    result = vector_series([rectangle([0, 50, 20, 100]), rectangle([20, 20, 40, 100])], Calibration(c))[0]
    assert result["bins"] == [[0, 2], [2, 4]] and result["y"] == [5, 8]
    c["series"][0].update(geometry="band")
    result = vector_series([line([[10, 20], [50, 30]]), line([[10, 40], [50, 50]])], Calibration(c))[0]
    assert result["y"] == [None, None]
    assert result["displayed_uncertainties"][0]["lower"] == 6


def test_ambiguous_hidden_stacked_and_outside():
    c = config()
    c["series"][0].update(geometry="scatter", path_indices=[0, 1])
    with pytest.raises(ExtractionError, match="overlapping"):
        vector_series([rectangle([10, 10, 16, 16]), rectangle([12, 12, 18, 18])], Calibration(c))
    c["series"][0].update(stacked=True)
    with pytest.raises(ExtractionError, match="stacked"):
        vector_series([rectangle([0, 0, 10, 10])] * 2, Calibration(c))
    c = config()
    with pytest.raises(ExtractionError, match="out-of-ROI"):
        vector_series([line([[0, 0], [101, 0]])], Calibration(c))
    c["series"] = []
    with pytest.raises(ExtractionError, match="path_indices"):
        vector_series([], Calibration(c))


def test_capabilities_and_missing_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert Backend("pymupdf").capabilities()["available"] is False
    with pytest.raises(ExtractionError, match="missing"):
        Backend("pymupdf").digitize(tmp_path / "a.pdf", Calibration(config()), tmp_path)


@pytest.fixture
def visual_run(tmp_path):
    """Known pixels, parsed source, inventory and classification with real manifests."""
    from PIL import Image, ImageDraw
    from kamiknows import parsing

    parsed = tmp_path / "parsed"
    root = parsed / "papers" / "synthetic" / "source"
    root.mkdir(parents=True)
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).line([(10, 90), (90, 10)], fill="black", width=2)
    image.save(root / "plot.png")
    latex = (
        r"\documentclass{article}\begin{document}\section{Results}\begin{figure}\includegraphics{plot.png}"
        r"\begin{axis}[x unit={GeV},y unit={events}]\addplot coordinates {(1,1) (5,5) (9,9)};\end{axis}"
        r"\caption{Energy distribution plot.}\end{figure}\end{document}"
    )
    (root / "main.tex").write_text(latex)
    flat = parsing.flatten_latex_file(root / "main.tex", root)
    (root.parent / "flat.tex").write_text(flat)
    sections, _, _ = parsing.parse_latex(flat)
    write_json(root.parent / "sections.json", sections)
    write_jsonl(parsed / "papers.jsonl", [{"paper_id": "synthetic", "status": "success"}])
    write_manifest(parsed, step="parse", params={}, inputs={}, summary={})
    inv = tmp_path / "inventory"
    classified = tmp_path / "classified"
    run_inventory(parsed_dir=parsed, output_dir=inv)
    run_classification(inventory_dir=inv, output_dir=classified)
    figure = read_jsonl(classified / "figures.jsonl")[0]
    asset = figure["source_assets"][0]
    c = config("pixels")
    c.update(
        figure_id=figure["figure_id"],
        asset_id=asset["asset_id"],
        raster_extraction={"kind": "colored_traces", "method": "pixel_geometry_v1"},
    )
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"schema_version": "1", "plots": [c]})
    return parsed, classified, figure, asset, c, calibration


def install_fake_raster(monkeypatch):
    from kamiknows.plot_extractor.digitization import series_record

    def fake(self, path, calibration, work_dir, dpi=144):
        from PIL import Image

        # Like the real raster backend, keep the rendered source next to the overlay.
        Image.new("RGB", (100, 100), "white").save(work_dir / "source.png")
        return [
            series_record(
                calibration.config["series"][0], [[10, 90], [50, 50], [90, 10]], calibration, "marker_centers"
            )
        ], {"render_scale": 1}

    monkeypatch.setattr(Backend, "digitize", fake)


def test_pipeline_benchmark_and_immutability(visual_run, tmp_path, monkeypatch):
    parsed, classified, figure, asset, c, calibration = visual_run
    before = {
        p: p.read_bytes()
        for base in (parsed, classified, tmp_path / "inventory")
        for p in base.rglob("*")
        if p.is_file()
    }
    install_fake_raster(monkeypatch)
    output = tmp_path / "digitized"
    summary = run_digitization(
        parsed_dir=parsed,
        classified_dir=classified,
        output_dir=output,
        backend="raster",
        calibration_file=calibration,
    )
    records = read_jsonl(output / "digitized_data.jsonl")
    assert summary["records"] == 1 and summary["extraction_status_counts"] == {"success": 1}
    success = next(r for r in records if r["extraction_status"] == "success")
    assert success["schema_version"] == "1.3" and success["source_method"] == "raster_digitized"
    assert (output / "overlays" / success["data_id"] / "overlay.svg").is_file()
    reference = {
        "series_id": "s",
        "observable": "yield",
        "coordinate_definition": "x energy, y yield",
        "units": {"x": "GeV", "y": "events"},
        "x": [9, 5, 1],
        "y": [9, 5, 1],
    }
    truth = tmp_path / "truth.json"
    write_json(
        truth,
        {
            "schema_version": "1",
            "references": [
                {
                    "figure_id": figure["figure_id"],
                    "asset_id": asset["asset_id"],
                    "panel_id": None,
                    "source_kind": "synthetic",
                    "independent": True,
                    "provenance": {"generator": "exact line y=x, axis ticks independent"},
                    "series": reference,
                    "comparison": {"tolerance": {"x": 0.01, "y": 0.01}},
                }
            ],
        },
    )
    results = run_benchmark(digitized_dir=output, output_dir=tmp_path / "benchmark", ground_truth_file=truth)
    assert (
        results["coverage"] == 1
        and results["abstention_rate"] == 0
        and results["mean_correct_fraction"] == 1
    )
    detail = next(
        d for d in read_jsonl(tmp_path / "benchmark" / "benchmark_details.jsonl") if d["status"] == "compared"
    )
    assert detail["pairs"] == [[0, 2], [1, 1], [2, 0]]
    assert detail["absolute_errors"]["y"]["p95"] == 0
    assert all(p.read_bytes() == v for p, v in before.items())
    with pytest.raises((ValueError, FileExistsError)):
        run_digitization(parsed_dir=parsed, classified_dir=classified, output_dir=output)


def test_missing_calibration_source_change_and_selection(visual_run, tmp_path, monkeypatch):
    parsed, classified, figure, asset, c, calibration = visual_run
    summary = run_digitization(
        parsed_dir=parsed, classified_dir=classified, output_dir=tmp_path / "missing", backend="raster"
    )
    assert summary["extraction_status_counts"] == {"needs_calibration": 1}
    (parsed / "papers" / "synthetic" / "source" / "plot.png").write_bytes(b"changed")
    summary = run_digitization(
        parsed_dir=parsed,
        classified_dir=classified,
        output_dir=tmp_path / "changed",
        calibration_file=calibration,
    )
    assert summary["extraction_status_counts"] == {"source_mismatch": 1}


def test_original_skip_override_and_recovery_benchmark(visual_run, tmp_path, monkeypatch):
    from kamiknows.plot_extractor import run_recovery

    parsed, classified, figure, asset, c, calibration = visual_run
    recovery = tmp_path / "recovery"
    run_recovery(parsed_dir=parsed, classified_dir=classified, output_dir=recovery)
    original = next(
        r for r in read_jsonl(recovery / "plot_data.jsonl") if r["source_method"] == "original_source_data"
    )
    install_fake_raster(monkeypatch)
    kwargs = dict(
        parsed_dir=parsed,
        classified_dir=classified,
        backend="raster",
        calibration_file=calibration,
        recovery_dir=recovery,
    )
    # Figure-level data alone does not prove that this included PNG represents it.
    assert run_digitization(**kwargs, output_dir=tmp_path / "unmapped")["extraction_status_counts"] == {
        "success": 1
    }
    c["original_data_ids"] = [original["data_id"]]
    c["benchmark_references"] = [
        {
            "data_id": original["data_id"],
            "source_series_id": original["series"][0]["series_id"],
            "series_id": "s",
            "observable": "yield",
            "units": {"x": "GeV", "y": "events"},
            "coordinate_definition": "x energy, y yield",
            "comparison": {"tolerance": {"x": 0.01, "y": 0.01}},
        }
    ]
    write_json(calibration, {"schema_version": "1", "plots": [c]})
    assert run_digitization(**kwargs, output_dir=tmp_path / "skipped")["extraction_status_counts"] == {
        "skipped_original": 1
    }
    output = tmp_path / "override"
    assert run_digitization(**kwargs, output_dir=output, selection="all")["extraction_status_counts"] == {
        "success": 1
    }
    benchmark = run_benchmark(
        digitized_dir=output, recovery_dir=recovery, output_dir=tmp_path / "native_benchmark"
    )
    assert benchmark["mean_correct_fraction"] == 1 and benchmark["coverage"] == 1


def test_original_hepdata_bins_errors_and_unit_mismatch():
    from kamiknows.plot_extractor.recovery import parse_hepdata_table
    from kamiknows.plot_extractor.validation import original_series

    table = {
        "independent_variables": [
            {"header": {"name": "Energy", "units": "GeV"}, "values": [{"low": 1, "high": 3}]}
        ],
        "dependent_variables": [
            {
                "header": {"name": "yield", "units": "events"},
                "values": [{"value": 10, "errors": [{"label": "stat", "symerror": "10%"}]}],
            }
        ],
    }
    axes, series, _ = parse_hepdata_table(table)
    mapping = {
        "series_id": "s",
        "observable": "yield",
        "coordinate_definition": "xy",
        "units": {"x": "GeV", "y": "events"},
        "uncertainty_label": "stat",
    }
    result = original_series({"backend": "hepdata", "axes": axes}, series[0], mapping)
    assert result["bins"] == [[1, 3]] and result["x"] == [2]
    assert result["displayed_uncertainties"] == [{"lower": 9, "upper": 11, "semantics": "statistical"}]
    with pytest.raises(ValueError, match="units"):
        original_series(
            {"backend": "hepdata", "axes": axes}, series[0], {**mapping, "units": {"x": "MeV", "y": "events"}}
        )


def test_benchmark_leakage_and_invalid_reference(visual_run, tmp_path, monkeypatch):
    parsed, classified, figure, asset, c, calibration = visual_run
    install_fake_raster(monkeypatch)
    c["calibration_source"] = "evaluation_points"
    write_json(calibration, {"schema_version": "1", "plots": [c]})
    output = tmp_path / "digitized"
    run_digitization(
        parsed_dir=parsed,
        classified_dir=classified,
        output_dir=output,
        backend="raster",
        calibration_file=calibration,
    )
    reference = {
        "figure_id": figure["figure_id"],
        "asset_id": asset["asset_id"],
        "panel_id": None,
        "source_kind": "synthetic",
        "independent": True,
        "provenance": {"generator": "known y=x"},
        "series": {
            "series_id": "s",
            "x": [1, 5, 9],
            "y": [1, 5, 9],
            "observable": "yield",
            "coordinate_definition": "x energy, y yield",
            "units": {"x": "GeV", "y": "events"},
        },
        "comparison": {"tolerance": {"x": 0.01, "y": 0.01}},
    }
    truth = tmp_path / "truth.json"
    write_json(truth, {"schema_version": "1", "references": [reference]})
    result = run_benchmark(digitized_dir=output, ground_truth_file=truth, output_dir=tmp_path / "benchmark")
    assert result["compared"] == 0 and result["abstention_rate"] == 1
    reference["source_kind"] = "raster_digitized"
    write_json(truth, {"schema_version": "1", "references": [reference]})
    with pytest.raises(ValueError, match="independent"):
        run_benchmark(digitized_dir=output, ground_truth_file=truth, output_dir=tmp_path / "invalid")


def test_bin_edge_matching_and_endpoint_metrics():
    s = {
        "series_id": "s",
        "observable": "yield",
        "coordinate_definition": "bins",
        "units": {"x": "", "y": ""},
        "x": [1, 3],
        "y": [5, 7],
        "bins": [[0, 2], [2, 4]],
        "displayed_uncertainties": [
            {"lower": 4, "upper": 6, "semantics": "unlabeled_displayed_error"},
            {"lower": 6, "upper": 8, "semantics": "unlabeled_displayed_error"},
        ],
    }
    t = copy.deepcopy(s)
    t["bins"][0] = [-1, 3]
    result = compare_series(s, t, {"tolerance": {"x": 0.1, "y": 0.1}})
    assert result["matching_bin_fraction"] == 0.5 and result["missing_bin_fraction"] == 0.5
    assert result["endpoint_errors"]["lower"]["max"] == 0


def test_metrics_false_missing_nearzero_and_interpolation():
    a = {
        "series_id": "s",
        "observable": "yield",
        "coordinate_definition": "xy",
        "units": {"x": "", "y": ""},
        "x": [0, 1, 9],
        "y": [0, 1.1, 9],
    }
    b = {**a, "x": [0, 1, 2], "y": [0, 1, 2]}
    result = compare_series(a, b, {"tolerance": {"x": 0.2, "y": 0.2}})
    assert result["correct_fraction"] == 2 / 3 and result["false_detections"] == 1
    assert result["near_zero_relative_omitted"] == {"x": 1, "y": 1}
    result = compare_series(
        {**a, "x": [0, 2], "y": [0, 2]},
        b,
        {"method": "linear_interpolation", "domain": [0, 2], "samples": 9, "tolerance": {"x": 0.1, "y": 0.1}},
    )
    assert result["absolute_errors"]["y"]["max"] == 0 and result["false_detections"] is None
    assert compare_series(a, {**b, "observable": "other"}, {})["status"] == "not_comparable"


def test_cli_modes_and_strict_options(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "extract_plots.py"
    for arguments in (
        ["inventory", "--parsed-dir", "unused", "--backend", "auto"],
        ["benchmark", "--digitized-dir", "unused"],
        ["digitize", "--classified-dir", "unused", "--parsed-dir", "unused", "--raster-dpi", "500"],
    ):
        completed = subprocess.run(
            [sys.executable, str(script), "--output-dir", str(tmp_path / "out"), *arguments],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2


def test_optional_pymupdf_integration(tmp_path):
    pymupdf = pytest.importorskip("pymupdf", reason="optional real PyMuPDF not installed")
    path = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=100, height=100)
        shape = page.new_shape()
        shape.draw_polyline([(10, 90), (50, 50), (90, 10)])
        shape.finish(color=(0, 0, 0), closePath=False)
        shape.commit()
        doc.save(path)
    before = path.read_bytes()
    series, metadata = Backend("pymupdf").digitize(path, Calibration(config()), tmp_path)
    assert series[0]["x"] == [1, 5, 9] and series[0]["y"] == [1, 5, 9]
    assert metadata["page"] == 1 and path.read_bytes() == before


def test_step_histogram_preserves_plateaus_without_inventing_bins(tmp_path):
    vertices = [[10, 120], [10, 70], [30, 70], [30, 40], [90, 40], [90, 120]]
    path = {"items": [["l", a, b] for a, b in zip(vertices, vertices[1:])], "rect": [10, 40, 90, 120]}
    c = config()
    c["series"][0]["geometry"] = "step_histogram"
    series = vector_series([path], Calibration(c))[0]
    assert series["geometry"] == "rendered_step_intervals"
    assert series["x"] == [2, 6] and series["y"] == [3, 6]
    assert series["rendered_intervals"] == [[1, 3], [3, 9]]
    assert series["original_bin_edges_verified"] is False and "bins" not in series
    assert len(series["source_horizontal_segments"]) == 2
    from kamiknows.plot_extractor.validation import structural_check

    structural_check(series)
    c["y"] = {
        "scale": "log10",
        "unit": "events",
        "anchors": [{"position": 0, "value": 1e-2}, {"position": 100, "value": 1e-4}],
    }
    logarithmic = vector_series([path], Calibration(c))[0]
    assert logarithmic["y"] == pytest.approx([10**-3.4, 10**-2.8])


@pytest.mark.parametrize(
    "vertices",
    [
        [[10, 70], [30, 60]],
        [[30, 70], [10, 70]],
        [[10, 70], [110, 70]],
    ],
)
def test_step_histogram_rejects_diagonals_backtracking_and_crop(vertices):
    c = config()
    c["series"][0]["geometry"] = "step_histogram"
    path = {"items": [["l", a, b] for a, b in zip(vertices, vertices[1:])], "rect": [0, 0, 100, 100]}
    with pytest.raises(ExtractionError):
        vector_series([path], Calibration(c))


def test_digitization_figure_selection_validates_before_output(visual_run, tmp_path, monkeypatch):
    parsed, classified, figure, asset, c, calibration = visual_run
    install_fake_raster(monkeypatch)
    with pytest.raises(ValueError, match="figure_ids"):
        run_digitization(
            parsed_dir=parsed, classified_dir=classified, output_dir=tmp_path / "bad", figure_ids=["missing"]
        )
    assert not (tmp_path / "bad").exists()
    summary = run_digitization(
        parsed_dir=parsed,
        classified_dir=classified,
        output_dir=tmp_path / "selected",
        figure_ids=[figure["figure_id"]],
        backend="raster",
        calibration_file=calibration,
    )
    assert summary["records"] == 1
    assert read_json(tmp_path / "selected" / "manifest.json")["params"]["figure_ids"] == [figure["figure_id"]]


def step_config():
    c = config()
    c["series"][0].update(geometry="step_histogram", max_step_drift=0.1)
    return c


def test_step_histogram_in_pieces_lists_bins_below_the_frame():
    first = [[10, 150], [10, 40], [30, 40], [30, 60], [50, 60], [50, 150], [60, 150]]
    second = [[70, 150], [70, 80], [90, 80], [90, 150]]
    items = [["l", a, b] for piece in (first, second) for a, b in zip(piece, piece[1:])]
    series = vector_series([{"items": items, "rect": [10, 40, 90, 150]}], Calibration(step_config()))[0]
    assert series["x"] == [2, 4, 8] and series["y"] == [6, 4, 2]
    assert series["rendered_intervals"] == [[1, 3], [3, 5], [7, 9]]
    assert series["below_range_intervals"] == [[5, 6]]


def test_filled_step_histogram_keeps_the_steps_and_refuses_a_band():
    outline = [[10, 150], [10, 40], [30, 40], [30, 60], [50, 60], [50, 150], [10, 150]]
    fill = {"items": [["l", a, b] for a, b in zip(outline, outline[1:])], "fill": [0, 0, 1], "rect": [10, 40, 50, 150]}
    series = vector_series([fill], Calibration(step_config()))[0]
    assert series["x"] == [2, 4] and series["y"] == [6, 4] and series["below_range_intervals"] == []
    band = [[10, 40], [30, 40], [30, 50], [50, 50], [50, 70], [30, 70], [30, 80], [10, 80]]
    fill["items"] = [["l", a, b] for a, b in zip(band, band[1:])]
    with pytest.raises(ExtractionError, match="varying height"):
        vector_series([fill], Calibration(step_config()))


def test_step_above_the_frame_is_refused():
    above = [[10, 50], [10, -20], [30, -20], [30, 50], [50, 50]]
    path = {"items": [["l", a, b] for a, b in zip(above, above[1:])], "rect": [10, -20, 50, 50]}
    with pytest.raises(ExtractionError, match="above or across"):
        vector_series([path], Calibration(step_config()))


def test_step_histogram_requires_explicit_drift_tolerance():
    path = {"items": [["l", [10, 70], [30, 70.05]]], "rect": [10, 70, 30, 70.05]}
    c = config()
    c["series"][0]["geometry"] = "step_histogram"
    with pytest.raises(ExtractionError):
        vector_series([path], Calibration(c))
    c["series"][0]["max_step_drift"] = 0.1
    series = vector_series([path], Calibration(c))[0]
    assert series["source_vertical_drifts"] == pytest.approx([0.05])
    assert series["y"] == pytest.approx([2.9975])
    assert series["rendered_height_ranges"][0] == pytest.approx([2.995, 3])
    assert not series["displayed_uncertainties"]


def test_digitization_visual_report_has_no_surrogate_metrics_and_keeps_inputs(tmp_path):
    from PIL import Image, ImageDraw
    from kamiknows.plot_extractor.validation import run_digitization_review

    run = tmp_path / "digitized"
    work = run / "overlays" / "digitized_test"
    work.mkdir(parents=True)
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).line((10, 50, 90, 50), fill="blue", width=2)
    image.save(work / "source.png")
    cfg = config()
    record = {
        "data_id": "digitized_test",
        "figure_id": "fig",
        "extraction_status": "success",
        "provenance": {"asset_id": "asset"},
        "calibration": cfg,
        "raw_source": {"render_scale": 1, "paths": [{"color": [0, 0, 1]}]},
        "series": [{"series_id": "s", "geometry": "rendered_curve_samples", "x": [2, 5, 8], "y": [5, 5, 5]}],
    }
    write_jsonl(run / "digitized_data.jsonl", [record])
    write_manifest(
        run,
        step="extract_plots",
        params={"mode": "digitize"},
        inputs={},
        summary={"data_sha256": sha256_file(run / "digitized_data.jsonl")},
    )
    before = sha256_file(run / "digitized_data.jsonl")
    out = tmp_path / "review"
    result = run_digitization_review(digitized_dir=run, output_dir=out)
    assert result["reported_assets"] == 1 and result["review_errors"] == 0
    metrics = read_json(out / "extraction_summary.json")
    assert "scientific accuracy not established" in metrics["assessment"]
    assert metrics["assets"][0]["elements"] == 3
    assert metrics["assets"][0]["series"][0]["axis_metadata"]["x"]["unit"] == "GeV"
    assert (
        "x_unit,x_unit_status,x_scale,y_unit,y_unit_status,y_scale"
        in (out / "digitized_test" / "extracted_values.csv").read_text()
    )
    assert (out / "digitized_test" / "extracted_values.csv").exists()
    assert not (out / "comparison_metrics.json").exists()
    page = (out / "review.html").read_text()
    assert "original scientific data" in page
    assert "Axis units and scales" in page and "GeV" in page
    import re

    assert not re.search(r"\bratio\b|\bdifferences?\b|source_render_reference", page, re.I)
    assert (out / "digitized_test" / "numeric_overlay.svg").exists()
    assert 'src="data:image/png;base64,' in page
    assert 'href="data:text/csv;base64,' in page and " download=" in page
    assert 'src="digitized_test/' not in page and "<object" not in page
    assert (out / "report.pdf").exists() and (out / "exports.zip").exists()
    import zipfile

    with zipfile.ZipFile(out / "exports.zip") as archive:
        assert "digitized_test/extracted_values.csv" in archive.namelist()
        assert not any("quantitative_comparison" in n for n in archive.namelist())

    assert sha256_file(run / "digitized_data.jsonl") == before
    (run / "digitized_data.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_digitization_review(digitized_dir=run, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_review_preserves_downward_increasing_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kamiknows.plot_extractor.validation import _apply_source_axis_limits

    cfg = config()
    cfg["y"]["anchors"] = [{"position": 0, "value": 0}, {"position": 100, "value": 10}]
    fig, ax = plt.subplots()
    _apply_source_axis_limits(ax, Calibration(cfg))
    assert ax.get_ylim() == pytest.approx((10, 0)) and ax.yaxis_inverted()
    plt.close(fig)


def test_review_embedding_preserves_bytes_and_rejects_missing_resources(tmp_path):
    import base64
    from kamiknows.plot_extractor.validation import _embed_review_resources

    (tmp_path / "sample.csv").write_bytes(b"x,y\n1,2\n")
    page = _embed_review_resources('<a href="sample.csv">CSV</a>', tmp_path)
    assert base64.b64encode(b"x,y\n1,2\n").decode() in page
    assert "download=" in page
    with pytest.raises(ValueError, match="missing report resource"):
        _embed_review_resources('<img src="absent.png">', tmp_path)


def test_content_inspection_distinguishes_bitmap_pdf_and_vector_pdf(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    from PIL import Image
    from kamiknows.plot_extractor.digitization import inspect_representation

    bitmap = tmp_path / "pixels.png"
    Image.new("RGB", (40, 40), "blue").save(bitmap)
    for vector, expected in [(False, "raster"), (True, "hybrid")]:
        target = tmp_path / (expected + ".pdf")
        with pdf.open() as doc:
            page = doc.new_page(width=100, height=100)
            page.insert_image(page.rect, filename=str(bitmap))
            if vector:
                page.draw_line((10, 10), (80, 80))
            doc.save(target)
        assert inspect_representation(target)["representation"] == expected


def test_pixel_backend_recovers_simple_colored_trace_without_vector_paths(tmp_path):
    pytest.importorskip("cv2")
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (260, 180), "white")
    ImageDraw.Draw(image).line((25, 145, 215, 35), fill="blue", width=2)
    path = tmp_path / "curve.png"
    image.save(path)
    cfg = config("pixels")
    cfg.update(roi=[20, 20, 220, 160], raster_extraction={"kind": "colored_traces"})
    cfg["x"]["anchors"] = [{"position": 20, "value": 0}, {"position": 220, "value": 10}]
    cfg["y"]["anchors"] = [{"position": 20, "value": 10}, {"position": 160, "value": 0}]
    work = tmp_path / "work"
    work.mkdir()
    series, metadata = Backend("raster").digitize(path, Calibration(cfg), work)
    assert len(series) == 1 and len(series[0]["x"]) > 150
    assert series[0]["y"][0] == pytest.approx(1, abs=0.15)
    assert series[0]["sampling"]["original_measurement_points"] is False
    assert metadata["paths"] == [] and (work / "source.png").exists()


def test_pixel_backend_extracts_display_blocks_and_rejects_smooth_images(tmp_path):
    pytest.importorskip("cv2")
    from PIL import Image
    import numpy as np

    rgb = np.full((140, 210, 3), 255, dtype=np.uint8)
    palette = np.array(
        [[int(255 * i / 119), 0, int(255 * (1 - i / 119))] for i in range(120)], dtype=np.uint8
    )
    rgb[10:130, 170:190] = palette[:, None, :]
    for rect, index in [
        ((10, 10, 80, 70), 20),
        ((80, 10, 150, 70), 40),
        ((10, 70, 80, 130), 60),
        ((80, 70, 150, 130), 80),
    ]:
        x0, y0, x1, y1 = rect
        rgb[y0:y1, x0:x1] = palette[index]
    path = tmp_path / "map.png"
    Image.fromarray(rgb).save(path)
    cfg = config("pixels")
    cfg["roi"] = [10, 10, 150, 130]
    cfg["x"]["anchors"] = [{"position": 10, "value": 0}, {"position": 150, "value": 2}]
    cfg["y"]["anchors"] = [{"position": 10, "value": 2}, {"position": 130, "value": 0}]
    z = {
        "scale": "linear",
        "unit": "counts",
        "anchors": [{"position": 10, "value": 10}, {"position": 130, "value": 0}],
    }
    cfg.update(
        raster_extraction={"kind": "histogram_2d"},
        histogram2d_panels=[
            {"roi": cfg["roi"], "x": cfg["x"], "y": cfg["y"], "z": z, "colorbar_roi": [170, 10, 190, 130]}
        ],
    )
    work = tmp_path / "work"
    work.mkdir()
    series, _ = Backend("raster").digitize(path, Calibration(cfg), work)
    assert len(series[0]["z"]) == 4
    assert sorted(series[0]["z"]) == pytest.approx([3.29, 4.96, 6.63, 8.29], abs=0.1)
    assert not series[0]["original_bin_edges_verified"]
    rgb[10:130, 10:150] = palette[:, None, :]
    Image.fromarray(rgb).save(path)
    with pytest.raises(ExtractionError, match="constant-color raster block"):
        Backend("raster").digitize(path, Calibration(cfg), work)


def test_native_unsupported_geometry_can_fall_back_without_changing_original_config(tmp_path, monkeypatch):
    from PIL import Image
    import io
    from kamiknows.plot_extractor import digitization

    image = Image.new("RGB", (100, 100), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    cfg = config()
    cfg["automatic_raster_fallback"] = True
    cal = Calibration(cfg)
    before = copy.deepcopy(cfg)
    monkeypatch.setattr(
        digitization, "read_pdf", lambda *args: ([line([[10, 90], [90, 10]])], {}, buffer.getvalue())
    )

    def unsupported(*args):
        raise ExtractionError("unsupported", "synthetic vector restriction")

    monkeypatch.setattr(digitization, "vector_series", unsupported)

    def pixels(path, selected, work, dpi):
        assert selected.config["raster_extraction"]["kind"] == "colored_traces"
        return [], {"render_scale": 1}

    monkeypatch.setattr(digitization, "pixel_digitization", pixels)
    monkeypatch.setattr(Backend, "available", lambda self: True)
    series, metadata = Backend("pymupdf").digitize(tmp_path / "plot.pdf", cal, tmp_path)
    assert metadata["backend_used"] == "raster" and metadata["backend_requested"] == "pymupdf"
    assert cfg == before
