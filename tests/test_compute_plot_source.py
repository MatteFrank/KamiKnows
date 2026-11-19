"""Digitized plot records become compute series."""

import pytest

from kamiknows.common import write_jsonl, write_manifest
from kamiknows.compute import UNKNOWN, NotComputable, Unit
from kamiknows.compute.sources.plots import load_plots

FIGURE = {"paper_id": "1111_00001v1", "figure_id": "1111_00001v1::figure::0001"}


def _heatmap(asset, y_unit="cell_index"):
    calibration = {"x": {"scale": "linear", "unit": "cell_index", "unit_status": "explicit"},
                   "y": {"scale": "linear", "unit": y_unit, "unit_status": "explicit" if y_unit != "unknown" else "missing"},
                   "z": {"scale": "log10", "unit": "MeV", "unit_status": "explicit"}}
    series = {"series_id": "panel_1_cells", "geometry": "histogram_2d_cells",
              "units": {"x": "cell_index", "y": y_unit, "z": "MeV"},
              "x": [0.0, 1.0], "y": [0.0, 0.0], "z": [10.0, 2000.0],
              "cell_bounds": [{"x": [-0.5, 0.5], "y": [-0.5, 0.5]}, {"x": [0.5, 1.5], "y": [-0.5, 0.5]}],
              "value_states": ["color_inverted_estimate"] * 2, "original_bin_edges_verified": False,
              "axis_calibration": calibration}
    return {**FIGURE, "provenance": {"asset_id": asset}, "extraction_status": "success", "axes": {}, "series": [series]}


def _steps(asset):
    series = {"series_id": "path_2", "geometry": "rendered_step_intervals", "units": {"x": "GeV", "y": "A.U."},
              "x": [45.0, 55.0], "y": [0.005, 0.004], "rendered_intervals": [[40.0, 50.0], [50.0, 60.0]],
              "value_states": ["visible"] * 2, "original_bin_edges_verified": False}
    axes = {"x": {"scale": "linear", "unit": "GeV"}, "y": {"scale": "log10", "unit": "A.U."}}
    return {**FIGURE, "provenance": {"asset_id": asset}, "extraction_status": "success", "axes": axes, "series": [series]}


def _failed(asset, status="needs_calibration"):
    return {**FIGURE, "provenance": {"asset_id": asset}, "extraction_status": status, "axes": {}, "series": []}


def _run(tmp_path, name, records):
    run = tmp_path / name
    write_jsonl(run / "digitized_data.jsonl", records)
    write_manifest(run, step="extract_plots", params={}, inputs={}, summary={})
    return run


def test_heatmap_and_histogram_become_series(tmp_path):
    catalog = load_plots(_run(tmp_path, "d", [_heatmap("a1"), _steps("a2")]))
    heatmap = catalog.get("a1/panel_1_cells")
    assert heatmap.dim == 2 and heatmap.size == 2 and heatmap.has_support
    assert heatmap.value_axis.unit == Unit.parse("MeV") and heatmap.value_axis.scale == "log10"
    assert "values are read from the colour scale" in heatmap.caveats
    histogram = catalog.get("a2/path_2")
    assert histogram.dim == 1
    assert histogram.lower[:, 0].tolist() == [40.0, 50.0]
    assert histogram.value_axis.scale == "log10"


def test_missing_unit_stays_unknown(tmp_path):
    catalog = load_plots(_run(tmp_path, "d", [_heatmap("a1", y_unit="unknown")]))
    assert catalog.get("a1/panel_1_cells").axis("y").unit == UNKNOWN


def test_undigitized_asset_explains_why(tmp_path):
    catalog = load_plots(_run(tmp_path, "d", [_failed("a3")]))
    with pytest.raises(NotComputable, match="needs_calibration"):
        catalog.get("a3")


def test_same_series_recorded_twice_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="duplicate handle 'a2/path_2'"):
        load_plots(_run(tmp_path, "d", [_steps("a2"), _steps("a2")]))


def test_unsupported_geometry_is_reported(tmp_path):
    record = _steps("a2")
    record["series"][0]["geometry"] = "scatter_points"
    with pytest.raises(NotComputable, match="not supported"):
        load_plots(_run(tmp_path, "d", [record])).get("a2/path_2")


def test_wrong_run_type_is_rejected(tmp_path):
    run = tmp_path / "chunks"
    write_jsonl(run / "digitized_data.jsonl", [])
    write_manifest(run, step="chunk", params={}, inputs={}, summary={})
    with pytest.raises(ValueError, match="expected 'extract_plots'"):
        load_plots(run)
