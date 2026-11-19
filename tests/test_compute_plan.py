"""Compositions, the operation registry, the plan executor and the compute step."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kamiknows.common import read_json, read_jsonl, write_jsonl, write_manifest
from kamiknows.compute import (
    Axis, Catalog, Entry, NotComputable, Quantity, Series, Unit, crossing, describe, distance, evaluate,
    execute, fwhm, linear_fit, mean, normalize, quantile, reduce, std,
)
from kamiknows.compute.pipeline import compare, run_compute
from kamiknows.compute.registry import OPERATIONS, catalogue

ROOT = Path(__file__).resolve().parents[1]
GEV, AU = Unit.parse("GeV"), Unit.parse("A.U.")


def histogram(values, edges, name="h"):
    edges = np.asarray(edges, dtype=float)
    return Series(name, (Axis("x", GEV),), Axis("y", AU, "log10"),
                  centers=((edges[:-1] + edges[1:]) / 2)[:, None], values=np.asarray(values, dtype=float),
                  lower=edges[:-1, None], upper=edges[1:, None], caveats=("drawn values",))


def samples(x, values, name="c"):
    return Series(name, (Axis("x", GEV),), Axis("y", AU),
                  centers=np.asarray(x, dtype=float)[:, None], values=np.asarray(values, dtype=float))


def gaussian(mu=0.005, sigma=2.0):
    edges = np.linspace(-10, 10, 2001)
    centers = (edges[:-1] + edges[1:]) / 2
    return histogram(np.exp(-0.5 * ((centers - mu) / sigma) ** 2), edges)


# --- compositions -------------------------------------------------------------

def test_crossing_finds_where_one_series_overtakes_the_other():
    blue, red = histogram([5, 4, 2, 1], [0, 10, 20, 30, 40]), histogram([3, 3, 3, 3], [0, 10, 20, 30, 40])
    overtakes, = crossing(red, blue, "up")
    assert overtakes.bounds == {"x": (20, 30)}
    assert overtakes.value.value == 3 and overtakes.value.unit == Unit.parse("A.U.")   # red's own bin
    assert overtakes.origin == "crossing(h, h, up)[0]"
    curve_a, curve_b = samples([0, 1, 2], [0, 2, 4]), samples([0, 1, 2], [1, 1, 1])
    point, = crossing(curve_a, curve_b, "up")
    assert point.coords["x"].value == pytest.approx(0.5) and point.value.value == pytest.approx(1)


def test_moments_quantile_and_width_of_a_gaussian():
    h = gaussian()
    assert mean(h, "x").value == pytest.approx(0.005, abs=1e-3)
    assert std(h, "x").value == pytest.approx(2.0, rel=1e-3) and std(h, "x").unit == GEV
    assert quantile(h, 0.5).value == pytest.approx(0.005, abs=1e-3)
    assert quantile(h, 0.8413).value == pytest.approx(0.005 + 2.0, abs=0.01)   # one sigma above
    assert fwhm(h).value == pytest.approx(2 * np.sqrt(2 * np.log(2)) * 2.0, abs=0.02)
    assert reduce(normalize(h), "integral").value == pytest.approx(1)
    with pytest.raises(NotComputable):
        quantile(h, 1.5)


def test_fwhm_needs_both_sides_of_the_peak():
    with pytest.raises(NotComputable, match="both sides"):
        fwhm(histogram([5, 4, 3], [0, 1, 2, 3]))


def test_linear_fit_recovers_a_line_with_units():
    fit = linear_fit(samples([0, 1, 2, 3], [1, 3, 5, 7]))
    assert fit["slope"].value == pytest.approx(2) and fit["slope"].unit == AU / GEV
    assert fit["intercept"].value == pytest.approx(1) and fit["intercept"].unit == AU


def test_distance_along_an_axis_and_euclidean():
    cell = Unit.parse("cell_index")
    grid = Series("g", (Axis("x", cell), Axis("y", cell)), Axis("z", AU),
                  centers=[[0.0, 0.0], [3.0, 4.0]], values=[1.0, 2.0],
                  lower=[[-0.5, -0.5], [2.5, 3.5]], upper=[[0.5, 0.5], [3.5, 4.5]])
    a, b = evaluate(grid, {"x": 0, "y": 0}), evaluate(grid, {"x": 3, "y": 4})
    assert distance(a, b, "x").value == 3
    assert distance(a, b).value == pytest.approx(5)


def test_describe_reports_axes_and_caveats():
    info = describe(histogram([1, 2], [0, 10, 20]))
    assert info["axes"]["x"] == {"role": "domain", "unit": "GeV", "scale": "linear", "range": [0, 20]}
    assert info["axes"]["y"]["scale"] == "log10"
    assert info["elements_are"] == "bins or cells" and info["caveats"] == ["drawn values"]


# --- registry -----------------------------------------------------------------

def test_registry_matches_the_function_signatures():
    for operation in OPERATIONS.values():
        signature = inspect.signature(operation.function).parameters
        assert [p.name for p in operation.parameters] == list(signature), operation.name
        for parameter in operation.parameters:
            has_default = signature[parameter.name].default is not inspect.Parameter.empty
            assert parameter.required != has_default, (operation.name, parameter.name)
        assert operation.summary
    text = catalogue()
    assert all(f"{name}(" in text for name in OPERATIONS)
    assert "reduce(series: series, how: sum|integral|mean|min|max|count) -> quantity" in text


# --- plan executor ------------------------------------------------------------

@pytest.fixture
def catalog():
    return Catalog([
        Entry("blue", histogram([5, 4, 2, 1], [0, 10, 20, 30, 40], "blue")),
        Entry("red", histogram([3, 3, 3, 3], [0, 10, 20, 30, 40], "red")),
        Entry("figure2", None, "plot not digitized (extraction status: needs_calibration)"),
    ])


def test_plan_chains_steps_through_references(catalog):
    plan = {"steps": [
        {"id": "b", "op": "reduce", "args": {"series": "blue", "how": "max"}},
        {"id": "r", "op": "reduce", "args": {"series": "red", "how": "max"}},
        {"id": "ratio", "op": "combine", "args": {"left": "$b", "right": "$r", "op": "div"}},
        {"id": "c", "op": "crossing", "args": {"first": "red", "second": "blue", "direction": "up"}},
        {"id": "v", "op": "evaluate", "args": {"series": "blue", "at": {"x": "$c[0].x"}}},
        {"id": "w", "op": "evaluate", "args": {"series": "blue", "at": {"x": {"value": 35000, "unit": "MeV"}}}},
    ], "result": ["$ratio", "$v", "$w.value"]}
    outcome = execute(plan, catalog)
    assert outcome["status"] == "computed"
    ratio, point, value = outcome["results"]
    assert ratio["value"] == pytest.approx(5 / 3) and ratio["unit"] == "1"
    assert ratio["origin"] == "div(max(blue), max(red))"
    assert point["coords"]["x"] == {"value": 25, "unit": "GeV", "lower": 20, "upper": 30}
    assert value["value"] == 1
    assert outcome["caveats"] == ["drawn values"]
    assert [step["id"] for step in outcome["steps"]] == ["b", "r", "ratio", "c", "v", "w"]


def test_declining_plan(catalog):
    outcome = execute({"not_computable": "event counts are not shown"}, catalog)
    assert outcome == {"status": "declined", "reason": "event counts are not shown", "steps": []}


@pytest.mark.parametrize("plan, message", [
    ({"steps": [{"id": "a", "op": "integrate", "args": {}}], "result": ["$a"]}, "unknown operation"),
    ({"steps": [{"id": "a", "op": "reduce", "args": {"series": "blue"}}], "result": ["$a"]}, "needs 'how'"),
    ({"steps": [{"id": "a", "op": "reduce", "args": {"series": "blue", "how": "median"}}], "result": ["$a"]},
     "how must be"),
    ({"steps": [{"id": "a", "op": "reduce", "args": {"series": "blue", "how": "max", "axis": "x"}}],
      "result": ["$a"]}, "no parameter"),
    ({"steps": [{"id": "a", "op": "combine", "args": {"left": "$b", "right": 1, "op": "add"}}],
      "result": ["$a"]}, "has not run"),
    ({"steps": [{"id": "a", "op": "reduce", "args": {"series": "blue", "how": "max"}},
                {"id": "a", "op": "reduce", "args": {"series": "red", "how": "max"}}], "result": ["$a"]}, "new identifier"),
    ({"steps": [{"id": "a", "op": "reduce", "args": {"series": "blue", "how": "max"}}], "result": "a"}, "'result'"),
    ({"steps": [{"id": "a", "op": "reduce", "args": {"series": "blue", "how": "max"}}], "result": ["$a.x"]},
     "cannot take"),
    ({"answer": 3}, "exactly 'steps' and 'result'"),
])
def test_malformed_plans_are_invalid(catalog, plan, message):
    outcome = execute(plan, catalog)
    assert outcome["status"] == "invalid_plan" and message in outcome["reason"]


def test_bad_arguments_show_the_signature(catalog):
    plan = {"steps": [{"id": "a", "op": "argextremum", "args": {"series": "blue", "kind": "localmax"}}],
            "result": ["$a"]}
    reason = execute(plan, catalog)["reason"]
    assert "argextremum: kind must be" in reason and "scope?: global|local" in reason


@pytest.mark.parametrize("plan, step", [
    ({"steps": [{"id": "a", "op": "reduce", "args": {"series": "figure2", "how": "max"}}], "result": ["$a"]}, "a"),
    ({"steps": [{"id": "a", "op": "evaluate", "args": {"series": "blue", "at": {"x": 99}}}], "result": ["$a"]}, "a"),
    ({"steps": [{"id": "c", "op": "solve", "args": {"series": "blue", "level": 100}}], "result": ["$c[0]"]}, "result"),
])
def test_data_limits_are_not_computable(catalog, plan, step):
    outcome = execute(plan, catalog)
    assert outcome["status"] == "not_computable" and outcome["failed_step"] == step


# --- compute step -------------------------------------------------------------

def test_compare_checks_a_subset_within_tolerance():
    outcome = {"status": "computed", "results": [{"type": "quantity", "value": 1.5985, "unit": "1", "origin": "o"}]}
    assert compare(outcome, {"status": "computed", "results": [{"value": 1.599, "unit": "1"}]}, 0.01)["match"]
    assert not compare(outcome, {"status": "computed", "results": [{"value": 1.7}]}, 0.01)["match"]
    assert not compare(outcome, {"status": "declined"}, 0.01)["match"]
    assert compare(outcome, None, 0.01)["match"] is None


def test_run_compute_executes_expected_plans(tmp_path):
    digitized = tmp_path / "plots"
    series = {"series_id": "path_2", "geometry": "rendered_step_intervals", "units": {"x": "GeV", "y": "A.U."},
              "x": [5.0, 15.0], "y": [2.0, 4.0], "rendered_intervals": [[0.0, 10.0], [10.0, 20.0]],
              "value_states": ["visible"] * 2, "original_bin_edges_verified": False}
    write_jsonl(digitized / "digitized_data.jsonl", [{
        "paper_id": "p", "figure_id": "p::figure::0001", "provenance": {"asset_id": "p::a1"},
        "extraction_status": "success", "axes": {"x": {"scale": "linear"}, "y": {"scale": "log10"}},
        "series": [series]}])
    write_manifest(digitized, step="extract_plots", params={}, inputs={}, summary={})
    questions = tmp_path / "questions.jsonl"
    peak_plan = {"steps": [{"id": "p", "op": "argextremum", "args": {"series": "p::a1/path_2"}}], "result": ["$p[0]"]}
    write_jsonl(questions, [
        {"question_id": "q1", "question": "Peak?", "expected_plan": peak_plan,
         "expected_results": {"status": "computed", "results": [{"coords": {"x": {"value": 15}}, "value": {"value": 4}}]}},
        {"question_id": "q2", "question": "Counts?", "expected_plan": {"not_computable": "no counts"},
         "expected_results": {"status": "declined"}},
        {"question_id": "q3", "question": "Wrong?", "expected_plan": peak_plan,
         "expected_results": {"status": "computed", "results": [{"value": {"value": 3}}]}},
        {"question_id": "q4", "question": "No plan"},
    ])
    summary = run_compute(questions_file=questions, digitized_dir=digitized, output_dir=tmp_path / "run", rel_tol=0.01)
    assert summary["with_plan"] == 3 and summary["matched"] == 2
    assert summary["mismatched_question_ids"] == ["q3"]
    assert summary["status_counts"] == {"computed": 2, "declined": 1}
    assert read_json(tmp_path / "run" / "manifest.json")["step"] == "compute"
    assert [r["question_id"] for r in read_jsonl(tmp_path / "run" / "results.jsonl")] == ["q1", "q2", "q3", "q4"]


def test_compute_script_help():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/compute.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--digitized-dir" in result.stdout


def test_plan_questions_file_is_well_formed():
    path = ROOT / "data/questions/fastcalo_plots_values_plans.jsonl"
    original = read_jsonl(ROOT / "data/questions/fastcalo_plots_values.jsonl")
    questions = read_jsonl(path)
    assert [q["question_id"] for q in questions] == [q["question_id"] for q in original]
    for question, source in zip(questions, original):
        assert {k: v for k, v in question.items() if k not in ("expected_plan", "expected_results")} == source
        assert "expected_plan" in question and question["expected_results"]["status"] in ("computed", "declined")
    json.dumps(questions)
