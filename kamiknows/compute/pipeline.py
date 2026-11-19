"""Step compute: run the expected plan of every question and check its results.

Plans written by hand (``expected_plan``) and the numbers they should give
(``expected_results``) are evaluation annotations. Running them validates
the operations and the extracted data with no language model involved, so a
later wrong answer can be blamed on the plan the model wrote, not on the math.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from kamiknows.common import create_output_dir, write_jsonl, write_manifest
from kamiknows.compute.plan import execute
from kamiknows.compute.sources.plots import DATA_FILE, load_plots
from kamiknows.retrieval import load_questions


def run_compute(*, questions_file: Path, digitized_dir: Path, output_dir: Path, rel_tol: float) -> dict[str, Any]:
    if rel_tol <= 0:
        raise ValueError("rel_tol must be positive")
    questions = load_questions(questions_file)
    catalog = load_plots(digitized_dir)
    output_dir = create_output_dir(output_dir)
    records = []
    for question in questions:
        record = {"question_id": question["question_id"], "question": question["question"]}
        if "expected_plan" not in question:
            records.append({**record, "outcome": None, "check": {"match": None, "reason": "no expected_plan"}})
            continue
        outcome = execute(question["expected_plan"], catalog)
        check = compare(outcome, question.get("expected_results"), rel_tol)
        records.append({**record, "plan": question["expected_plan"], "outcome": outcome,
                        "expected_results": question.get("expected_results"), "check": check})
    write_jsonl(output_dir / "results.jsonl", records)
    planned = [r for r in records if r["outcome"] is not None]
    summary = {
        "questions": len(records),
        "with_plan": len(planned),
        "status_counts": dict(Counter(r["outcome"]["status"] for r in planned)),
        "matched": sum(r["check"]["match"] is True for r in planned),
        "mismatched_question_ids": [r["question_id"] for r in planned if r["check"]["match"] is False],
        "series_available": sum(entry.series is not None for entry in catalog),
        "assets_without_series": sum(entry.series is None for entry in catalog),
    }
    inputs = {"questions": questions_file, "digitization_manifest": Path(digitized_dir) / "manifest.json",
              "digitized_data": Path(digitized_dir) / DATA_FILE}
    write_manifest(output_dir, step="compute",
                   params={"plans": "expected_plan", "rel_tol": rel_tol, "digitized_dir": digitized_dir},
                   inputs=inputs, summary=summary)
    return summary


def compare(outcome: Mapping[str, Any], expected: Mapping[str, Any] | None, rel_tol: float) -> dict[str, Any]:
    """Check an outcome against the expected status and results.

    Expected results are a subset of the rendered results: every key given
    must be present, numbers must agree within ``rel_tol``, strings and list
    lengths must be equal. Keys left out (origins, texts) are not checked.
    """
    if expected is None:
        return {"match": None, "reason": "no expected_results"}
    differences = []
    if outcome["status"] != expected.get("status"):
        differences.append(f"status: expected {expected.get('status')}, got {outcome['status']}"
                           + (f" ({outcome['reason']})" if outcome.get("reason") else ""))
    elif "results" in expected:
        differences = _differences(expected["results"], outcome.get("results"), "results", rel_tol)
    return {"match": not differences, "differences": differences}


def _differences(expected: Any, actual: Any, path: str, rel_tol: float) -> list[str]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected an object, got {actual!r}"]
        found = []
        for key, item in expected.items():
            if key not in actual:
                found.append(f"{path}.{key}: missing")
            else:
                found += _differences(item, actual[key], f"{path}.{key}", rel_tol)
        return found
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: expected {len(expected)} items, got {actual!r}"]
        return [d for i, (e, a) in enumerate(zip(expected, actual)) for d in _differences(e, a, f"{path}[{i}]", rel_tol)]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        number = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        if not number or not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=1e-12):
            return [f"{path}: expected {expected:g}, got {actual!r}"]
        return []
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]
