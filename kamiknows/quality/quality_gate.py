"""Quality gate for KamiKnows tutorial runs.

The quality gate combines two layers:

1. formal traceability from ``dataset_manifest.json``;
2. human scientific review from ``*_manual_review_summary.json`` files.

It does not judge scientific correctness automatically. It only turns already
recorded formal checks and human review outcomes into an operational decision:
``ACCEPT``, ``REVISE`` or ``REJECT``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

QUALITY_GATE_VERSION = "quality_gate_v0"
ALLOWED_GATE_DECISIONS = {"ACCEPT", "REVISE", "REJECT"}


class QualityGateError(RuntimeError):
    """Raised when a quality gate input cannot be evaluated."""


def _decision_rank(decision: str) -> int:
    ranks = {"ACCEPT": 0, "REVISE": 1, "REJECT": 2}
    if decision not in ranks:
        raise QualityGateError(f"unknown quality gate decision: {decision}")
    return ranks[decision]


def _max_decision(*decisions: str) -> str:
    if not decisions:
        return "ACCEPT"
    return max(decisions, key=_decision_rank)


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    json_path = Path(path)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QualityGateError(f"invalid JSON file {json_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise QualityGateError(f"JSON file must contain an object: {json_path}")
    return data


def discover_manual_review_summary_paths(manifest: dict[str, Any]) -> list[Path]:
    """Return manual review summary paths declared in a dataset manifest."""
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise QualityGateError("dataset manifest 'files' must be a list")

    paths: list[Path] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "manual_review_summary":
            continue
        path = entry.get("path")
        if isinstance(path, str) and path.strip():
            paths.append(Path(path))
    return paths


def _evaluate_manifest(manifest: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    decision = "ACCEPT"

    status = manifest.get("status")
    if status != "PASS":
        decision = _max_decision(decision, "REVISE")
        reasons.append(f"dataset manifest status is {status!r}, not 'PASS'")

    missing_paths = manifest.get("missing_paths", [])
    if missing_paths:
        decision = _max_decision(decision, "REJECT")
        reasons.append(f"dataset manifest reports missing files: {len(missing_paths)}")

    files = manifest.get("files", [])
    if not files:
        decision = _max_decision(decision, "REJECT")
        reasons.append("dataset manifest does not register any files")

    return decision, reasons


def _evaluate_manual_summary(summary: dict[str, Any], index: int) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    decision = "ACCEPT"

    status = summary.get("status")
    total_records = int(summary.get("total_records", 0) or 0)
    outcome_counts = summary.get("outcome_counts", {})
    if not isinstance(outcome_counts, dict):
        outcome_counts = {}

    reject_count = int(outcome_counts.get("reject", 0) or 0)
    revise_count = int(outcome_counts.get("revise", 0) or 0)
    unclear_count = int(outcome_counts.get("unclear", 0) or 0)
    pass_count = int(outcome_counts.get("pass", 0) or 0)
    fully_checked_records = int(summary.get("fully_checked_records", 0) or 0)

    if total_records <= 0:
        decision = _max_decision(decision, "REVISE")
        reasons.append(f"manual review summary {index} has no reviewed records")

    if status == "PASS":
        pass
    elif status == "REVIEW_REJECTED_RECORDS" or reject_count > 0:
        decision = _max_decision(decision, "REJECT")
        reasons.append(f"manual review summary {index} contains rejected records")
    elif status in {"REVIEW_REVISIONS_NEEDED", "REVIEW_INCOMPLETE"}:
        decision = _max_decision(decision, "REVISE")
        reasons.append(f"manual review summary {index} status is {status}")
    else:
        decision = _max_decision(decision, "REVISE")
        reasons.append(f"manual review summary {index} has unknown status {status!r}")

    if fully_checked_records < total_records:
        decision = _max_decision(decision, "REVISE")
        reasons.append(
            f"manual review summary {index} has {fully_checked_records}/{total_records} fully checked records"
        )

    compact = {
        "index": index,
        "source_path": summary.get("source_path"),
        "status": status,
        "total_records": total_records,
        "pass": pass_count,
        "revise": revise_count,
        "reject": reject_count,
        "unclear": unclear_count,
        "fully_checked_records": fully_checked_records,
        "decision": decision,
    }
    return decision, reasons, compact


def evaluate_quality_gate(
    manifest: dict[str, Any],
    manual_review_summaries: Iterable[dict[str, Any]] | None = None,
    *,
    require_manual_review: bool = True,
) -> dict[str, Any]:
    """Evaluate whether a KamiKnows run should be accepted, revised or rejected.

    Args:
        manifest: Parsed ``dataset_manifest.json``.
        manual_review_summaries: Parsed review summary JSON objects. If omitted,
            the decision is based only on the manifest unless ``require_manual_review``
            is true.
        require_manual_review: When true, at least one manual review summary is
            required to return ``ACCEPT``.
    """
    if not isinstance(manifest, dict):
        raise QualityGateError("manifest must be a dictionary")

    reasons: list[str] = []
    decision, manifest_reasons = _evaluate_manifest(manifest)
    reasons.extend(manifest_reasons)

    summaries = list(manual_review_summaries or [])
    manual_results: list[dict[str, Any]] = []

    if require_manual_review and not summaries:
        decision = _max_decision(decision, "REVISE")
        reasons.append("manual review is required but no manual review summaries were provided")

    total_reviewed_records = 0
    aggregate_outcomes = {"pass": 0, "revise": 0, "reject": 0, "unclear": 0}

    for index, summary in enumerate(summaries, start=1):
        if not isinstance(summary, dict):
            raise QualityGateError(f"manual review summary {index} must be a dictionary")
        summary_decision, summary_reasons, compact = _evaluate_manual_summary(summary, index)
        decision = _max_decision(decision, summary_decision)
        reasons.extend(summary_reasons)
        manual_results.append(compact)
        total_reviewed_records += compact["total_records"]
        for key in aggregate_outcomes:
            aggregate_outcomes[key] += compact[key]

    if not reasons:
        reasons.append("manifest and manual review summaries passed the quality gate")

    return {
        "quality_gate_version": QUALITY_GATE_VERSION,
        "decision": decision,
        "scope": "formal traceability plus human manual review summary; no automatic scientific correctness judgment",
        "require_manual_review": require_manual_review,
        "reasons": reasons,
        "manifest": {
            "status": manifest.get("status"),
            "name": manifest.get("name"),
            "missing_paths": manifest.get("missing_paths", []),
            "file_count": len(manifest.get("files", []) or []),
        },
        "manual_review": {
            "summary_count": len(summaries),
            "total_reviewed_records": total_reviewed_records,
            "outcome_counts": aggregate_outcomes,
            "summaries": manual_results,
        },
    }


def evaluate_quality_gate_from_files(
    manifest_path: str | Path,
    manual_review_summary_paths: Iterable[str | Path] | None = None,
    *,
    require_manual_review: bool = True,
    discover_from_manifest: bool = True,
) -> dict[str, Any]:
    """Read quality gate inputs from files and return the gate report."""
    manifest_path = Path(manifest_path)
    manifest = read_json_object(manifest_path)

    explicit_paths = [Path(path) for path in manual_review_summary_paths or []]
    paths = list(explicit_paths)
    discovered_missing: list[Path] = []
    if discover_from_manifest and not paths:
        discovered_paths = discover_manual_review_summary_paths(manifest)
        for path in discovered_paths:
            if path.exists():
                paths.append(path)
            else:
                discovered_missing.append(path)

    # Explicitly supplied paths are strict inputs and must be readable.
    # Manifest-discovered paths may be absent while a review is still pending;
    # in that case the gate report should be written with a REVISE decision
    # rather than failing before producing an auditable report.
    summaries = [read_json_object(path) for path in paths]
    report = evaluate_quality_gate(
        manifest,
        summaries,
        require_manual_review=require_manual_review,
    )
    if discovered_missing:
        report["reasons"].append(
            "manual review summaries are declared in the manifest but not present: "
            + ", ".join(str(path) for path in discovered_missing)
        )
        if report["decision"] == "ACCEPT":
            report["decision"] = "REVISE"

    report["input_files"] = {
        "manifest": str(manifest_path),
        "manual_review_summaries": [str(path) for path in paths],
        "missing_discovered_manual_review_summaries": [
            str(path) for path in discovered_missing
        ],
    }
    return report
