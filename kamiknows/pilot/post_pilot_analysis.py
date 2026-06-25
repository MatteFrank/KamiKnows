"""Post-pilot analysis helpers for KamiKnows.

This module summarizes the outputs of a controlled HEP pilot after extraction,
formal checks, manual review and quality gate. It does not inspect scientific
correctness directly. It turns existing machine-readable artifacts into a small
operational report: what passed, what needs revision, and what the next cycle
should do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

POST_PILOT_ANALYSIS_VERSION = "post_pilot_analysis_v0"


class PostPilotAnalysisError(RuntimeError):
    """Raised when post-pilot analysis inputs cannot be read."""


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    json_path = Path(path)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PostPilotAnalysisError(f"invalid JSON file {json_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PostPilotAnalysisError(f"JSON file must contain an object: {json_path}")
    return data


def _resolve_path(raw_path: str | Path, *, manifest_path: Path | None = None) -> Path:
    """Resolve a manifest path robustly for local tutorial runs.

    Older scripts may store paths relative to the repository root, while a user
    often passes a manifest from inside an output directory. This helper tries:

    1. the path as written;
    2. the same file name next to the manifest;
    3. the path relative to the manifest directory.
    """
    path = Path(raw_path)
    if path.is_absolute() or path.exists() or manifest_path is None:
        return path

    sibling = manifest_path.parent / path.name
    if sibling.exists():
        return sibling

    relative_to_manifest = manifest_path.parent / path
    if relative_to_manifest.exists():
        return relative_to_manifest

    return path


def files_by_role(
    manifest: dict[str, Any],
    *,
    role: str,
    manifest_path: str | Path | None = None,
) -> list[Path]:
    """Return manifest files matching one role."""
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise PostPilotAnalysisError("dataset manifest 'files' must be a list")

    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else None
    paths: list[Path] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != role:
            continue
        raw_path = entry.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            paths.append(_resolve_path(raw_path, manifest_path=resolved_manifest_path))
    return paths


def role_counts(manifest: dict[str, Any]) -> dict[str, int]:
    """Count file roles declared in a manifest."""
    counts: dict[str, int] = {}
    for entry in manifest.get("files", []) or []:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "<missing>")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _read_existing_json_objects(paths: Iterable[Path]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            objects.append(read_json_object(path))
    return objects


def _summarize_formal_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    total_records = 0
    for summary in summaries:
        status = str(summary.get("evaluation_status") or "<missing>")
        statuses[status] = statuses.get(status, 0) + 1
        total_records += int(summary.get("total_records", 0) or 0)
    return {
        "summary_count": len(summaries),
        "evaluation_status_counts": statuses,
        "total_records_reported": total_records,
    }


def _summarize_manual_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    outcome_counts = {"pass": 0, "revise": 0, "reject": 0, "unclear": 0}
    total_records = 0
    fully_checked_records = 0

    for review in reviews:
        status = str(review.get("status") or "<missing>")
        status_counts[status] = status_counts.get(status, 0) + 1
        total_records += int(review.get("total_records", 0) or 0)
        fully_checked_records += int(review.get("fully_checked_records", 0) or 0)
        counts = review.get("outcome_counts", {})
        if isinstance(counts, dict):
            for key in outcome_counts:
                outcome_counts[key] += int(counts.get(key, 0) or 0)

    return {
        "summary_count": len(reviews),
        "status_counts": status_counts,
        "total_reviewed_records": total_records,
        "fully_checked_records": fully_checked_records,
        "outcome_counts": outcome_counts,
    }


def _recommendation(
    *,
    manifest: dict[str, Any],
    quality_gate_report: dict[str, Any] | None,
    manual_review_summary_count: int,
) -> tuple[str, list[str]]:
    """Return operational recommendation and next actions."""
    next_actions: list[str] = []

    if manifest.get("status") != "PASS" or manifest.get("missing_paths"):
        return "FIX_TRACEABILITY", [
            "Fix missing files or paths reported by dataset_manifest.json.",
            "Regenerate dataset_manifest.json after files are present.",
        ]

    if quality_gate_report is None:
        return "RUN_QUALITY_GATE", [
            "Run scripts/run_quality_gate.py on the pilot dataset_manifest.json.",
            "Use --fail-on-non-accept only after the review workflow is complete.",
        ]

    decision = quality_gate_report.get("decision")
    if decision == "ACCEPT":
        next_actions.extend(
            [
                "Freeze the accepted pilot outputs as the baseline run.",
                "Prepare the next pilot cycle: add a second model or expand the schema only one change at a time.",
                "Do not start LoRA/QLoRA until multiple accepted extraction runs exist.",
            ]
        )
        return "READY_FOR_NEXT_PILOT_CYCLE", next_actions

    if decision == "REJECT":
        next_actions.extend(
            [
                "Inspect rejected records in the manual review summary.",
                "Fix prompt/schema or filtering before processing more papers.",
                "Rerun the pilot on the same frozen metadata after changes.",
            ]
        )
        return "STOP_AND_FIX", next_actions

    if decision == "REVISE":
        if manual_review_summary_count == 0:
            next_actions.extend(
                [
                    "Complete the manual quality checklist on the pilot sample.",
                    "Run scripts/summarize_manual_quality_review.py.",
                    "Regenerate dataset_manifest.json and rerun the quality gate.",
                ]
            )
        else:
            next_actions.extend(
                [
                    "Inspect review outcomes marked revise or unclear.",
                    "Adjust prompt/schema only if the same error repeats across records.",
                    "Rerun on the same frozen metadata before scaling to more papers.",
                ]
            )
        return "REVISE_BEFORE_SCALING", next_actions

    return "INVESTIGATE_GATE_STATUS", [
        f"Unexpected quality gate decision: {decision!r}.",
        "Open quality_gate_report.json and inspect reasons manually.",
    ]


def build_post_pilot_analysis(
    *,
    manifest: dict[str, Any],
    manifest_path: str | Path | None = None,
    quality_gate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a post-pilot operational analysis report."""
    if not isinstance(manifest, dict):
        raise PostPilotAnalysisError("manifest must be a dictionary")

    manifest_path_obj = Path(manifest_path) if manifest_path is not None else None

    formal_summary_paths = files_by_role(
        manifest,
        role="summary",
        manifest_path=manifest_path_obj,
    )
    manual_review_paths = files_by_role(
        manifest,
        role="manual_review_summary",
        manifest_path=manifest_path_obj,
    )
    jsonl_paths = files_by_role(
        manifest,
        role="jsonl_extractions",
        manifest_path=manifest_path_obj,
    )
    metadata_paths = files_by_role(
        manifest,
        role="metadata",
        manifest_path=manifest_path_obj,
    )
    checklist_paths = files_by_role(
        manifest,
        role="manual_quality_checklist",
        manifest_path=manifest_path_obj,
    )
    pilot_report_paths = files_by_role(
        manifest,
        role="pilot_report",
        manifest_path=manifest_path_obj,
    )

    formal_summaries = _read_existing_json_objects(formal_summary_paths)
    manual_reviews = _read_existing_json_objects(manual_review_paths)

    if quality_gate_report is None and manifest_path_obj is not None:
        candidate = manifest_path_obj.parent / "quality_gate_report.json"
        if candidate.exists():
            quality_gate_report = read_json_object(candidate)

    recommendation, next_actions = _recommendation(
        manifest=manifest,
        quality_gate_report=quality_gate_report,
        manual_review_summary_count=len(manual_reviews),
    )

    jsonl_record_count = 0
    for entry in manifest.get("files", []) or []:
        if isinstance(entry, dict) and entry.get("role") == "jsonl_extractions":
            jsonl_record_count += int(entry.get("record_count", 0) or 0)

    quality_gate_compact = None
    if quality_gate_report is not None:
        quality_gate_compact = {
            "decision": quality_gate_report.get("decision"),
            "reasons": quality_gate_report.get("reasons", []),
        }

    return {
        "post_pilot_analysis_version": POST_PILOT_ANALYSIS_VERSION,
        "scope": "operational post-pilot status; no automatic scientific correctness judgment",
        "recommendation": recommendation,
        "next_actions": next_actions,
        "manifest": {
            "path": str(manifest_path) if manifest_path is not None else None,
            "name": manifest.get("name"),
            "status": manifest.get("status"),
            "file_count": len(manifest.get("files", []) or []),
            "missing_paths": manifest.get("missing_paths", []),
            "role_counts": role_counts(manifest),
            "run_context": manifest.get("run_context", {}),
        },
        "artifacts": {
            "metadata_files": [str(path) for path in metadata_paths],
            "jsonl_files": [str(path) for path in jsonl_paths],
            "formal_summary_files": [str(path) for path in formal_summary_paths],
            "manual_quality_checklists": [str(path) for path in checklist_paths],
            "manual_review_summaries": [str(path) for path in manual_review_paths],
            "pilot_reports": [str(path) for path in pilot_report_paths],
            "jsonl_record_count_from_manifest": jsonl_record_count,
        },
        "formal_summary": _summarize_formal_summaries(formal_summaries),
        "manual_review": _summarize_manual_reviews(manual_reviews),
        "quality_gate": quality_gate_compact,
    }


def build_post_pilot_analysis_from_manifest(
    manifest_path: str | Path,
    *,
    quality_gate_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read manifest and optional gate report, then build post-pilot analysis."""
    manifest_path = Path(manifest_path)
    manifest = read_json_object(manifest_path)
    quality_gate_report = None
    if quality_gate_report_path is not None:
        quality_gate_report = read_json_object(quality_gate_report_path)
    return build_post_pilot_analysis(
        manifest=manifest,
        manifest_path=manifest_path,
        quality_gate_report=quality_gate_report,
    )


def write_post_pilot_analysis(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a post-pilot analysis JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
