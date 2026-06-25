"""Parse completed manual quality checklists for KamiKnows.

The checklist Markdown is created by ``create_manual_quality_checklist.py`` and
then edited by a human reviewer. This module extracts a small machine-readable
summary from that edited Markdown.

It intentionally does not judge scientific correctness automatically. It only
records the human review state: checked items, notes and outcome labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

ALLOWED_REVIEW_OUTCOMES = {"pass", "revise", "reject", "unclear"}
REQUIRED_CHECKS = {
    "main_claim_supported": "main_claim",
    "method_supported": "method",
    "limitations_supported": "limitations",
    "no_unsupported_claim": "No unsupported scientific claim",
    "confidence_plausible": "confidence label",
}


class ManualReviewError(RuntimeError):
    """Raised when a completed manual review cannot be parsed."""


@dataclass(frozen=True, slots=True)
class ManualReviewRecord:
    """One parsed manual review section."""

    record_index: int
    title: str
    arxiv_id: str
    checks: dict[str, bool]
    notes: str
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "title": self.title,
            "arxiv_id": self.arxiv_id,
            "checks": dict(self.checks),
            "notes": self.notes,
            "outcome": self.outcome,
        }


def _split_record_sections(markdown_text: str) -> list[tuple[int, str, str]]:
    matches = list(re.finditer(r"^## Record\s+(\d+):\s*(.*?)\s*$", markdown_text, re.M))
    sections: list[tuple[int, str, str]] = []
    for position, match in enumerate(matches):
        start = match.start()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(markdown_text)
        record_index = int(match.group(1))
        title = match.group(2).strip()
        sections.append((record_index, title, markdown_text[start:end]))
    return sections


def _extract_arxiv_id(section_text: str) -> str:
    match = re.search(r"^- arXiv ID:\s*`?([^`\n]+)`?\s*$", section_text, re.M)
    if not match:
        return ""
    value = match.group(1).strip()
    return "" if value == "-" else value


def _checkbox_checked(section_text: str, marker: str) -> bool:
    pattern = re.compile(r"^- \[(?P<state>[ xX])\]\s+.*" + re.escape(marker) + r".*$", re.M)
    match = pattern.search(section_text)
    if not match:
        return False
    return match.group("state").lower() == "x"


def _extract_notes(section_text: str) -> str:
    # Preferred parse: user edits the existing line to '- [x] Notes: text'.
    match = re.search(r"^- \[[ xX]\]\s+Notes:\s*(.*?)\s*$", section_text, re.M)
    if match:
        return match.group(1).strip()

    # Fallback parse: user adds a small notes block after the checklist line.
    start = re.search(r"^- \[[ xX]\]\s+Notes:\s*$", section_text, re.M)
    if not start:
        return ""
    remainder = section_text[start.end() :]
    outcome = re.search(r"^Review outcome:", remainder, re.M)
    notes_block = remainder[: outcome.start()] if outcome else remainder
    return notes_block.strip()


def _extract_outcome(section_text: str) -> str:
    match = re.search(r"^Review outcome:\s*`?([^`\n]+)`?\s*$", section_text, re.M)
    if not match:
        return "unclear"

    raw = match.group(1).strip().lower()
    if raw in ALLOWED_REVIEW_OUTCOMES:
        return raw

    # Unedited template line: `pass | revise | reject | unclear`.
    if "|" in raw:
        return "unclear"

    return "unclear"


def parse_manual_quality_checklist(markdown_text: str) -> list[ManualReviewRecord]:
    """Parse completed checklist Markdown into review records."""
    if not markdown_text.strip():
        raise ManualReviewError("manual quality checklist is empty")

    sections = _split_record_sections(markdown_text)
    if not sections:
        raise ManualReviewError("no '## Record N:' sections found")

    records: list[ManualReviewRecord] = []
    for record_index, title, section_text in sections:
        checks = {
            key: _checkbox_checked(section_text, marker)
            for key, marker in REQUIRED_CHECKS.items()
        }
        records.append(
            ManualReviewRecord(
                record_index=record_index,
                title=title,
                arxiv_id=_extract_arxiv_id(section_text),
                checks=checks,
                notes=_extract_notes(section_text),
                outcome=_extract_outcome(section_text),
            )
        )
    return records


def summarize_manual_review_records(
    records: list[ManualReviewRecord],
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact summary from parsed manual review records."""
    if not records:
        raise ManualReviewError("no manual review records to summarize")

    outcome_counts = {outcome: 0 for outcome in sorted(ALLOWED_REVIEW_OUTCOMES)}
    check_pass_counts = {key: 0 for key in REQUIRED_CHECKS}

    for record in records:
        outcome_counts[record.outcome] = outcome_counts.get(record.outcome, 0) + 1
        for key, passed in record.checks.items():
            if passed:
                check_pass_counts[key] += 1

    total_records = len(records)
    fully_checked_records = sum(
        1 for record in records if all(record.checks.get(key, False) for key in REQUIRED_CHECKS)
    )
    pass_records = outcome_counts.get("pass", 0)

    if pass_records == total_records and fully_checked_records == total_records:
        status = "PASS"
    elif outcome_counts.get("reject", 0) > 0:
        status = "REVIEW_REJECTED_RECORDS"
    elif outcome_counts.get("revise", 0) > 0:
        status = "REVIEW_REVISIONS_NEEDED"
    else:
        status = "REVIEW_INCOMPLETE"

    return {
        "review_summary_version": "manual_review_summary_v0",
        "source_path": str(source_path) if source_path is not None else None,
        "total_records": total_records,
        "status": status,
        "outcome_counts": outcome_counts,
        "check_pass_counts": check_pass_counts,
        "fully_checked_records": fully_checked_records,
        "records": [record.to_dict() for record in records],
    }


def summarize_manual_quality_checklist_file(path: str | Path) -> dict[str, Any]:
    """Read a completed Markdown checklist and return a review summary."""
    checklist_path = Path(path)
    markdown_text = checklist_path.read_text(encoding="utf-8")
    records = parse_manual_quality_checklist(markdown_text)
    return summarize_manual_review_records(records, source_path=checklist_path)
