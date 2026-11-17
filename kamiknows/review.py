"""Step 5: manual review of generated answers.

``prepare`` joins the answers of one or more generation runs with their
questions (including the reference answer of the questions file) and cited
passages into one CSV, with empty reviewer columns. ``summarize`` counts the
labels of a completed CSV. Automatic checks (JSON validity, citation labels)
never replace this review.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kamiknows.common import create_output_dir, read_jsonl, read_manifest, write_manifest

REVIEW_LABELS = {
    "reviewer_correctness": ["correct", "partially_correct", "incorrect", "not_applicable"],
    "reviewer_citation_support": ["supported", "partially_supported", "unsupported", "not_applicable"],
    "reviewer_abstention": ["appropriate", "should_have_answered", "should_have_abstained", "not_applicable"],
}
COLUMNS = [
    "run",
    "question_id",
    "pair_id",
    "source_variant",
    "question",
    "expected_behavior",
    "reference_answer",
    "required_answer_points",
    "reference_evidence",
    "answer_status",
    "answer",
    "evidence",
    "limitations",
    "citation_status",
    "warnings",
    "retrieval_hit",
    "cited_passages",
    "other_passages",
    *REVIEW_LABELS,
    "reviewer_notes",
    "review_status",
]


def _passage_text(passage: dict[str, Any], *, full: bool) -> str:
    header = f"[{passage['label']}] {passage['arxiv_id']} / {passage['section_heading']}"
    return f"{header}\n{passage['text']}" if full else header


def review_rows(qa_dir: Path) -> list[dict[str, str]]:
    manifest = read_manifest(qa_dir, "generate")
    contexts = {c["question_id"]: c for c in read_jsonl(Path(manifest["inputs"]["contexts"]["path"]))}
    rows = []
    for answer in read_jsonl(qa_dir / "answers.jsonl"):
        context = contexts[answer["question_id"]]
        annotations = context["annotations"]
        cited = set((answer.get("citation_validation") or {}).get("valid_labels", []))
        rows.append(
            {
                "run": qa_dir.name,
                "question_id": answer["question_id"],
                "pair_id": annotations.get("pair_id", ""),
                "source_variant": annotations.get("source_variant", ""),
                "question": answer["question"],
                "expected_behavior": annotations.get("expected_behavior", ""),
                "reference_answer": annotations.get("reference_answer", ""),
                "required_answer_points": "\n".join(annotations.get("required_answer_points", [])),
                "reference_evidence": "\n\n".join(
                    (
                        f"{e['arxiv_id']} / {e['section_heading']}: {e['quote']}"
                        if {"arxiv_id", "section_heading", "quote"} <= e.keys()
                        else json.dumps(e, ensure_ascii=False, sort_keys=True)
                    )
                    for e in annotations.get("reference_evidence", [])
                ),
                "answer_status": answer["answer_status"],
                "answer": answer.get("answer", ""),
                "evidence": answer.get("evidence", ""),
                "limitations": answer.get("limitations", ""),
                "citation_status": (answer.get("citation_validation") or {}).get("status", ""),
                "warnings": " ".join(answer["warnings"]),
                "retrieval_hit": str(context["evaluation"].get("hit", "")),
                "cited_passages": "\n\n".join(
                    _passage_text(p, full=True) for p in context["passages"] if p["label"] in cited
                ),
                "other_passages": "\n".join(
                    _passage_text(p, full=False) for p in context["passages"] if p["label"] not in cited
                ),
                **{column: "" for column in REVIEW_LABELS},
                "reviewer_notes": "",
                "review_status": "pending",
            }
        )
    return rows


def run_prepare(*, qa_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    rows = [row for qa_dir in qa_dirs for row in review_rows(qa_dir)]
    output_dir = create_output_dir(output_dir)
    with (output_dir / "review.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    esc = html.escape
    cards = []
    for identifier, group in grouped.items():
        answers = []
        for row in group:
            answers.append(
                f'<section><h3>{esc(row["run"])}</h3><p><b>{esc(row["answer_status"])}</b>'
                f' · citation labels: {esc(row["citation_status"])}</p><p>{esc(row["answer"])}</p>'
                f'<p>{esc(row["limitations"])}</p><details><summary>Cited evidence</summary>'
                f'<pre>{esc(row["cited_passages"])}</pre></details></section>'
            )
        cards.append(
            f'<article><h2>{esc(identifier)}</h2><p>{esc(group[0]["question"])}</p>'
            f"<details><summary>Draft evaluation answer — never supplied to the model</summary>"
            f'<p>{esc(group[0]["reference_answer"])}</p></details><div class="answers">'
            + "".join(answers)
            + "</div></article>"
        )
    (output_dir / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>QA evidence comparison</title>'
        "<style>body{font:16px system-ui;background:#f2f5f8;color:#16283c;margin:30px}"
        "article{background:white;padding:24px;margin:24px 0;border-radius:12px}"
        ".answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}"
        "section{border:1px solid #ccd5df;padding:16px;overflow-wrap:anywhere}pre{white-space:pre-wrap}"
        "h3{font-size:15px}</style><h1>QA evidence comparison</h1>"
        "<p>Human review pending. Valid citation labels do not establish factual correctness. "
        "Digitized plot values estimate displayed geometry; this comparison does not validate original scientific data.</p>"
        + "".join(cards)
        + "</html>",
        encoding="utf-8",
    )
    summary = {
        "rows": len(rows),
        "runs": [qa_dir.name for qa_dir in qa_dirs],
        "allowed_labels": REVIEW_LABELS,
    }
    write_manifest(
        output_dir,
        step="review",
        params={"qa_dirs": qa_dirs},
        inputs={f"answers_{qa_dir.name}": qa_dir / "answers.jsonl" for qa_dir in qa_dirs},
        summary=summary,
    )
    return summary


def summarize_review(csv_path: Path) -> dict[str, Any]:
    """Label counts per run; rows whose review_status is still 'pending' are counted apart."""
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result: dict[str, Any] = defaultdict(
        lambda: {"reviewed": 0, "pending": 0, **{c: Counter() for c in REVIEW_LABELS}}
    )
    invalid = []
    for row in rows:
        run = result[row["run"]]
        if row["review_status"].strip() == "pending":
            run["pending"] += 1
            continue
        run["reviewed"] += 1
        for column, allowed in REVIEW_LABELS.items():
            value = row[column].strip()
            run[column][value or "<empty>"] += 1
            if value and value not in allowed:
                invalid.append({"run": row["run"], "question_id": row["question_id"], column: value})
    return json.loads(json.dumps({"runs": result, "invalid_labels": invalid}))
