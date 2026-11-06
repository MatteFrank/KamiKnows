"""Reproducible inventory run with per-paper failures and immutable input assets."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from kamiknows.common import (create_output_dir, read_json, read_jsonl, read_manifest,
                             write_json, write_jsonl, write_manifest)

from .assets import local_path
from .inventory import FigureExtractor
from .latex import issue
from .records import SCHEMA_VERSION


def run_inventory(*, parsed_dir: Path, output_dir: Path, mode: str = "inventory") -> dict[str, Any]:
    """Inventory an existing parse run; no network, TeX execution or numerical extraction."""
    if mode != "inventory":
        raise ValueError("only mode='inventory' is implemented (Task 01)")
    parsed_dir = Path(parsed_dir).resolve()
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(parsed_dir):
        raise ValueError("output directory must not be inside the input parse run")
    manifest_path = local_path(parsed_dir, parsed_dir / "manifest.json")
    read_manifest(parsed_dir, "parse")
    papers_file = local_path(parsed_dir, parsed_dir / "papers.jsonl")
    papers = read_jsonl(papers_file)
    identifiers = [p.get("paper_id") for p in papers]
    if any(not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
           for identifier in identifiers):
        raise ValueError("papers.jsonl contains an invalid paper_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("papers.jsonl contains duplicate paper_id values")
    inputs = {"parse_manifest": manifest_path, "papers": papers_file}
    output_dir = create_output_dir(output_dir)
    figures = []
    panels = []
    diagnostics = []
    paper_results = []
    for paper in papers:
        paper_id = paper["paper_id"]
        try:
            paper_dir = local_path(parsed_dir, parsed_dir / "papers" / paper_id)
            flat_path = local_path(paper_dir, paper_dir / "flat.tex")
            source_root = local_path(paper_dir, paper_dir / "source")
            if not source_root.is_dir():
                raise FileNotFoundError(f"missing source directory for {paper_id}")
            flat_tex = flat_path.read_text(encoding="utf-8", errors="replace")
            inputs[f"{paper_id}:flat"] = flat_path
            sections_path = local_path(paper_dir, paper_dir / "sections.json")
            sections = []
            paper_diagnostics = []
            if sections_path.exists():
                inputs[f"{paper_id}:sections"] = sections_path
                sections = read_json(sections_path)
                if not isinstance(sections, list) or any(not isinstance(s, dict) or
                                                        not isinstance(s.get("section_id"), str) for s in sections):
                    raise ValueError("sections.json must contain section records with string section_id")
            else:
                paper_diagnostics.append(issue("sections_missing", "Cannot link figures to saved sections."))
            found, explicit, extracted_diagnostics, source_inputs = FigureExtractor().extract(
                paper_id=paper_id, flat_tex=flat_tex, source_root=source_root, sections=sections)
            figures.extend(f.to_dict() for f in found)
            panels.extend(p.to_dict() for p in explicit)
            paper_diagnostics.extend(extracted_diagnostics)
            diagnostics.extend({**entry, "paper_id": paper_id} for entry in paper_diagnostics)
            inputs.update({f"{paper_id}:source:{name}": path for name, path in source_inputs.items()})
            paper_results.append({"paper_id": paper_id, "parse_status": paper.get("status"),
                                  "status": "partial" if paper_diagnostics else "success", "figures": len(found)})
        except (OSError, ValueError) as exc:
            diagnostics.append({**issue("paper_input_error", f"{type(exc).__name__}: {exc}"),
                                "severity": "error", "paper_id": paper_id})
            paper_results.append({"paper_id": paper_id, "parse_status": paper.get("status"),
                                  "status": "failed", "figures": 0})
    assets = [a for figure in figures for a in figure["source_assets"]]
    summary = {
        "schema_version": SCHEMA_VERSION, "mode": mode, "papers": len(papers),
        "paper_status_counts": dict(Counter(p["status"] for p in paper_results)),
        "figures": len(figures), "panels": len(panels), "asset_references": len(assets),
        "resolved_references": sum(a["status"] == "resolved" for a in assets),
        "unresolved_references": sum(a["status"] != "resolved" for a in assets),
        "unique_assets": len({(f["paper_id"], a["path"]) for f in figures for a in f["source_assets"]
                              if a["path"] is not None}),
        "figure_status_counts": dict(Counter(f["status"] for f in figures)),
        "diagnostic_counts": dict(Counter(d["code"] for d in diagnostics)),
        "paper_results": paper_results, "manual_review": "pending",
    }
    write_jsonl(output_dir / "figures.jsonl", figures)
    write_jsonl(output_dir / "panels.jsonl", panels)
    write_jsonl(output_dir / "diagnostics.jsonl", [{"schema_version": SCHEMA_VERSION, **d} for d in diagnostics])
    write_json(output_dir / "figure_summary.json", summary)
    write_manifest(output_dir, step="extract_plots",
                   params={"mode": mode, "schema_version": SCHEMA_VERSION, "parsed_dir": parsed_dir},
                   inputs=inputs, summary=summary)
    return summary
