"""Reproducible inventory run with per-paper failures and immutable input assets."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from kamiknows.common import (create_output_dir, read_json, read_jsonl, read_manifest,
                             write_json, write_jsonl, write_manifest)

from .assets import local_path
from .inventory import CLASSIFICATION_METHOD, FigureClassifier, FigureExtractor
from .latex import issue
from .records import CLASSIFIED_SCHEMA_VERSION, SCHEMA_VERSION, FigureRecord, PanelRecord


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


# Classification reads the saved inventory only. Validation happens before creating
# output so malformed schemas and broken links cannot leave a plausible partial run.
def _inventory_records(figures: list[dict[str, Any]], panels: list[dict[str, Any]]) -> tuple[
        list[FigureRecord], list[PanelRecord]]:
    for kind, records in (("figure", figures), ("panel", panels)):
        ids = set()
        for record in records:
            identifier = record.get(f"{kind}_id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"invalid or duplicate {kind}_id")
            ids.add(identifier)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{identifier}: expected inventory schema {SCHEMA_VERSION}")
            for field in ("paper_id", "figure_id"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"{identifier}: invalid {field}")
            for field in ("caption", "caption_raw", "latex_label" if kind == "figure" else "panel_label"):
                if record.get(field) is not None and not isinstance(record[field], str):
                    raise ValueError(f"{identifier}: invalid {field}")
            for field in ("latex_labels", "panels" if kind == "figure" else "source_asset_ids"):
                value = record.get(field, [] if field == "latex_labels" else None)
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    raise ValueError(f"{identifier}: invalid {field}")
            if not isinstance(record.get("warnings", []), list):
                raise ValueError(f"{identifier}: invalid warnings")
            if kind == "figure":
                captions = record.get("captions", [])
                if not isinstance(captions, list) or any(
                        not isinstance(c, dict) or any(c.get(key) is not None and not isinstance(c[key], str)
                                                      for key in ("normalized", "raw")) for c in captions):
                    raise ValueError(f"{identifier}: invalid captions")
                if not isinstance(record.get("source_assets"), list):
                    raise ValueError(f"{identifier}: invalid source_assets")
                for asset in record["source_assets"]:
                    if not isinstance(asset, dict) or not isinstance(asset.get("asset_id"), str):
                        raise ValueError(f"{identifier}: invalid asset record")
                    for field in ("reference", "path", "panel_id"):
                        if asset.get(field) is not None and not isinstance(asset[field], str):
                            raise ValueError(f"{identifier}: invalid asset {field}")
                if any(not isinstance(record.get(field), dict) for field in ("provenance", "source_hashes")):
                    raise ValueError(f"{identifier}: invalid provenance or source_hashes")
            elif not isinstance(record.get("source_region"), dict):
                raise ValueError(f"{identifier}: invalid source_region")
    try:
        figure_records = [FigureRecord(**record) for record in figures]
        panel_records = [PanelRecord(**record) for record in panels]
    except TypeError as exc:
        raise ValueError(f"invalid inventory record fields: {exc}") from exc
    figure_map = {f.figure_id: f for f in figure_records}
    for panel in panel_records:
        parent = figure_map.get(panel.figure_id)
        if parent is None or parent.paper_id != panel.paper_id or panel.panel_id not in parent.panels:
            raise ValueError(f"{panel.panel_id}: broken figure/paper link")
    for figure in figure_records:
        linked = [p for p in panel_records if p.figure_id == figure.figure_id]
        if len(set(figure.panels)) != len(figure.panels) or set(figure.panels) != {p.panel_id for p in linked}:
            raise ValueError(f"{figure.figure_id}: broken panel links")
        assets = {a["asset_id"]: a for a in figure.source_assets}
        if len(assets) != len(figure.source_assets):
            raise ValueError(f"{figure.figure_id}: duplicate asset IDs")
        for panel in linked:
            if (len(set(panel.source_asset_ids)) != len(panel.source_asset_ids)
                    or any(identifier not in assets or assets[identifier].get("panel_id") != panel.panel_id
                           for identifier in panel.source_asset_ids)):
                raise ValueError(f"{panel.panel_id}: broken asset links")
        for asset in figure.source_assets:
            owner = asset.get("panel_id")
            if owner is not None and not any(p.panel_id == owner and asset["asset_id"] in p.source_asset_ids for p in linked):
                raise ValueError(f"{figure.figure_id}: broken asset owner")
    return figure_records, panel_records


def run_classification(*, inventory_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Classify inventory v1.0 metadata into a separate v1.1 run, preserving IDs."""
    inventory_dir = Path(inventory_dir).resolve()
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(inventory_dir):
        raise ValueError("output directory must not be inside the input inventory run")
    inputs = {name: local_path(inventory_dir, inventory_dir / filename) for name, filename in (
        ("inventory_manifest", "manifest.json"), ("figures", "figures.jsonl"),
        ("panels", "panels.jsonl"), ("inventory_summary", "figure_summary.json"),
        ("inventory_diagnostics", "diagnostics.jsonl"))}
    manifest = read_manifest(inventory_dir, "extract_plots")
    params = manifest.get("params", {})
    if not isinstance(params, dict) or params.get("mode") != "inventory" or params.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("classification requires an inventory run with schema 1.0")
    original_summary = read_json(inputs["inventory_summary"])
    if (not isinstance(original_summary, dict) or original_summary.get("schema_version") != SCHEMA_VERSION
            or original_summary.get("mode") != "inventory" or original_summary != manifest.get("summary")):
        raise ValueError("inventory summary/schema does not match its manifest")
    figures, panels = _inventory_records(read_jsonl(inputs["figures"]), read_jsonl(inputs["panels"]))
    if original_summary.get("figures") != len(figures) or original_summary.get("panels") != len(panels):
        raise ValueError("inventory record counts do not match the summary")
    inherited = read_jsonl(inputs["inventory_diagnostics"])
    if any(d.get("schema_version") != SCHEMA_VERSION or not isinstance(d.get("code"), str) for d in inherited):
        raise ValueError("invalid inventory diagnostics schema")
    diagnostics = [{**d, "schema_version": CLASSIFIED_SCHEMA_VERSION, "origin_mode": "inventory"} for d in inherited]
    classified_figures = []
    classified_panels = {}
    classifier = FigureClassifier()
    for figure in figures:
        result, children = classifier.classify(figure, [p for p in panels if p.figure_id == figure.figure_id])
        classified_figures.append(result.to_dict())
        classified_panels.update({p.panel_id: p.to_dict() for p in children})
        for record in [result, *children]:
            evidence = record.classification_evidence
            context = {"paper_id": record.paper_id, "figure_id": record.figure_id,
                       "schema_version": CLASSIFIED_SCHEMA_VERSION, "origin_mode": "classify"}
            if isinstance(record, PanelRecord):
                context["panel_id"] = record.panel_id
            reasons = []
            if evidence["assigned_class"] == "unknown":
                reasons.append(("classification_unknown", evidence["reason"]))
            if evidence["contradictory_indicators"]:
                reasons.append(("classification_conflict", "Conflicting textual evidence requires manual review."))
            if evidence.get("visual_panel_detection_reasons"):
                reasons.append(("visual_panel_detection_candidate", "; ".join(evidence["visual_panel_detection_reasons"])))
            diagnostics.extend({**issue(code, message), **context} for code, message in reasons)
    # Preserve the input panel order, even when records are not grouped by figure.
    panel_output = [classified_panels[p.panel_id] for p in panels]
    all_records = [*classified_figures, *panel_output]
    summary = {
        "schema_version": CLASSIFIED_SCHEMA_VERSION, "mode": "classify", "classification_method": CLASSIFICATION_METHOD,
        "figures": len(figures), "panels": len(panels),
        "figure_type_counts": dict(Counter(f["figure_type"] for f in classified_figures)),
        "panel_type_counts": dict(Counter(p["panel_type"] for p in panel_output)),
        "unknown_figures": [f["figure_id"] for f in classified_figures if f["figure_type"] == "unknown"],
        "unknown_panels": [p["panel_id"] for p in panel_output if p["panel_type"] == "unknown"],
        "mixed_figures": [f["figure_id"] for f in classified_figures if f["figure_type"] == "mixed"],
        "contradictory_figures": [f["figure_id"] for f in classified_figures if f["classification_evidence"]["contradictory_indicators"]],
        "contradictory_panels": [p["panel_id"] for p in panel_output if p["classification_evidence"]["contradictory_indicators"]],
        "visual_panel_detection_candidates": [f["figure_id"] for f in classified_figures
                                               if f["classification_evidence"]["visual_panel_detection_reasons"]],
        "review_required": {"figures": len(figures), "panels": len(panels)},
        "review_status_counts": dict(Counter(r["classification_review_status"] for r in all_records)),
        "diagnostic_counts": dict(Counter(d["code"] for d in diagnostics)),
        "manual_review": "pending", "inventory_summary": original_summary,
    }
    output_dir = create_output_dir(output_dir)
    write_jsonl(output_dir / "figures.jsonl", classified_figures)
    write_jsonl(output_dir / "panels.jsonl", panel_output)
    write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
    write_json(output_dir / "figure_summary.json", summary)
    write_manifest(output_dir, step="extract_plots",
                   params={"mode": "classify", "schema_version": CLASSIFIED_SCHEMA_VERSION,
                           "classification_method": CLASSIFICATION_METHOD, "inventory_dir": inventory_dir},
                   inputs=inputs, summary=summary)
    return summary
