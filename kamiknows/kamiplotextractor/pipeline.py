"""Reproducible inventory run with per-paper failures and immutable input assets."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from kamiknows.common import (create_output_dir, read_json, read_jsonl, read_manifest,
                             sha256_file, write_json, write_jsonl, write_manifest)

from .assets import local_path
from .inventory import CLASSIFICATION_METHOD, FigureClassifier, FigureExtractor
from .latex import issue
from .records import (CLASSIFIED_SCHEMA_VERSION, DIGITIZED_SCHEMA_VERSION, NUMERICAL_SCHEMA_VERSION,
                      SCHEMA_VERSION, FigureRecord, PanelRecord, PlotDataRecord)


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


def _numerical_run(directory: Path, mode: str, schema: str, filename: str, inputs: dict) -> tuple[dict,list]:
    """Read numerical runs by role; record both manifest and payload hashes."""
    manifest_path=local_path(directory,directory/"manifest.json")
    manifest=read_manifest(directory,"extract_plots")
    if manifest.get("params",{}).get("mode")!=mode or manifest["params"].get("schema_version")!=schema:
        raise ValueError(f"expected {mode} run with schema {schema}")
    path=local_path(directory,directory/filename)
    records=read_jsonl(path)
    if any(r.get("schema_version")!=schema for r in records):
        raise ValueError("numerical record schema mismatch")
    if len(records)!=manifest["summary"].get("records"):
        raise ValueError("numerical record count differs from manifest")
    digest=manifest["summary"].get("data_sha256")
    if digest and sha256_file(path)!=digest:
        raise ValueError("numerical payload hash mismatch")
    inputs[mode+"_manifest"],inputs[mode+"_data"]=manifest_path,path
    return manifest,records


def run_digitization(*, classified_dir: Path, parsed_dir: Path, output_dir: Path,
                     backend: str = "auto", calibration_file: Path | None = None,
                     imports_dir: Path | None = None, recovery_dir: Path | None = None,
                     inventory_dir: Path | None = None, selection: str = "unresolved",
                     raster_dpi: int = 144, include_generic: bool = False) -> dict:
    """Reconstruct selected visual geometry in a new, provenance-linked schema 1.3 run."""
    import hashlib
    import json
    import time
    from .digitization import BACKENDS, Backend, Calibration, ExtractionError, load_config, raster_image
    from .validation import compare_series, overlay, structural_check

    if backend not in {*BACKENDS,"auto","compare"} or selection not in {"all","unresolved"}:
        raise ValueError("invalid backend/selection")
    if type(raster_dpi) is not int or not 72<=raster_dpi<=300:
        raise ValueError("raster_dpi must be an integer between 72 and 300")
    classified_dir,parsed_dir,output_dir=Path(classified_dir).resolve(),Path(parsed_dir).resolve(),Path(output_dir)
    figures,panels,papers,inputs,original,upstream=_recovery_inputs(classified_dir,parsed_dir,inventory_dir)
    calibrations=load_config(Path(calibration_file),"plots") if calibration_file else []
    exports=[]
    if imports_dir:
        imports_dir=Path(imports_dir).resolve()
        mapping=local_path(imports_dir,imports_dir/"imports.json")
        exports=load_config(mapping,"exports");inputs["imports_mapping"]=mapping
    if calibration_file:inputs["calibration_file"]=Path(calibration_file)
    identities={(f.figure_id,a["asset_id"],a.get("panel_id")): (f,a) for f in figures for a in f.source_assets}
    configured={}
    for c in calibrations:
        key=(c.get("figure_id"),c.get("asset_id"),c.get("panel_id"))
        if key not in identities or key in configured:
            raise ValueError("unknown or duplicate figure/panel/asset calibration mapping")
        configured[key]=c
    export_ids=set()
    for e in exports:
        key=(e.get("figure_id"),e.get("asset_id"),e.get("panel_id"))
        if key not in identities or e.get("paper_id")!=identities[key][0].paper_id:
            raise ValueError("incorrect figure/paper/panel/asset export mapping")
        if e.get("backend") not in {"chartdetective","plotsift"}:
            raise ValueError("import backend must be chartdetective or plotsift")
        identity=(*key,e["backend"],e.get("series_id"))
        if identity in export_ids:
            raise ValueError("ambiguous duplicate export series mapping")
        export_ids.add(identity)
        if not isinstance(e.get("calibration"),dict):
            raise ValueError("external export requires calibration metadata")
    originals=[]
    if recovery_dir:
        recovery_dir=Path(recovery_dir).resolve()
        previous,originals=_numerical_run(recovery_dir,"recover",NUMERICAL_SCHEMA_VERSION,"plot_data.jsonl",inputs)
        if previous["inputs"].get("classified_manifest",{}).get("sha256")!=sha256_file(inputs["classified_manifest"]):
            raise ValueError("recovery and digitization classified lineage differs")
    for root in (classified_dir,parsed_dir,upstream,imports_dir,recovery_dir):
        if root and output_dir.resolve().is_relative_to(root):
            raise ValueError("output directory must not be inside an input run/source directory")
    if calibration_file and output_dir.resolve().is_relative_to(Path(calibration_file).resolve()):
        raise ValueError("output directory overlaps calibration input")
    output_dir=create_output_dir(output_dir)
    records=[];diagnostics=[];source_roots={};source_errors={}
    for paper in papers:
        try:
            _,root=_verify_paper_sources(parsed_dir,paper["paper_id"],original["inputs"],inputs)
            source_roots[paper["paper_id"]]=root
        except (OSError,ValueError) as exc:
            source_errors[paper["paper_id"]]=str(exc)
    capabilities={name:Backend(name).capabilities() for name in BACKENDS}
    for figure in figures:
        # An unresolved includegraphics still has an identity and gets a diagnostic record.
        for asset in figure.source_assets:
            key=(figure.figure_id,asset["asset_id"],asset.get("panel_id"))
            c=configured.get(key)
            mapped=[e for e in exports if (e["figure_id"],e["asset_id"],e.get("panel_id"))==key]
            fmt=asset.get("format","").lower()
            eligible="pymupdf" if fmt=="pdf" else "plotdigitizer" if fmt in {"png","jpg","jpeg","webp"} else None
            # Auto never guesses that a raster contains one trajectory or that paths are distinguishable.
            if backend=="auto":
                if c and eligible=="pymupdf" and c.get("series") and capabilities[eligible]["available"]:chosen=[eligible]
                elif c and eligible=="plotdigitizer" and c.get("single_curve_evidence") and capabilities[eligible]["available"]:chosen=[eligible]
                elif mapped:chosen=[sorted({e["backend"] for e in mapped})[0]]
                else:chosen=[eligible or "auto"]
            else:chosen=list(BACKENDS) if backend=="compare" else [backend]
            panel=next((p for p in panels if p.panel_id==asset.get("panel_id")),None)
            generic=(panel.panel_type=="generic" if panel else figure.figure_type=="generic")
            verified=[r for r in originals if r.get("figure_id")==figure.figure_id and r.get("panel_id")==asset.get("panel_id")
                      and r.get("association",{}).get("state")=="verified"
                      and r.get("source_method") in {"original_source_data","original_published_data"}
                      and r.get("validation_status")=="structural_checks_passed" and r.get("extraction_status")=="success"
                      and r.get("series") and all(s.get("x") and len(s["x"])==len(s.get("y",[]))
                                                  and all(v is not None for v in [*s["x"],*s["y"]]) for s in r["series"])
                      and (r.get("provenance",{}).get("asset_sha256")==asset.get("sha256")
                           or r["data_id"] in (c or {}).get("original_data_ids",[]))]
            for name in chosen:
                started=time.perf_counter()
                entry_ids=sorted([e for e in mapped if e["backend"]==name],key=lambda e:e["series_id"])
                identity=json.dumps([figure.paper_id,*key,asset.get("sha256"),name,c,entry_ids],sort_keys=True)
                data_id="digitized_"+hashlib.sha256(identity.encode()).hexdigest()[:24]
                record=PlotDataRecord(figure.paper_id,figure.figure_id,asset.get("panel_id"),data_id,"unavailable",
                    {"state":"verified","kind":"inventory_asset","asset_id":asset["asset_id"]},
                    {"figure_type":figure.figure_type,"panel_type":panel.panel_type if panel else None,
                     "method":figure.classification_method,"evidence":figure.classification_evidence},
                    {"asset_id":asset["asset_id"],"asset_sha256":asset.get("sha256"),"source_path":asset.get("path"),
                     "latex_layout":asset.get("layout",{}),"classified_manifest_sha256":sha256_file(inputs["classified_manifest"]),
                     "inventory_manifest_sha256":sha256_file(inputs["upstream_inventory_manifest"]),
                     "manual_review":"pending"},backend=name,backend_version=str(capabilities.get(name,{}).get("version")),
                    schema_version=DIGITIZED_SCHEMA_VERSION)
                record.quality={"structural_validity":"not_checked","geometry_calibration":"not_checked",
                                "scientific_agreement":"not_evaluated","manual_review":"pending","flags":[]}
                work=output_dir/"overlays"/data_id
                try:
                    if figure.paper_id in source_errors:
                        raise ExtractionError("source_mismatch",source_errors[figure.paper_id])
                    if generic and not include_generic:raise ExtractionError("skipped_generic","generic classification excluded")
                    if verified and selection=="unresolved":
                        record.provenance["original_data_ids"]=[r["data_id"] for r in verified]
                        raise ExtractionError("skipped_original","explicitly asset-associated original vectors available; scientific review still pending")
                    if asset.get("status")!="resolved":raise ExtractionError("unsupported","unresolved source asset")
                    path=local_path(source_roots[figure.paper_id],source_roots[figure.paper_id]/asset["path"])
                    if sha256_file(path)!=asset.get("sha256"):raise ExtractionError("source_mismatch","asset SHA256 mismatch")
                    entries=[e for e in mapped if e["backend"]==name]
                    interactive=name in {"chartdetective","plotsift"}
                    if interactive and not entries:raise ExtractionError("needs_manual_review","no explicitly mapped interactive export")
                    if name=="chartdetective" and fmt!="pdf":raise ExtractionError("unsupported","ChartDetective import requires a vector PDF source asset")
                    if not interactive and name!=eligible:raise ExtractionError("unsupported","backend does not support this asset format")
                    selected=entries[0]["calibration"] if interactive else c
                    if selected is None:raise ExtractionError("needs_calibration","explicit axis calibration is missing")
                    try:cal=Calibration(selected)
                    except (ValueError,KeyError,TypeError,OverflowError) as exc:
                        raise ExtractionError("needs_calibration",str(exc)) from exc
                    record.calibration=cal.metadata()
                    record.axes={axis:{"scale":selected[axis]["scale"],"unit":selected[axis]["unit"]} for axis in ("x","y")}
                    layout=asset.get("layout",{})
                    if layout.get("page") and str(selected.get("page",1))!=str(layout["page"]).strip("{} "):
                        raise ExtractionError("needs_manual_review","selected PDF page conflicts with includegraphics page")
                    if any(layout.get(k) is not None for k in ("trim","viewport","angle","origin")):
                        if selected.get("extraction_scope")!="source_roi":
                            raise ExtractionError("needs_manual_review","LaTeX crop/rotation requires explicit extraction_scope=source_roi; full display not reconstructed")
                        record.quality["flags"].append("source_roi_only_latex_transform_not_reconstructed")
                    if fmt!="pdf" and selected.get("page",1)!=1:raise ValueError("raster assets have only page 1")
                    work.mkdir(parents=True)
                    if interactive:
                        for entry in entries:
                            if entry["calibration"]!=selected:raise ValueError("series imports disagree on calibration")
                            if entry.get("asset_sha256")!=asset["sha256"]:raise ValueError("source asset SHA256 mismatch in import")
                            if any(entry.get(k)!=selected.get(k,entry.get(k)) for k in ("figure_id","panel_id","asset_id")):
                                raise ValueError("import calibration identity mismatch")
                            export_path=local_path(imports_dir,imports_dir/entry["file"])
                            if export_path.is_file():inputs[f"export:{data_id}:{entry['series_id']}"]=export_path
                            data,export_path=Backend(name).import_results(imports_dir,entry,cal)
                            record.series.append(data);inputs[f"export:{data_id}:{data['series_id']}"]=export_path
                        record.backend_version=entries[0]["backend_version"]
                        record.provenance["exports"]=entries
                        record.source_method="manual_digitized"
                        if fmt=="pdf":
                            from .digitization import read_pdf
                            if Backend("pymupdf").available():
                                _,record.raw_source,preview=read_pdf(path,cal,raster_dpi)
                                (work/"source.png").write_bytes(preview)
                            else:record.quality["flags"].append("source_bounds_and_overlay_unverified_without_pymupdf")
                        else:
                            image=raster_image(path,cal);image.save(work/"source.png")
                            record.raw_source={"width":image.width,"height":image.height,"render_scale":1}
                    else:
                        record.series,record.raw_source=Backend(name).digitize(path,cal,work,raster_dpi)
                        record.source_method="vector_reconstructed" if name=="pymupdf" else "raster_digitized"
                        if name=="plotdigitizer":raster_image(path,cal).save(work/"source.png")
                    for s in record.series:structural_check(s)
                    record.extraction_status="success";record.validation_status="structural_checks_passed"
                    record.quality.update(structural_validity="passed",geometry_calibration="explicit_calibration_applied")
                    try:
                        record.provenance["overlay"]=overlay(record.to_dict(),work)
                    except (OSError,ValueError,ImportError) as exc:
                        record.provenance["overlay"]={"status":"failed","reason":str(exc)}
                        record.quality["flags"].append("overlay_failed_manual_review_required")
                except (OSError,ValueError,RuntimeError,KeyError,TypeError,OverflowError) as exc:
                    record.extraction_status=exc.status if isinstance(exc,ExtractionError) else "backend_error"
                    record.source_method="unavailable";record.series=[]
                    record.warnings=[issue(record.extraction_status,str(exc))]
                    if (work/"geometry.json").is_file():
                        record.raw_source={"geometry_file":str((work/"geometry.json").relative_to(output_dir))}
                record.quality["elapsed_seconds"]=time.perf_counter()-started
                record.quality["attempt"]="completed" if record.extraction_status=="success" else (
                    "skipped" if record.extraction_status.startswith("skipped") else "impossible")
                diagnostics.extend({**w,"data_id":data_id,"figure_id":figure.figure_id,"asset_id":asset["asset_id"],
                                    "backend":name,"schema_version":DIGITIZED_SCHEMA_VERSION} for w in record.warnings)
                records.append(record.to_dict())
    comparisons=[]
    if backend=="compare":
        for i,a in enumerate(records):
            for b in records[i+1:]:
                if (a["figure_id"],a["panel_id"],a["provenance"]["asset_id"])!=(b["figure_id"],b["panel_id"],b["provenance"]["asset_id"]):continue
                item={"a":a["data_id"],"b":b["data_id"],"status":"not_comparable","reason":"one or both backends abstained"}
                if a["extraction_status"]==b["extraction_status"]=="success":
                    policy=a["calibration"].get("comparison")
                    item["reason"]="explicit common-series metadata and comparison tolerances required"
                    if policy:
                        item["series"]=[compare_series(s,t,policy) for s in a["series"] for t in b["series"] if s["series_id"]==t["series_id"]]
                        item["status"]="attempted"
                comparisons.append(item)
    write_jsonl(output_dir/"digitized_data.jsonl",records)
    summary={"mode":"digitize","schema_version":DIGITIZED_SCHEMA_VERSION,"backend":backend,"selection":selection,
             "figures":len(figures),"panels":len(panels),"records":len(records),"capabilities":capabilities,
             "extraction_status_counts":dict(Counter(r["extraction_status"] for r in records)),
             "comparisons":comparisons,"manual_review":"pending","data_sha256":sha256_file(output_dir/"digitized_data.jsonl")}
    write_json(output_dir/"digitization_summary.json",summary);write_jsonl(output_dir/"diagnostics.jsonl",diagnostics)
    write_manifest(output_dir,step="extract_plots",inputs=inputs,summary=summary,
                   params={"mode":"digitize","schema_version":DIGITIZED_SCHEMA_VERSION,"backend":backend,
                           "selection":selection,"raster_dpi":raster_dpi,"include_generic":include_generic,
                           "classified_dir":classified_dir,"parsed_dir":parsed_dir,"inventory_dir":upstream,
                           "recovery_dir":recovery_dir,"calibration_file":calibration_file,"imports_dir":imports_dir,
                           "limits":{"source_bytes":32*1024*1024,"pixels":16_000_000,"pdf_pages":100,"subprocess_seconds":45}})
    return summary


def run_benchmark(*, digitized_dir: Path, output_dir: Path, ground_truth_file: Path | None = None,
                  recovery_dir: Path | None = None) -> dict:
    """Evaluate only explicit independent references, leaving scientific interpretation to review."""
    from .digitization import load_config
    from .validation import compare_series, original_series

    if (ground_truth_file is None)==(recovery_dir is None):
        raise ValueError("benchmark requires exactly one of ground_truth_file and recovery_dir")
    digitized_dir,output_dir=Path(digitized_dir).resolve(),Path(output_dir)
    inputs={}
    manifest,records=_numerical_run(digitized_dir,"digitize",DIGITIZED_SCHEMA_VERSION,"digitized_data.jsonl",inputs)
    references=[]
    if ground_truth_file:
        ground_truth_file=Path(ground_truth_file)
        references=load_config(ground_truth_file,"references");inputs["ground_truth"]=ground_truth_file
        if ground_truth_file.resolve().is_relative_to(digitized_dir):
            raise ValueError("ground truth must be independent of the digitized run")
    else:
        recovery_dir=Path(recovery_dir).resolve()
        recovered,originals=_numerical_run(recovery_dir,"recover",NUMERICAL_SCHEMA_VERSION,"plot_data.jsonl",inputs)
        if recovered["inputs"].get("classified_manifest",{}).get("sha256")!=manifest["inputs"].get("classified_manifest",{}).get("sha256"):
            raise ValueError("benchmark recovery lineage differs")
        for record in records:
            for mapping in record["calibration"].get("benchmark_references",[]):
                matches=[r for r in originals if r["data_id"]==mapping.get("data_id") and r.get("figure_id")==record["figure_id"]
                         and r.get("association",{}).get("state")=="verified" and r.get("validation_status")=="structural_checks_passed"
                         and r.get("extraction_status")=="success" and r.get("source_method") in {"original_source_data","original_published_data"}]
                if len(matches)!=1:raise ValueError("benchmark original reference is not uniquely verified")
                source=matches[0]
                matches_s=[s for s in source["series"] if s.get("series_id")==mapping.get("source_series_id")]
                if len(matches_s)!=1:raise ValueError("benchmark source series mapping is ambiguous")
                extracted=next((s for s in record["series"] if s["series_id"]==mapping.get("series_id")),None)
                if extracted is None:continue
                # User explicitly attests the physical mapping; Task 03 does not invent these identities.
                if mapping.get("observable")!=extracted.get("observable") or mapping.get("units")!=extracted.get("units"):
                    raise ValueError("reference mapping must explicitly confirm observable and units")
                references.append({"figure_id":record["figure_id"],"asset_id":record["provenance"]["asset_id"],
                    "panel_id":record["panel_id"],"source_kind":source["source_method"],"independent":True,
                    "provenance":{"recovery_data_id":source["data_id"]},"comparison":mapping["comparison"],
                    "series":original_series(source,matches_s[0],mapping)})
        # The same source mapping appears in each compared backend's calibration.
        import json
        references=list({json.dumps(r,sort_keys=True):r for r in references}.values())
    if not references:raise ValueError("no explicit independent ground-truth references")
    keys=set()
    for ref in references:
        if (ref.get("source_kind") not in {"synthetic","original_source_data","original_published_data"}
                or ref.get("independent") is not True or not ref.get("provenance")):
            raise ValueError("ground truth requires independent synthetic/original provenance")
        if ref.get("data_id") in {r["data_id"] for r in records} or ref.get("source_method") in {"vector_reconstructed","raster_digitized","manual_digitized"}:
            raise ValueError("digitized data cannot be its own independent ground truth")
        key=(ref.get("figure_id"),ref.get("asset_id"),ref.get("panel_id"),ref["series"].get("series_id"))
        if key in keys:raise ValueError("duplicate ambiguous ground-truth series")
        keys.add(key)
    for root in (digitized_dir,recovery_dir):
        if root and output_dir.resolve().is_relative_to(root):raise ValueError("benchmark output must be outside input runs")
    details=[]
    for record in records:
        for ref in references:
            if (ref.get("figure_id"),ref.get("asset_id"),ref.get("panel_id"))!=(record["figure_id"],record["provenance"]["asset_id"],record["panel_id"]):continue
            detail={"data_id":record["data_id"],"backend":record["backend"],"reference_provenance":ref["provenance"],
                    "series_id":ref["series"]["series_id"],"status":"abstained","extraction_status":record["extraction_status"],
                    "calibration_residuals":record["calibration"].get("fit"),"manual_interventions":record["calibration"].get("manual_interventions",[]),
                    "elapsed_seconds":record["quality"]["elapsed_seconds"],"structural_validity":record["quality"]["structural_validity"],
                    "geometry_calibration":record["quality"]["geometry_calibration"],"manual_review":"pending"}
            found=[s for s in record["series"] if s["series_id"]==ref["series"]["series_id"]]
            if found:
                if record["calibration"].get("calibration_source")!="axis_ticks":
                    detail.update(status="not_comparable",reason="independent axis-tick calibration not documented; possible evaluation leakage")
                else:detail.update(compare_series(found[0],ref["series"],ref.get("comparison",{})))
            details.append(detail)
    if not details:raise ValueError("ground truth does not match any digitized asset")
    compared=[d for d in details if d["status"]=="compared"]
    by_backend={name:{"cases":sum(d["backend"]==name for d in details),
                      "compared":sum(d["backend"]==name for d in compared),
                      "elapsed_seconds":sum(r["quality"]["elapsed_seconds"] for r in records if r["backend"]==name)}
                for name in sorted({d["backend"] for d in details})}
    result={"mode":"benchmark","schema_version":DIGITIZED_SCHEMA_VERSION,"records":len(details),"cases":len(details),
            "compared":len(compared),"coverage":len(compared)/len(details),"abstention_rate":1-len(compared)/len(details),
            "mean_correct_fraction":sum(d["correct_fraction"] for d in compared)/len(compared) if compared else None,
            "by_backend":by_backend,"scope":"configured independent reference cases only; not a real-HEP accuracy claim",
            "manual_review":"pending","scientific_agreement":"numerical_metrics_only"}
    output_dir=create_output_dir(output_dir)
    write_json(output_dir/"benchmark_results.json",result);write_jsonl(output_dir/"benchmark_details.jsonl",details)
    write_manifest(output_dir,step="extract_plots",params={"mode":"benchmark","schema_version":DIGITIZED_SCHEMA_VERSION,
                   "digitized_dir":digitized_dir,"ground_truth_file":ground_truth_file,"recovery_dir":recovery_dir},inputs=inputs,summary=result)
    return result


# Classification reads the saved inventory only. Validation happens before creating
# output so malformed schemas and broken links cannot leave a plausible partial run.
def _inventory_records(figures: list[dict[str, Any]], panels: list[dict[str, Any]], *,
                       schema_version: str = SCHEMA_VERSION) -> tuple[
        list[FigureRecord], list[PanelRecord]]:
    for kind, records in (("figure", figures), ("panel", panels)):
        ids = set()
        for record in records:
            identifier = record.get(f"{kind}_id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"invalid or duplicate {kind}_id")
            ids.add(identifier)
            if record.get("schema_version") != schema_version:
                raise ValueError(f"{identifier}: expected schema {schema_version}")
            if record.get("figure_type" if kind == "figure" else "panel_type", "unknown") not in (
                    {"generic", "plot", "mixed", "unknown"} if kind == "figure" else {"generic", "plot", "unknown"}):
                raise ValueError(f"{identifier}: invalid classification")
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


# Task 03 verifies the saved chain, including the inventory's original input hashes.
# Historical absolute paths are not used to resolve files in a relocated parse run.
def _recovery_inputs(classified_dir: Path, parsed_dir: Path, inventory_dir: Path | None) -> tuple:
    inputs = {"classified_" + name: local_path(classified_dir, classified_dir / filename) for name, filename in (
        ("manifest", "manifest.json"), ("figures", "figures.jsonl"), ("panels", "panels.jsonl"),
        ("summary", "figure_summary.json"), ("diagnostics", "diagnostics.jsonl"))}
    classified = read_manifest(classified_dir, "extract_plots")
    if classified.get("params", {}).get("mode") != "classify" or classified["params"].get("schema_version") != CLASSIFIED_SCHEMA_VERSION:
        raise ValueError("recover requires a classified run with schema 1.1")
    summary = read_json(inputs["classified_summary"])
    if summary != classified.get("summary") or summary.get("schema_version") != CLASSIFIED_SCHEMA_VERSION:
        raise ValueError("classified summary does not match its manifest/schema")
    raw_figures = read_jsonl(inputs["classified_figures"])
    raw_panels = read_jsonl(inputs["classified_panels"])
    figures, panels = _inventory_records(raw_figures, raw_panels, schema_version=CLASSIFIED_SCHEMA_VERSION)
    if summary.get("figures") != len(figures) or summary.get("panels") != len(panels):
        raise ValueError("classified record counts do not match summary")
    if any(d.get("schema_version") != CLASSIFIED_SCHEMA_VERSION for d in read_jsonl(inputs["classified_diagnostics"])):
        raise ValueError("classified diagnostics schema mismatch")
    try:
        inventory_dir = (Path(inventory_dir) if inventory_dir is not None else
                         Path(classified["inputs"]["inventory_manifest"]["path"]).parent).resolve()
        for role, filename in (("inventory_manifest", "manifest.json"), ("figures", "figures.jsonl"),
                               ("panels", "panels.jsonl"), ("inventory_summary", "figure_summary.json"),
                               ("inventory_diagnostics", "diagnostics.jsonl")):
            path = local_path(inventory_dir, inventory_dir / filename)
            if sha256_file(path) != classified["inputs"][role]["sha256"]:
                raise ValueError(f"inventory provenance hash mismatch: {role}")
            inputs["upstream_" + role] = path
    except (OSError, KeyError) as exc:
        raise ValueError("inventory provenance unavailable; retain the original inventory or pass its relocated --inventory-dir") from exc
    inventory_manifest = read_manifest(inventory_dir, "extract_plots")
    if inventory_manifest.get("params", {}).get("mode") != "inventory" or inventory_manifest["params"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError("upstream inventory must use schema 1.0")
    originals = [*read_jsonl(inputs["upstream_figures"]), *read_jsonl(inputs["upstream_panels"])]
    current = [*raw_figures, *raw_panels]
    changed_fields = {"figure_type", "panel_type", "plot_type", "classification_method", "classification_confidence",
                      "classification_evidence", "classification_review_status", "schema_version"}
    if len(originals) != len(current) or any(
            any(after.get(key) != value for key, value in before.items() if key not in changed_fields)
            for before, after in zip(originals, current)):
        raise ValueError("classified IDs/source metadata do not match the original inventory")
    for role, filename in (("parse_manifest", "manifest.json"), ("papers", "papers.jsonl")):
        path = local_path(parsed_dir, parsed_dir / filename)
        if sha256_file(path) != inventory_manifest["inputs"][role]["sha256"]:
            raise ValueError(f"parse source mismatch: {role}")
        inputs[role] = path
    read_manifest(parsed_dir, "parse")
    papers = read_jsonl(inputs["papers"])
    ids = [p.get("paper_id") for p in papers]
    if (any(not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) for identifier in ids)
            or len(set(ids)) != len(ids) or any(f.paper_id not in ids for f in figures)):
        raise ValueError("paper identities do not match the parse run")
    return figures, panels, papers, inputs, inventory_manifest, inventory_dir


def _verify_paper_sources(parsed_dir: Path, paper_id: str, original: dict[str, Any],
                          inputs: dict[str, Path]) -> tuple[Path, Path]:
    paper_dir = local_path(parsed_dir, parsed_dir / "papers" / paper_id)
    source_root = local_path(paper_dir, paper_dir / "source")
    flat = local_path(paper_dir, paper_dir / "flat.tex")
    if not source_root.is_dir() or f"{paper_id}:flat" not in original:
        raise ValueError("missing source root or original flat.tex hash")
    for role, entry in original.items():
        if not role.startswith(paper_id + ":"):
            continue
        suffix = role[len(paper_id) + 1:]
        if suffix == "flat":
            path = flat
        elif suffix == "sections":
            path = local_path(paper_dir, paper_dir / "sections.json")
        elif suffix.startswith("source:"):
            path = local_path(source_root, source_root / suffix.removeprefix("source:"))
        else:
            raise ValueError(f"unsupported original input role: {role}")
        inputs[role] = path
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"source hash mismatch: {role}")
    return flat, source_root


def run_recovery(*, classified_dir: Path, parsed_dir: Path, output_dir: Path, source_method: str = "native",
                 inventory_dir: Path | None = None, hepdata_dir: Path | None = None, hepdata_map: Path | None = None,
                 enable_hepdata_network: bool = False, include_generic: bool = False) -> dict[str, Any]:
    """Recover explicit numerical sources into a new schema 1.2 run; offline by default."""
    from .source_data import (compare_sources, data_record, load_hepdata_mapping, recover_hepdata, recover_native)

    if source_method not in {"native", "hepdata", "auto", "compare"}:
        raise ValueError("source_method must be native, hepdata, auto or compare")
    if source_method == "native" and (hepdata_dir or hepdata_map or enable_hepdata_network):
        raise ValueError("HEPData options require source_method hepdata, auto or compare")
    classified_dir, parsed_dir, output_dir = Path(classified_dir).resolve(), Path(parsed_dir).resolve(), Path(output_dir)
    figures, panels, papers, inputs, original, upstream_dir = _recovery_inputs(classified_dir, parsed_dir, inventory_dir)
    mappings = []
    if hepdata_map is None and hepdata_dir is not None:
        hepdata_map = Path(hepdata_dir) / "mapping.json"
    hepdata_root = (Path(hepdata_dir) if hepdata_dir is not None else
                    Path(hepdata_map).parent if hepdata_map is not None else classified_dir).resolve()
    for source in (classified_dir, parsed_dir, upstream_dir):
        if output_dir.resolve().is_relative_to(source):
            raise ValueError("output directory must not be inside an input run/source directory")
    if hepdata_map is not None:
        hepdata_map = Path(hepdata_map).resolve()
        mappings = load_hepdata_mapping(hepdata_map)
        inputs["hepdata_map"] = hepdata_map
    output_dir = create_output_dir(output_dir)
    records = []
    diagnostics = []
    verified_figures = []
    paper_results = []
    for paper in papers:
        paper_id = paper["paper_id"]
        paper_figures = [f for f in figures if f.paper_id == paper_id]
        paper_panels = [p for p in panels if p.paper_id == paper_id]
        try:
            flat, root = _verify_paper_sources(parsed_dir, paper_id, original["inputs"], inputs)
            verified_figures.extend(paper_figures)
            if source_method in {"native", "auto", "compare"}:
                found, source_inputs = recover_native(paper_id=paper_id, flat_path=flat, source_root=root,
                                                      figures=paper_figures, panels=paper_panels, include_generic=include_generic)
                records.extend(found)
                inputs.update({f"{paper_id}:recovery:{role}": path for role, path in source_inputs.items()})
                for figure in paper_figures:
                    if (include_generic or figure.figure_type != "generic") and not any(r.figure_id == figure.figure_id for r in found):
                        empty = data_record(paper_id, figure, None, "native", "no_explicit_source",
                                            {"state": "unmatched", "kind": "no_explicit_plot_command"})
                        empty.warnings = [issue("native_data_unavailable", "No supported explicit numerical source in this figure.")]
                        records.append(empty)
            paper_results.append({"paper_id": paper_id, "status": "sources_verified"})
        except (OSError, ValueError) as exc:
            verified_figures = [f for f in verified_figures if f.paper_id != paper_id]
            diagnostic = {**issue("source_mismatch", str(exc)), "paper_id": paper_id, "severity": "error"}
            diagnostics.append(diagnostic)
            paper_results.append({"paper_id": paper_id, "status": "failed"})
            for figure in paper_figures:
                failed = data_record(paper_id, figure, None, "provenance", "source_mismatch",
                                     {"state": "unmatched", "kind": "source_verification_failed"})
                failed.warnings = [diagnostic]
                records.append(failed)
    if source_method in {"hepdata", "auto", "compare"}:
        found, source_inputs = recover_hepdata(figures=verified_figures, panels=panels, papers=papers,
                                               mappings=mappings, root=hepdata_root,
                                               enable_network=enable_hepdata_network, include_generic=include_generic)
        records.extend(found)
        inputs.update({"hepdata:" + role: path for role, path in source_inputs.items()})
    for record in records:
        diagnostics.extend({**warning, "paper_id": record.paper_id, "figure_id": record.figure_id,
                            "panel_id": record.panel_id, "data_id": record.data_id} for warning in record.warnings)
    summary = {
        "schema_version": NUMERICAL_SCHEMA_VERSION, "mode": "recover", "source_method": source_method,
        "figures": len(figures), "panels": len(panels),
        "records": len(records), "source_method_counts": dict(Counter(r.source_method for r in records)),
        "association_counts": dict(Counter(r.association["state"] for r in records)),
        "extraction_status_counts": dict(Counter(r.extraction_status for r in records)),
        "recovered_series": sum(len(r.series) for r in records if r.source_method in {"original_source_data", "original_published_data"}),
        "paper_results": paper_results, "diagnostic_counts": dict(Counter(d["code"] for d in diagnostics)),
        "comparisons": compare_sources(records) if source_method == "compare" else [],
        "manual_review": "pending",
    }
    write_jsonl(output_dir / "plot_data.jsonl", [r.to_dict() for r in records])
    write_jsonl(output_dir / "diagnostics.jsonl", [{**d, "schema_version": NUMERICAL_SCHEMA_VERSION} for d in diagnostics])
    write_json(output_dir / "recovery_summary.json", summary)
    # Missing files have structured diagnostics but cannot truthfully have a hash.
    readable_inputs = {role: path for role, path in inputs.items() if path.is_file()}
    write_manifest(output_dir, step="extract_plots", inputs=readable_inputs, summary=summary,
                   params={"mode": "recover", "schema_version": NUMERICAL_SCHEMA_VERSION,
                           "classified_dir": classified_dir, "parsed_dir": parsed_dir, "inventory_dir": upstream_dir,
                           "source_method": source_method, "hepdata_dir": hepdata_root if hepdata_dir or hepdata_map else None,
                           "hepdata_map": hepdata_map, "enable_hepdata_network": enable_hepdata_network,
                           "include_generic": include_generic, "max_source_bytes": 8 * 1024 * 1024,
                           "network_attempts": 3, "network_connect_timeout": 10, "network_read_timeout": 20})
    return summary
