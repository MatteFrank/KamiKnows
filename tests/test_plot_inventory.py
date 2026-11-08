"""Offline inventory fixtures exercise provenance, containment and partial failures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kamiknows import parsing
from kamiknows.common import read_json, read_jsonl, read_manifest, write_json, write_jsonl, write_manifest
from kamiknows.kamiplotextractor import (FigureClassifier, FigureExtractor, FigureRecord, PanelRecord,
                                      run_classification, run_inventory, run_recovery)
from kamiknows.kamiplotextractor.latex import scan

PAPER_ID = "1111_00001v1"


@pytest.fixture
def inventory_run(tmp_path):
    """Build the on-disk parse contract using the existing parser, without downloads."""
    def build(body, assets=None, sources=None, flat=None):
        run = tmp_path / "parsed"
        paper = run / "papers" / PAPER_ID
        root = paper / "source" / "extracted"
        root.mkdir(parents=True)
        main = "\\documentclass{article}\n\\begin{document}\n\\section{Results}\n" + body + "\n\\end{document}\n"
        (root / "main.tex").write_text(main, encoding="utf-8")
        for name, content in (sources or {}).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        for name in assets or []:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic asset bytes\x00\xff")
        expanded = flat if flat is not None else parsing.flatten_latex_file(root / "main.tex", root)
        (paper / "flat.tex").write_text(expanded, encoding="utf-8")
        sections, _, _ = parsing.parse_latex(expanded)
        write_json(paper / "sections.json", sections)
        write_jsonl(run / "papers.jsonl", [{"paper_id": PAPER_ID, "arxiv_id": "1111.00001v1", "status": "success"}])
        write_manifest(run, step="parse", params={}, inputs={}, summary={})
        return run, root
    return build


def inventory(run, tmp_path, name="inventory"):
    output = tmp_path / name
    summary = run_inventory(parsed_dir=run, output_dir=output)
    return summary, read_jsonl(output / "figures.jsonl"), read_jsonl(output / "panels.jsonl")


@pytest.mark.parametrize("reference,asset,preamble", [
    ("plot.png", "plot.png", ""),
    ("plot", "plot.pdf", ""),
    ("plot", "figs/plot.pdf", r"\graphicspath{{figs/}}"),
    ("deep/path/plot.jpeg", "deep/path/plot.jpeg", ""),
    ("plot.jpg", "plot.jpg", ""),
    ("plot.eps", "plot.eps", ""),
    ("plot.svg", "plot.svg", ""),
    ("plot", "plot.PDF", ""),
])
def test_formats_paths_metadata_and_hashes(inventory_run, tmp_path, reference, asset, preamble):
    body = preamble + "\\begin{figure}\\includegraphics{" + reference + r"}\caption{A \textbf{nested {caption}}.}\label{fig:one}\end{figure}"
    run, root = inventory_run(body, [asset])
    before = (root / asset).read_bytes()
    summary, figures, panels = inventory(run, tmp_path)
    figure = figures[0]
    assert summary["resolved_references"] == 1 and panels == []
    assert figure["caption"] == "A nested caption ."
    assert figure["caption_raw"] == r"A \textbf{nested {caption}}."
    assert figure["latex_label"] == "fig:one" and figure["section_id"] == "sec_001"
    assert figure["figure_type"] == "unknown" and figure["classification_confidence"] is None
    graphic = figure["source_assets"][0]
    assert graphic["path"] == f"extracted/{asset}"
    assert graphic["format"] == Path(asset).suffix[1:].lower()
    assert graphic["sha256"] == hashlib.sha256(before).hexdigest()
    assert graphic["source_origins"][0]["path"] == "extracted/main.tex"
    assert (root / asset).read_bytes() == before
    manifest = read_manifest(tmp_path / "inventory", "extract_plots")
    assert manifest["params"]["schema_version"] == figure["schema_version"] == "1.0"
    assert manifest["summary"] == summary == read_json(tmp_path / "inventory" / "figure_summary.json")
    assert manifest["inputs"][f"{PAPER_ID}:source:extracted/{asset}"]["sha256"] == graphic["sha256"]
    assert {"parse_manifest", "papers", f"{PAPER_ID}:flat", f"{PAPER_ID}:sections",
            f"{PAPER_ID}:source:extracted/main.tex"} <= manifest["inputs"].keys()


def test_star_multiple_repeated_assets_layout_and_deterministic_ids(inventory_run, tmp_path):
    run, root = inventory_run(r"""
\begin{figure*}
\includegraphics*[width={.5\textwidth},trim={1 2 3 4},clip,angle=90,page=2]{a.pdf}
\includegraphics{b.png}\includegraphics{b.png}
\caption[Short]{Full caption.}\label{wide}
\end{figure*}
""", ["a.pdf", "b.png"])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    summary, figures, _ = inventory(run, tmp_path)
    _, repeat, _ = inventory(run, tmp_path, "repeat")
    assert figures == repeat
    assert summary["asset_references"] == 3 and summary["unique_assets"] == 2
    figure = figures[0]
    assert figure["provenance"]["environment"] == "figure*"
    assert figure["captions"][0]["short_raw"] == "Short"
    first, second, third = figure["source_assets"]
    assert first["starred"] is True
    assert first["layout"]["trim"] == "{1 2 3 4}"
    assert first["layout"]["page"] == "2" and first["layout"]["angle"] == "90"
    assert first["layout"]["clip"] is True
    assert second["path"] == third["path"] and second["asset_id"] != third["asset_id"]
    assert second["source_origins"] != third["source_origins"]
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("panels", [
    r"\begin{subfigure}[b]{.45\textwidth}\includegraphics{a.png}\caption{Left}\label{left}\end{subfigure}"
    r"\begin{subfigure}{.45\textwidth}\includegraphics{b.png}\caption{Right}\label{right}\end{subfigure}",
    r"\subfigure[Left]{\includegraphics{a.png}\label{left}}"
    r"\subfloat[Short][Right]{\includegraphics{b.png}\label{right}}",
    r"\subcaptionbox{Left}[.45\textwidth]{\includegraphics{a.png}\label{left}}"
    r"\begin{subcaptionblock}{.45\textwidth}\includegraphics{b.png}\subcaption{Right}\label{right}\end{subcaptionblock}",
])
def test_explicit_panels(inventory_run, tmp_path, panels):
    run, _ = inventory_run(r"\begin{figure}" + panels + r"\caption{Combined}\label{all}\end{figure}", ["a.png", "b.png"])
    summary, figures, records = inventory(run, tmp_path)
    assert summary["panels"] == 2
    assert figures[0]["caption"] == "Combined" and figures[0]["latex_labels"] == ["all"]
    assert [p["caption"] for p in records] == ["Left", "Right"]
    assert [p["panel_label"] for p in records] == ["left", "right"]
    assert figures[0]["panels"] == [p["panel_id"] for p in records]
    for panel, asset in zip(records, figures[0]["source_assets"]):
        assert panel["source_asset_ids"] == [asset["asset_id"]]
        assert asset["panel_id"] == panel["panel_id"]
        assert panel["source_region"]["image_region"] is None


@pytest.mark.parametrize("reference,code", [
    ("missing.png", "asset_missing"),
    (r"\plotdir/plot.png", "unresolved_macro"),
    ("../outside.png", "unsafe_path"),
    ("/tmp/outside.png", "unsafe_path"),
    ("https://example.org/plot.png", "unsafe_path"),
    ("plot.root", "unsupported_format"),
])
def test_unresolved_paths_are_retained(inventory_run, tmp_path, reference, code):
    run, _ = inventory_run("\\begin{figure}\\includegraphics{" + reference + "}\\end{figure}")
    summary, figures, _ = inventory(run, tmp_path)
    asset = figures[0]["source_assets"][0]
    assert asset["reference"] == reference and asset["path"] is None and asset["sha256"] is None
    assert code in {w["code"] for w in asset["warnings"]}
    assert summary["unresolved_references"] == 1
    assert figures[0]["caption"] is None and figures[0]["status"] == "partial"


def test_included_source_directory(inventory_run, tmp_path):
    run, _ = inventory_run(r"\input{sections/results}", ["sections/plots/a.pdf"], {
        "sections/results.tex": r"\graphicspath{{plots/}}\begin{figure}\includegraphics{a}\caption{Included}\end{figure}",
    })
    _, figures, _ = inventory(run, tmp_path)
    asset = figures[0]["source_assets"][0]
    assert asset["path"] == "extracted/sections/plots/a.pdf"
    assert asset["source_origins"][0]["path"] == "extracted/sections/results.tex"


def test_figure_spans_included_fragment(inventory_run, tmp_path):
    run, _ = inventory_run(r"\begin{figure}\input{parts/graphics}\caption{Combined source}\end{figure}",
                           ["parts/a.svg"], {"parts/graphics.tex": r"\includegraphics{a.svg}"})
    _, figures, _ = inventory(run, tmp_path)
    assert figures[0]["source_assets"][0]["path"] == "extracted/parts/a.svg"


def test_ambiguous_asset_is_never_guessed(inventory_run, tmp_path):
    run, _ = inventory_run(r"\begin{figure}\includegraphics{a}\end{figure}", ["a.pdf", "a.png"])
    summary, figures, _ = inventory(run, tmp_path)
    asset = figures[0]["source_assets"][0]
    assert asset["path"] is None and asset["candidates"] == ["extracted/a.pdf", "extracted/a.png"]
    assert summary["diagnostic_counts"]["asset_ambiguous"] == 1


def test_ambiguous_source_origins(inventory_run, tmp_path):
    figure = r"\begin{figure}\includegraphics{a.png}\end{figure}"
    run, _ = inventory_run(r"\input{x/figure}\input{y/figure}", ["x/a.png", "y/a.png"],
                           {"x/figure.tex": figure, "y/figure.tex": figure})
    _, figures, _ = inventory(run, tmp_path)
    assert len(figures) == 2
    for figure in figures:
        asset = figure["source_assets"][0]
        assert asset["path"] is None and len(asset["source_origins"]) == 2
        assert "source_origin_ambiguous" in {w["code"] for w in asset["warnings"]}


def test_symlinks_cannot_escape_source_root(inventory_run, tmp_path):
    run, root = inventory_run(r"\begin{figure}\includegraphics{leak.pdf}\includegraphics{alias/secret.pdf}\end{figure}")
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"do not read")
    (root / "leak.pdf").symlink_to(outside)
    (root / "alias").symlink_to(tmp_path, target_is_directory=True)
    (root / "leak.tex").symlink_to(tmp_path / "private.tex")
    (tmp_path / "private.tex").write_text(r"\includegraphics{secret.pdf}")
    summary, figures, _ = inventory(run, tmp_path)
    assert summary["resolved_references"] == 0
    assert all(a["path"] is None for a in figures[0]["source_assets"])
    manifest = read_manifest(tmp_path / "inventory", "extract_plots")
    assert all(entry["path"] != str(outside) for entry in manifest["inputs"].values())
    assert outside.read_bytes() == b"do not read"


def test_no_figures_comments_verbatim_and_macro_definitions(inventory_run, tmp_path):
    run, _ = inventory_run(r"""
% \begin{figure}\includegraphics{comment.png}\end{figure}
\begin{verbatim}\begin{figure}\includegraphics{literal.png}\end{figure}\end{verbatim}
\verb|\begin{figure}|
\newcommand{\myfigure}[1]{\begin{figure}\includegraphics{#1}\end{figure}}
Text only, including escaped percent \%.
""")
    summary, figures, panels = inventory(run, tmp_path)
    assert summary["figures"] == 0 and figures == panels == []
    assert summary["diagnostic_counts"]["macro_definition_skipped"] == 1


def test_graphicspath_scope_and_nested_environments(inventory_run, tmp_path):
    run, _ = inventory_run(r"""
{\graphicspath{{left/}}\begin{figure}\begin{center}\includegraphics{a.png}\end{center}\end{figure}}
\graphicspath{{right/}}\begin{figure}\includegraphics{a.png}\end{figure}
""", ["left/a.png", "right/a.png"])
    _, figures, _ = inventory(run, tmp_path)
    assert [f["source_assets"][0]["path"] for f in figures] == ["extracted/left/a.png", "extracted/right/a.png"]


def test_malformed_figure_and_graphics_outside_are_reported(inventory_run, tmp_path):
    run, _ = inventory_run(r"\includegraphics{a.png}\begin{figure}\includegraphics{a.png}", ["a.png"])
    summary, figures, _ = inventory(run, tmp_path)
    assert len(figures) == 1
    assert "unclosed_environment" in summary["diagnostic_counts"]
    assert summary["diagnostic_counts"]["graphics_outside_figure"] == 1


def test_missing_input_remains_visible(inventory_run, tmp_path):
    run, _ = inventory_run(r"\input{missing}")
    summary, _, _ = inventory(run, tmp_path)
    assert summary["diagnostic_counts"]["unexpanded_input"] == 1


def test_section_mismatch_is_null(inventory_run, tmp_path):
    run, _ = inventory_run(r"\begin{figure}\includegraphics{a.png}\end{figure}", ["a.png"])
    write_json(run / "papers" / PAPER_ID / "sections.json", [{"section_id": "sec_999", "heading": "Other", "level": 1}])
    _, figures, _ = inventory(run, tmp_path)
    assert figures[0]["section_id"] is None
    assert "section_unresolved" in {w["code"] for w in figures[0]["warnings"]}


def test_failed_paper_does_not_drop_successful_paper(inventory_run, tmp_path):
    run, _ = inventory_run(r"\begin{figure}\includegraphics{a.png}\end{figure}", ["a.png"])
    papers = read_jsonl(run / "papers.jsonl")
    (run / "papers.jsonl").unlink()
    write_jsonl(run / "papers.jsonl", [*papers, {"paper_id": "2222_00002v1", "status": "failed"}])
    summary, figures, _ = inventory(run, tmp_path)
    assert len(figures) == 1 and summary["paper_status_counts"]["failed"] == 1
    errors = [d for d in read_jsonl(tmp_path / "inventory" / "diagnostics.jsonl") if d["severity"] == "error"]
    assert len(errors) == 1 and errors[0]["paper_id"] == "2222_00002v1"


def test_run_validation_and_no_overwrite(inventory_run, tmp_path):
    run, _ = inventory_run("No figures.")
    output = tmp_path / "inventory"
    run_inventory(parsed_dir=run, output_dir=output)
    with pytest.raises(FileExistsError):
        run_inventory(parsed_dir=run, output_dir=output)
    with pytest.raises(ValueError, match="inside"):
        run_inventory(parsed_dir=run, output_dir=run / "inventory")
    with pytest.raises(ValueError, match="only mode"):
        run_inventory(parsed_dir=run, output_dir=tmp_path / "bad", mode="compare")
    write_json(run / "manifest.json", {"step": "chunk"})
    with pytest.raises(ValueError, match="expected 'parse'"):
        run_inventory(parsed_dir=run, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("paper_ids", [["../escape"], [PAPER_ID, PAPER_ID]])
def test_unsafe_or_duplicate_paper_ids(inventory_run, tmp_path, paper_ids):
    run, _ = inventory_run("Text.")
    (run / "papers.jsonl").unlink()
    write_jsonl(run / "papers.jsonl", [{"paper_id": identifier} for identifier in paper_ids])
    with pytest.raises(ValueError, match="paper_id"):
        run_inventory(parsed_dir=run, output_dir=tmp_path / "inventory")


def test_importable_extractor_and_cli(inventory_run, tmp_path):
    run, root = inventory_run(r"\begin{figure}\includegraphics{a.png}\caption{A}\end{figure}", ["a.png"])
    figures, panels, _, _ = FigureExtractor().extract(
        paper_id=PAPER_ID, flat_tex=(run / "papers" / PAPER_ID / "flat.tex").read_text(),
        source_root=root.parent, sections=read_json(run / "papers" / PAPER_ID / "sections.json"))
    assert len(figures) == 1 and not panels
    result = subprocess.run([sys.executable, "scripts/extract_plots.py", "--parsed-dir", str(run),
                             "--output-dir", str(tmp_path / "cli"), "--mode", "inventory"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read_manifest(tmp_path / "cli", "extract_plots")["summary"]["figures"] == 1


def test_balanced_optional_braces_comments_and_escaped_delimiters():
    document = scan(r"""\begin{figure}
\includegraphics[width={a,b},trim={1 [2] 3 4}]{x.pdf}
\caption[Short {nested [text]}]{Literal \{brace\} and 5\% \textit{nested {text}}.}
% \end{figure}
\end{figure}""")
    assert len(document.environments) == 1 and document.environments[0].closed
    assert next(c for c in document.commands if c.name == "caption").options == ["Short {nested [text]}"]
    assert document.warnings == []


def test_caption_raw_comments_and_panel_label_in_optional_caption(inventory_run, tmp_path):
    run, _ = inventory_run(r"""\begin{figure}
\subfloat[Panel \label{panel:a}]{\includegraphics{a.png}}
\caption{First % preserve this raw comment
second.}\label{fig:all}
\end{figure}""", ["a.png"])
    _, figures, panels = inventory(run, tmp_path)
    assert figures[0]["caption_raw"] == "First % preserve this raw comment\nsecond."
    assert figures[0]["caption"] == "First second."
    assert panels[0]["panel_label"] == "panel:a" and panels[0]["caption"] == "Panel"


def test_new_inventory_is_recognized_by_outputs_index(inventory_run, tmp_path):
    run, _ = inventory_run("No figures.")
    inventory(run, tmp_path)
    index = tmp_path / "index.md"
    result = subprocess.run([sys.executable, "scripts/index_outputs.py", "--outputs-dir", str(tmp_path),
                             "--notes", str(tmp_path / "no-notes.json"), "--output", str(index)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "## Figure inventories" in index.read_text() and "`inventory`" in index.read_text()


def test_source_root_symlink_and_missing_flat_are_paper_failures(inventory_run, tmp_path):
    run, root = inventory_run("No figures.")
    (run / "papers" / PAPER_ID / "flat.tex").unlink()
    summary, figures, _ = inventory(run, tmp_path)
    assert summary["paper_status_counts"] == {"failed": 1} and not figures
    source = root.parent
    source.rename(source.with_name("original_source"))
    source.symlink_to(tmp_path, target_is_directory=True)
    summary, _, _ = inventory(run, tmp_path, "symlink_run")
    assert summary["paper_status_counts"] == {"failed": 1}
    diagnostic = read_jsonl(tmp_path / "symlink_run" / "diagnostics.jsonl")[0]
    assert "outside source root" in diagnostic["message"]


def test_unexpanded_macros_and_unsupported_panel_boundaries(inventory_run, tmp_path):
    run, _ = inventory_run(r"""\newcommand{\plot}{\includegraphics{a.png}}
\begin{figure}\plot\subcaption{No explicit panel}\end{figure}
\iffalse\begin{figure}\includegraphics{a.png}\end{figure}\fi""", ["a.png"])
    summary, figures, panels = inventory(run, tmp_path)
    assert len(figures) == 2 and panels == []
    assert summary["diagnostic_counts"]["no_graphics_references"] == 1
    assert summary["diagnostic_counts"]["panel_boundary_unresolved"] == 1
    assert "conditional_not_evaluated" in summary["diagnostic_counts"]


def test_malformed_includegraphics_retained_and_nesting_bounded(inventory_run, tmp_path):
    run, _ = inventory_run(r"\begin{figure}\includegraphics\caption{Bad}\end{figure}")
    summary, figures, _ = inventory(run, tmp_path)
    assert summary["unresolved_references"] == 1
    assert figures[0]["source_assets"][0]["reference"] == ""
    assert summary["diagnostic_counts"]["malformed_command"] == 1
    document = scan("{" * 150 + r"\includegraphics{hidden.png}" + "}" * 150)
    assert "nesting_limit" in {w["code"] for w in document.warnings}


def test_multiple_captions_and_graphicspath_macro_are_visible(inventory_run, tmp_path):
    run, _ = inventory_run(r"\graphicspath{{\unknown/}{ok/}}\begin{figure}\includegraphics{a.png}"
                           r"\caption{First}\caption*{Second}\label{one}\label{two}\end{figure}", ["ok/a.png"])
    summary, figures, _ = inventory(run, tmp_path)
    assert summary["resolved_references"] == 1
    assert figures[0]["caption"] == "First" and len(figures[0]["captions"]) == 2
    assert figures[0]["latex_labels"] == ["one", "two"]
    assert {"multiple_captions", "unresolved_macro"} <= summary["diagnostic_counts"].keys()


# Task 02 uses synthetic metadata, not image recognition or scientific ground truth.
@pytest.fixture
def classification_figure():
    return FigureRecord(
        paper_id=PAPER_ID, figure_id=f"{PAPER_ID}::figure::0001", caption=None, caption_raw=None,
        latex_label=None, section_id="sec_001", source_assets=[{
            "asset_id": "asset_01", "reference": "histogram.pdf", "path": None,
            "panel_id": None, "sha256": None, "status": "unresolved"}],
        source_hashes={}, panels=[], provenance={"manual_review": "pending"})


@pytest.mark.parametrize("caption,category,plot_type", [
    ("Histogram of the signal and background.", "plot", "histogram"),
    ("Scatter plot of reconstructed and true energy.", "plot", "scatter"),
    ("Energy distribution with statistical error bars.", "plot", "unknown"),
    ("ROC curve for the classifier.", "plot", "curve"),
    ("Detector schematic with a colour scale.", "generic", None),
    ("Experimental setup diagram.", "generic", None),
    ("Photograph of the ATLAS detector.", "generic", None),
    ("Event display with coloured tracks and an energy scale.", "generic", None),
    ("Architecture diagram of the generative model.", "generic", None),
    ("Processing workflow.", "generic", None),
    ("Transverse momentum distribution.", "plot", "unknown"),
    ("Invariant mass spectrum.", "plot", "unknown"),
    ("Efficiency versus transverse momentum.", "plot", "curve"),
    ("Resolution as a function of energy.", "plot", "curve"),
    ("Data/MC ratio.", "plot", "ratio"),
    ("Uncertainty band.", "plot", "other"),
    ("Heatmap of energy deposits.", "plot", "heatmap"),
    (None, "unknown", None),
    ("Data distribution from the model simulation in the calorimeter figure.", "unknown", None),
    ("Histogram and detector schematic.", "unknown", None),
    ("An event display with a heatmap.", "unknown", None),
    ("Not a histogram.", "unknown", None),
    ("Histogram without an uncertainty band.", "plot", "histogram"),
    ("Particles scatter in the calorimeter.", "unknown", None),
    ("Histogram. This is not a histogram.", "unknown", None),
])
def test_text_classification_rules(classification_figure, caption, category, plot_type):
    classification_figure.caption = caption
    before = classification_figure.to_dict()
    figure, panels = FigureClassifier().classify(classification_figure, [])
    assert figure.figure_type == category and panels == []
    assert figure.classification_evidence["plot_type"] == plot_type
    assert figure.classification_confidence is None
    assert figure.classification_review_status == "pending"
    assert figure.schema_version == "1.1" and figure.classification_method == "text_rules_v1"
    assert classification_figure.to_dict() == before
    assert figure.provenance == before["provenance"] and figure.source_assets == before["source_assets"]
    if category != "unknown":
        assert any(i["decisive"] and i["polarity"] == "positive" for i in figure.classification_evidence["indicators"])


@pytest.mark.parametrize("word", ["data", "distribution", "model", "simulation", "figure", "calorimeter"])
def test_weak_words_and_filenames_never_decide(classification_figure, word):
    classification_figure.caption = word
    classification_figure.latex_label = "fig:roc_curve"
    figure, _ = FigureClassifier().classify(classification_figure, [])
    assert figure.figure_type == "unknown"
    assert figure.classification_evidence["indicators"]
    assert all(not i["decisive"] for i in figure.classification_evidence["indicators"])


def test_raw_caption_fallback_and_filename_conflict(classification_figure):
    classification_figure.caption_raw = r"\textbf{ROC curve} for the model. % detector schematic"
    classification_figure.source_assets[0]["reference"] = "detector_schematic.png"
    figure, _ = FigureClassifier().classify(classification_figure, [])
    assert figure.figure_type == "plot"
    assert figure.classification_evidence["contradictory_indicators"] == []
    assert any(i["source"] == "caption_raw" and i["rule"] == "roc" for i in figure.classification_evidence["indicators"])


def test_multiple_saved_captions_are_all_considered(classification_figure):
    classification_figure.caption = "Histogram."
    classification_figure.captions = [{"normalized": "Histogram.", "raw": "Histogram."},
                                      {"normalized": "Photograph of detector.", "raw": "Photograph of detector."}]
    result, _ = FigureClassifier().classify(classification_figure, [])
    assert result.figure_type == "unknown"
    assert result.classification_evidence["contradictory_indicators"]


def test_conflict_summary_diagnostics_and_global_caption_priority(inventory_run, tmp_path):
    run, _ = inventory_run(r"""\begin{figure}\caption{Histogram and detector schematic}\end{figure}
\begin{figure}\subfloat[Photograph]{\includegraphics{a.png}}
\caption{Histogram}\end{figure}""", ["a.png"])
    inventory(run, tmp_path)
    summary = run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
    figures = read_jsonl(tmp_path / "classified" / "figures.jsonl")
    assert [f["figure_type"] for f in figures] == ["unknown", "generic"]
    assert summary["contradictory_figures"] == [f["figure_id"] for f in figures]
    assert summary["diagnostic_counts"]["classification_conflict"] == 2
    diagnostics = read_jsonl(tmp_path / "classified" / "diagnostics.jsonl")
    assert any(d["origin_mode"] == "inventory" for d in diagnostics)
    assert all(d["schema_version"] == "1.1" for d in diagnostics)


@pytest.mark.parametrize("captions,expected", [
    (["Histogram", "ROC curve"], "plot"),
    (["Histogram", "Detector schematic"], "mixed"),
    (["Photograph", "Experimental setup diagram"], "generic"),
    (["Histogram", "Simulation"], "unknown"),
    ([None, None], "unknown"),
    (["Histogram", "Detector schematic", None], "mixed"),
    (["Histogram", "Scatter plot and detector diagram"], "unknown"),
])
def test_panel_specific_evidence_and_aggregation(classification_figure, captions, expected):
    classification_figure.caption = "Histogram of all distributions."
    panels = [PanelRecord(
        panel_id=f"panel_{i}", figure_id=classification_figure.figure_id, paper_id=PAPER_ID,
        panel_label=None, source_region={"kind": "latex_span", "image_region": None},
        source_asset_ids=[], caption=caption, caption_raw=None, latex_labels=[])
        for i, caption in enumerate(captions)]
    classification_figure.panels = [p.panel_id for p in panels]
    figure, results = FigureClassifier().classify(classification_figure, panels)
    assert figure.figure_type == expected
    assert [p.panel_id for p in results] == classification_figure.panels
    assert all(p.schema_version == "1.1" and p.classification_confidence is None for p in results)
    for result, caption in zip(results, captions):
        if caption in {None, "Simulation"}:
            assert result.panel_type == "unknown" and result.plot_type is None
    assert all(p.panel_type == "unknown" for p in panels)
    assert figure.classification_evidence["reason"] == "explicit_panel_aggregation"


def test_conflicting_panel_types_are_not_forced(classification_figure):
    classification_figure.panels = ["panel_1"]
    panel = PanelRecord("panel_1", classification_figure.figure_id, PAPER_ID, None, {}, [],
                        "Histogram and scatter plot.", None, [])
    figure, panels = FigureClassifier().classify(classification_figure, [panel])
    assert figure.figure_type == "plot" and panels[0].plot_type == "unknown"


def test_explicit_mixed_panels_resolve_global_content_conflict(classification_figure):
    classification_figure.caption = "Histogram and detector schematic."
    classification_figure.panels = ["left", "right"]
    panels = [PanelRecord(identifier, classification_figure.figure_id, PAPER_ID, None, {}, [], caption, None, [])
              for identifier, caption in (("left", "Histogram"), ("right", "Detector schematic"))]
    result, _ = FigureClassifier().classify(classification_figure, panels)
    assert result.figure_type == "mixed"
    assert result.classification_evidence["contradictory_indicators"] == []


def test_multiasset_and_ratio_candidates_do_not_create_panels(classification_figure):
    classification_figure.caption = "Histogram, with a Data/MC ratio in the lower panel."
    classification_figure.source_assets.append({"asset_id": "asset_02", "reference": "diagram.png"})
    figure, panels = FigureClassifier().classify(classification_figure, [])
    assert figure.figure_type == "plot" and panels == [] and figure.panels == []
    assert figure.classification_evidence["visual_panel_detection_reasons"] == [
        "multiple_assets_without_explicit_panels", "panel_layout_mentioned_without_explicit_panels"]


def test_classify_inventory_v1_preserves_inputs_ids_hashes_and_order(inventory_run, tmp_path, monkeypatch):
    import kamiknows.kamiplotextractor.inventory as module

    run, root = inventory_run(r"""\begin{figure}
\subfloat[Histogram]{\includegraphics{a.pdf}}
\subfloat[Detector schematic]{\includegraphics{b.png}}
\caption{Combined figure}\end{figure}
\begin{figure}\includegraphics{a.pdf}\caption{Simulation data}\end{figure}""", ["a.pdf", "b.png"])
    inventory(run, tmp_path)
    source = tmp_path / "inventory"
    original = {p: p.read_bytes() for p in source.iterdir()}
    source_assets = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    old_figures, old_panels = read_jsonl(source / "figures.jsonl"), read_jsonl(source / "panels.jsonl")
    assert all("classification_evidence" not in f for f in old_figures)
    assert all("classification_method" not in p for p in old_panels)

    def forbidden(*args, **kwargs):
        raise AssertionError("classify must not scan LaTeX or re-extract figures")

    monkeypatch.setattr(module, "scan", forbidden)
    monkeypatch.setattr(module.FigureExtractor, "extract", forbidden)
    summary = run_classification(inventory_dir=source, output_dir=tmp_path / "classified")
    run_classification(inventory_dir=source, output_dir=tmp_path / "classified_repeat")
    for filename in ("figures.jsonl", "panels.jsonl", "figure_summary.json", "diagnostics.jsonl"):
        assert (tmp_path / "classified" / filename).read_bytes() == (tmp_path / "classified_repeat" / filename).read_bytes()
    figures = read_jsonl(tmp_path / "classified" / "figures.jsonl")
    panels = read_jsonl(tmp_path / "classified" / "panels.jsonl")
    for before, after in zip([*old_figures, *old_panels], [*figures, *panels]):
        for key in ("paper_id", "figure_id", "panel_id", "source_assets", "source_hashes", "provenance", "source_region"):
            if key in before:
                assert after[key] == before[key]
        assert after["schema_version"] == "1.1"
        assert not {"points", "bins", "uncertainties", "axes"} & after.keys()
    assert summary["figure_type_counts"] == {"mixed": 1, "unknown": 1}
    assert summary["panel_type_counts"] == {"plot": 1, "generic": 1}
    assert summary["review_required"] == {"figures": 2, "panels": 2}
    assert summary["unknown_figures"] == [figures[1]["figure_id"]]
    assert summary["mixed_figures"] == [figures[0]["figure_id"]]
    manifest = read_manifest(tmp_path / "classified", "extract_plots")
    assert manifest["params"]["mode"] == "classify" and manifest["params"]["schema_version"] == "1.1"
    assert manifest["summary"] == summary
    assert len(manifest["inputs"]) == 5
    for entry in manifest["inputs"].values():
        assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    assert original == {p: p.read_bytes() for p in original}
    assert source_assets == {p: p.read_bytes() for p in source_assets}


@pytest.mark.parametrize("fault", ["schema", "mode", "step", "record_schema", "duplicate", "caption_type",
                                      "missing_panels", "missing_field", "bad_asset", "counts", "summary"])
def test_classify_rejects_invalid_inventory_before_output(inventory_run, tmp_path, fault):
    run, _ = inventory_run(r"\begin{figure}\caption{Histogram}\end{figure}")
    inventory(run, tmp_path)
    source = tmp_path / "inventory"
    manifest = read_json(source / "manifest.json")
    figures = read_jsonl(source / "figures.jsonl")
    if fault == "schema":
        manifest["params"]["schema_version"] = "9.0"
    elif fault == "mode":
        manifest["params"]["mode"] = "classify"
    elif fault == "step":
        manifest["step"] = "parse"
    elif fault == "record_schema":
        figures[0]["schema_version"] = "1.1"
    elif fault == "duplicate":
        figures.append(figures[0])
    elif fault == "caption_type":
        figures[0]["caption"] = ["Histogram"]
    elif fault == "missing_panels":
        figures[0]["panels"] = ["missing"]
    elif fault == "missing_field":
        del figures[0]["section_id"]
    elif fault == "bad_asset":
        figures[0]["source_assets"] = [None]
    elif fault == "counts":
        figures = []
    elif fault == "summary":
        manifest["summary"]["figures"] = 42
    write_json(source / "manifest.json", manifest)
    (source / "figures.jsonl").write_text("".join(json.dumps(f) + "\n" for f in figures))
    with pytest.raises(ValueError):
        run_classification(inventory_dir=source, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_classify_empty_inventory_and_no_overwrite(inventory_run, tmp_path):
    run, _ = inventory_run("No figures.")
    inventory(run, tmp_path)
    source = tmp_path / "inventory"
    summary = run_classification(inventory_dir=source, output_dir=tmp_path / "classified")
    assert summary["figures"] == summary["panels"] == 0
    with pytest.raises(FileExistsError):
        run_classification(inventory_dir=source, output_dir=tmp_path / "classified")
    with pytest.raises(ValueError, match="inside"):
        run_classification(inventory_dir=source, output_dir=source / "child")


def test_classify_cli_needs_no_parse_tree(inventory_run, tmp_path):
    run, _ = inventory_run(r"\begin{figure}\caption{ROC curve}\end{figure}")
    inventory(run, tmp_path)
    run.rename(tmp_path / "moved_parse_run")
    command = [sys.executable, "scripts/extract_plots.py", "--inventory-dir", str(tmp_path / "inventory"),
               "--output-dir", str(tmp_path / "classified"), "--mode", "classify"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read_jsonl(tmp_path / "classified" / "figures.jsonl")[0]["figure_type"] == "plot"
    command[-1] = "inventory"
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0 and "requires --parsed-dir" in result.stderr


# Task 03: exact synthetic ground truth, with no digitization or live network.
@pytest.fixture
def native_fixture(inventory_run, tmp_path):
    def build(plot, axis_options="", files=None, caption="Transverse momentum distribution", sources=None):
        body = (r"\begin{figure}\begin{tikzpicture}\begin{axis}" + axis_options + plot
                + r"\end{axis}\end{tikzpicture}\caption{" + caption + r"}\label{fig:native}\end{figure}")
        run, root = inventory_run(body, sources=sources)
        for name, value in (files or {}).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        inventory(run, tmp_path)
        run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
        figures = [FigureRecord(**f) for f in read_jsonl(tmp_path / "classified" / "figures.jsonl")]
        panels = [PanelRecord(**p) for p in read_jsonl(tmp_path / "classified" / "panels.jsonl")]
        return run, root, figures, panels
    return build


def native_extract(run, root, figures, panels):
    from kamiknows.kamiplotextractor.source_data import recover_native
    return recover_native(paper_id=PAPER_ID, flat_path=run / "papers" / PAPER_ID / "flat.tex",
                          source_root=root.parent, figures=figures, panels=panels)


def test_native_coordinates_precision_and_multiple_series(native_fixture):
    run, root, figures, panels = native_fixture(
        r"\addplot coordinates {(1,2) (-2.5,4e-3) (3,8)};\addlegendentry{Signal}"
        r"\addplot coordinates {(1,-2E2) (2,0.123456789012345678901)};\addlegendentry{Background}")
    results, _ = native_extract(run, root, figures, panels)
    assert len(results) == 2 and all(r.source_method == "original_source_data" for r in results)
    assert results[0].series[0]["x"] == [1, -2.5, 3]
    assert results[0].series[0]["y"] == [2, 0.004, 8]
    assert results[1].series[0]["y"][0] == -200
    assert results[1].series[0]["raw_values"][1][1] == "0.123456789012345678901"
    assert [r.series[0]["label"] for r in results] == ["Signal", "Background"]
    assert [r.series[0]["series_index"] for r in results] == [0, 1]
    assert all(r.schema_version == "1.2" and r.review_status == "pending" for r in results)


@pytest.mark.parametrize("command,files,x,y", [
    (r"\addplot table[x=energy,y=counts]{data.dat};", {"data.dat": "energy counts\n1 4\n2 8\n"}, [1, 2], [4, 8]),
    (r"\addplot table[x index=0,y index=2,col sep=comma]{data.csv};",
     {"data.csv": "energy,unused,counts\n1,42,4\n2,42,8\n"}, [1, 2], [4, 8]),
    ("\\addplot table[x=energy,y=counts]{\nenergy counts\n1 4\n2 8\n};", {}, [1, 2], [4, 8]),
    (r"\addplot table[row sep=\\]{1 4\\2 8\\};", {}, [1, 2], [4, 8]),
    (r"\addplot table{data.dat};", {"data.dat": "1 4\n2 8\n"}, [1, 2], [4, 8]),
])
def test_native_tables(native_fixture, command, files, x, y):
    run, root, figures, panels = native_fixture(command, files=files)
    results, inputs = native_extract(run, root, figures, panels)
    record = results[0]
    assert record.extraction_status == "success", record.warnings
    assert record.series[0]["x"] == x and record.series[0]["y"] == y
    assert record.series[0]["uncertainties"] == []
    assert record.series[0]["columns"]["default_first_two"] == ("x=" not in command and "x index" not in command)
    if files:
        assert any(key.startswith("data:") for key in inputs)
        assert record.provenance["sources"][-1]["sha256"] == hashlib.sha256((root / next(iter(files))).read_bytes()).hexdigest()


def test_native_axes_and_symmetric_uncertainty(native_fixture):
    run, root, figures, panels = native_fixture(
        r"\addplot[error bars/.cd,y dir=both,y explicit] table[x=energy,y=counts,y error=err]{data.dat};",
        axis_options=r"[xlabel={Energy},ylabel={Counts},xmode=log,x unit={GeV}]",
        files={"data.dat": "energy counts err\n1 4 0.2\n2 8 0.5\n"})
    results, _ = native_extract(run, root, figures, panels)
    record = results[0]
    assert record.extraction_status == "success", record.warnings
    assert record.axes["x"] == {"label": "Energy", "scale": "log", "unit": "GeV"}
    assert record.axes["y"]["unit"] is None
    uncertainty = record.series[0]["uncertainties"][0]
    assert uncertainty["symerror"] == [0.2, 0.5] and uncertainty["columns"] == ["err"]
    assert uncertainty["kind"] is None and uncertainty["label"] is None


@pytest.mark.parametrize("command,files,code", [
    (r"\addplot table{missing.dat};", {}, "data_file_missing"),
    (r"\addplot coordinates {(1,nan)};", {}, "invalid_numeric_value"),
    (r"\addplot coordinates {(1,\value)};", {}, "unresolved_macro"),
    (r"\addplot {x*x};", {}, "unsupported_expression"),
    (r"\addplot table[x expr=\thisrow{x}*2,y=y]{data.dat};", {"data.dat": "x y\n1 2\n"}, "unsupported_expression"),
    (r"\addplot[y filter/.expression={y*2}] coordinates {(1,2)};", {}, "unsupported_transformation"),
    (r"\addplot table[x=missing,y=y]{data.dat};", {"data.dat": "x y\n1 2\n"}, "ambiguous_columns"),
    (r"\addplot table{../outside.dat};", {}, "unsafe_path"),
    (r"\addplot table{data.root};", {}, "unsupported_data_format"),
])
def test_native_failures_never_emit_untransformed_vectors(native_fixture, command, files, code):
    run, root, figures, panels = native_fixture(command, files=files)
    results, _ = native_extract(run, root, figures, panels)
    assert results[0].series == [] and results[0].source_method == "unavailable"
    assert results[0].warnings[0]["code"] == code


def test_native_unknown_and_unlinked_candidate(native_fixture):
    run, root, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};", caption="Simulation",
        sources={"standalone.tex": r"\begin{axis}\addplot coordinates {(9,99)};\end{axis}"})
    results, _ = native_extract(run, root, figures, panels)
    assert figures[0].figure_type == "unknown"
    assert results[0].classification["figure_type"] == "unknown" and results[0].series[0]["y"] == [2]
    candidate = next(r for r in results if r.figure_id is None)
    assert candidate.source_method == "candidate_data" and candidate.association["state"] == "candidate"
    assert candidate.series == []


def test_native_unrelated_csv_and_symlink_escape(native_fixture, tmp_path):
    run, root, figures, panels = native_fixture(r"\includegraphics{plot.pdf}", files={"unrelated.csv": "x,y\n1,2\n"})
    results, _ = native_extract(run, root, figures, panels)
    assert results == []
    outside = tmp_path / "private.dat"
    outside.write_text("1 2\n")
    (root / "alias.dat").symlink_to(outside)
    from kamiknows.kamiplotextractor.source_data import DataError, _data_path
    with pytest.raises(DataError, match="outside source root"):
        _data_path(root.parent, "alias.dat", [root / "main.tex"])


@pytest.fixture
def hepdata_table():
    return {
        "independent_variables": [{"header": {"name": "Energy", "units": "GeV"},
                                   "values": [{"low": 0, "high": 10}, {"low": 10, "high": 20}]}],
        "dependent_variables": [{"header": {"name": "Signal", "units": "pb"},
            "qualifiers": [{"name": "SQRT(S)", "value": 13000, "units": "GeV"}],
            "values": [{"value": 5.5, "errors": [
                {"symerror": 0.2, "label": "stat"},
                {"asymerror": {"plus": "5%", "minus": "-3%"}, "label": "sys,calibration"},
                {"symerror": "-2%"}]}, {"value": "-", "errors": []}]}],
    }


def test_hepdata_errors_bins_missing_and_qualifiers(hepdata_table):
    from kamiknows.kamiplotextractor.source_data import parse_hepdata_table
    axes, series, warnings = parse_hepdata_table(hepdata_table)
    assert axes["independent_variables"] == [{"name": "Energy", "units": "GeV"}]
    signal = series[0]
    assert signal["y"] == [5.5, None] and signal["x"] == [None, None]
    assert signal["bins"][0]["low"] == 0 and signal["bins"][0]["high"] == 10
    assert signal["value_states"][1] == {"value": None, "raw": "-", "state": "missing"}
    assert signal["qualifiers"] == hepdata_table["dependent_variables"][0]["qualifiers"]
    errors = signal["uncertainties"][0]["components"]
    assert len(errors) == 3
    assert errors[0]["symerror"]["value"] == 0.2 and errors[0]["kind"] == "statistical"
    assert errors[1]["asymerror"]["minus"] == {"value": -3, "state": "numeric", "raw": "-3%", "relative": True, "unit": "percent"}
    assert errors[1]["kind"] == "systematic"
    assert errors[2]["kind"] is None and errors[2]["symerror"]["value"] == -2
    assert warnings[0]["code"] == "missing_or_nonnumeric_published_value"


def test_hepdata_multidimensional_multiple_variables(hepdata_table):
    from copy import deepcopy
    from kamiknows.kamiplotextractor.source_data import parse_hepdata_table
    hepdata_table["independent_variables"].append({"header": {"name": "Region"}, "values": [{"value": "barrel"}, {"value": "endcap"}]})
    hepdata_table["dependent_variables"].append(deepcopy(hepdata_table["dependent_variables"][0]))
    hepdata_table["dependent_variables"][1]["header"]["name"] = "Background"
    _, series, _ = parse_hepdata_table(hepdata_table)
    assert [s["label"] for s in series] == ["Signal", "Background"]
    assert all(s["x"] is None and s["bins"] is None and len(s["independent_variables"]) == 2 for s in series)
    assert series[0]["independent_variables"][1]["values"][0]["value"]["raw"] == "barrel"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_hepdata_offline_verified_mapping(native_fixture, tmp_path, hepdata_table, suffix):
    from kamiknows.kamiplotextractor.source_data import recover_hepdata
    run, root, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};")
    directory = tmp_path / "hepdata"
    directory.mkdir()
    if suffix == ".yaml":
        import yaml
        content = yaml.safe_dump(hepdata_table)
    else:
        content = json.dumps(hepdata_table)
    (directory / ("table" + suffix)).write_text(content)
    mapping = {"arxiv_id": "1111.00001", "figure_id": figures[0].figure_id,
               "record_id": "123", "table_name": "Table 7", "source_file": "table" + suffix}
    records, inputs = recover_hepdata(figures=figures, panels=panels, papers=read_jsonl(run / "papers.jsonl"),
                                     mappings=[mapping], root=directory)
    assert len(records) == 1 and records[0].source_method == "original_published_data"
    assert records[0].association["state"] == "verified" and records[0].figure_id == figures[0].figure_id
    assert records[0].series[0]["y"] == [5.5, None]
    assert records[0].extraction_status == "partial" and records[0].review_status == "pending"
    assert records[0].provenance["sources"][0]["sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert inputs["table" + suffix] == directory / ("table" + suffix)


@pytest.mark.parametrize("condition,state", [("publication_only", "candidate"), ("wrong_arxiv", "unmatched"),
                                               ("ambiguous_label", "ambiguous"), ("missing_table_identity", "candidate")])
def test_hepdata_unverified_links_never_promoted(native_fixture, tmp_path, condition, state):
    from copy import deepcopy
    from kamiknows.kamiplotextractor.source_data import recover_hepdata
    run, _, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};")
    entry = {"arxiv_id": "1111.00001", "latex_label": "fig:native", "record_id": "123", "table_name": "Table 7"}
    if condition == "publication_only":
        del entry["latex_label"]
    elif condition == "wrong_arxiv":
        entry["arxiv_id"] = "9999.99999"
    elif condition == "ambiguous_label":
        copy = deepcopy(figures[0])
        copy.figure_id += "_other"
        figures.append(copy)
    else:
        del entry["table_name"]
    records, _ = recover_hepdata(figures=figures, panels=panels, papers=read_jsonl(run / "papers.jsonl"),
                                  mappings=[entry], root=tmp_path)
    assert records[0].association["state"] == state
    assert all(not r.series and r.source_method != "original_published_data" for r in records)


def test_hepdata_missing_mapping_and_network_off(native_fixture, tmp_path):
    from kamiknows.kamiplotextractor.source_data import recover_hepdata
    run, _, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};")
    args = dict(figures=figures, panels=panels, papers=read_jsonl(run / "papers.jsonl"), root=tmp_path)
    records, _ = recover_hepdata(**args, mappings=[])
    assert records[0].association["state"] == "unmatched"
    records, _ = recover_hepdata(**args, mappings=[{"paper_id": PAPER_ID, "figure_id": figures[0].figure_id,
                                                  "record_id": "123", "table_name": "Table 7"}])
    assert records[0].warnings[0]["code"] == "network_disabled"


def test_hepdata_safe_loader_and_raw_precision():
    from kamiknows.kamiplotextractor.source_data import DataError, decode_hepdata
    table, _ = decode_hepdata(b'{"value": 0.123456789012345678901}', ".json")
    assert table["value"] == "0.123456789012345678901"
    with pytest.raises(DataError):
        decode_hepdata(b"!!python/object/apply:os.system ['echo unsafe']", ".yaml")
    with pytest.raises(DataError, match="aliases"):
        decode_hepdata(b"a: &value [1, 2]\nb: *value", ".yaml")


def test_hepdata_network_failure_is_bounded_and_isolated(native_fixture, tmp_path, monkeypatch):
    import requests
    from kamiknows.kamiplotextractor.source_data import recover_hepdata
    calls = []
    def failure(url, **kwargs):
        calls.append((url, kwargs))
        raise requests.Timeout("mock timeout")
    monkeypatch.setattr(requests, "get", failure)
    run, _, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};")
    records, _ = recover_hepdata(figures=figures, panels=panels, papers=read_jsonl(run / "papers.jsonl"), root=tmp_path,
                                enable_network=True, mappings=[{"paper_id": PAPER_ID, "figure_id": figures[0].figure_id,
                                                                "record_id": "123", "table_name": "Table 7"}])
    assert len(calls) == 3 and all(c[1]["timeout"] == (10, 20) and c[1]["stream"] for c in calls)
    assert records[0].series == [] and records[0].warnings[0]["code"] == "hepdata_backend_error"


@pytest.mark.parametrize("scenario", ["success", "redirect_escape", "oversize", "stream_oversize"])
def test_hepdata_network_limits(monkeypatch, scenario):
    import requests
    import kamiknows.kamiplotextractor.source_data as data
    class Response:
        status_code = 302 if scenario == "redirect_escape" else 200
        headers = {"Location": "https://other.example/table"} if scenario == "redirect_escape" else {
            "Content-Length": "100" if scenario == "oversize" else "1"}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size):
            yield b"x" * (100 if scenario == "stream_oversize" else 1)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(data, "MAX_SOURCE_BYTES", 10)
    if scenario == "success":
        content, url = data.fetch_hepdata("123", "Table 7")
        assert content == b"x" and "table=Table+7" in url
    else:
        with pytest.raises(data.DataError):
            data.fetch_hepdata("123", "Table 7")


def test_recovery_pipeline_stable_ids_manifest_and_immutable_inputs(native_fixture, tmp_path):
    run, root, figures, _ = native_fixture(r"\addplot table[x=x,y=y]{data.dat};", files={"data.dat": "x y\n1 2\n2 4\n"})
    before = {p: p.read_bytes() for directory in (run, tmp_path / "classified", tmp_path / "inventory")
              for p in directory.rglob("*") if p.is_file()}
    kwargs = dict(classified_dir=tmp_path / "classified", parsed_dir=run)
    summary = run_recovery(**kwargs, output_dir=tmp_path / "recovery")
    run_recovery(**kwargs, output_dir=tmp_path / "repeat")
    for name in ("plot_data.jsonl", "diagnostics.jsonl", "recovery_summary.json"):
        assert (tmp_path / "recovery" / name).read_bytes() == (tmp_path / "repeat" / name).read_bytes()
    data = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")
    assert len(data) == 1 and data[0]["figure_id"] == figures[0].figure_id
    assert data[0]["source_method"] == "original_source_data" and data[0]["series"][0]["y"] == [2, 4]
    assert summary["recovered_series"] == 1
    manifest = read_manifest(tmp_path / "recovery", "extract_plots")
    assert manifest["params"]["mode"] == "recover" and manifest["params"]["schema_version"] == "1.2"
    assert manifest["summary"] == summary
    assert any("data:extracted/data.dat" in role for role in manifest["inputs"])
    for entry in manifest["inputs"].values():
        assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    assert before == {p: p.read_bytes() for p in before}
    with pytest.raises(FileExistsError):
        run_recovery(**kwargs, output_dir=tmp_path / "recovery")
    with pytest.raises(ValueError, match="inside"):
        run_recovery(**kwargs, output_dir=run / "nested")


def test_recovery_accepts_relocated_parse_and_inventory(native_fixture, tmp_path):
    run, _, _, _ = native_fixture(r"\addplot coordinates {(1,2)};")
    moved = tmp_path / "relocated"
    run.rename(moved)
    moved_inventory = tmp_path / "relocated_inventory"
    (tmp_path / "inventory").rename(moved_inventory)
    summary = run_recovery(classified_dir=tmp_path / "classified", parsed_dir=moved, inventory_dir=moved_inventory,
                           output_dir=tmp_path / "recovery")
    assert summary["recovered_series"] == 1


@pytest.mark.parametrize("fault", ["source", "flat", "classified_ids", "upstream", "parse_identity", "classified_schema"])
def test_recovery_rejects_untrustworthy_associations(native_fixture, tmp_path, fault):
    run, root, _, _ = native_fixture(r"\addplot coordinates {(1,2)};")
    if fault == "source":
        (root / "main.tex").write_text("changed")
    elif fault == "flat":
        (run / "papers" / PAPER_ID / "flat.tex").write_text("changed")
    elif fault == "classified_ids":
        path = tmp_path / "classified" / "figures.jsonl"
        records = read_jsonl(path)
        records[0]["figure_id"] = "wrong_figure"
        path.write_text(json.dumps(records[0]) + "\n")
    elif fault == "upstream":
        (tmp_path / "inventory" / "figures.jsonl").write_text("")
    elif fault == "parse_identity":
        (run / "papers.jsonl").write_text("{}\n")
    else:
        path = tmp_path / "classified" / "manifest.json"
        manifest = read_json(path)
        manifest["params"]["schema_version"] = "1.0"
        write_json(path, manifest)
    kwargs = dict(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "recovery")
    if fault in {"source", "flat"}:
        summary = run_recovery(**kwargs)
        assert summary["recovered_series"] == 0 and summary["paper_results"][0]["status"] == "failed"
        records = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")
        assert all(r["source_method"] == "unavailable" and r["series"] == [] for r in records)
    else:
        with pytest.raises(ValueError):
            run_recovery(**kwargs)
        assert not (tmp_path / "recovery").exists()


def test_recovery_failure_isolation_between_papers(inventory_run, tmp_path):
    import shutil
    second_id = "2222_00002v1"
    run, root = inventory_run(r"\begin{figure}\begin{axis}\addplot coordinates {(1,2)};\end{axis}\caption{Histogram}\end{figure}")
    shutil.copytree(run / "papers" / PAPER_ID, run / "papers" / second_id)
    papers = read_jsonl(run / "papers.jsonl")
    papers.append({"paper_id": second_id, "arxiv_id": "2222.00002v1", "status": "success"})
    (run / "papers.jsonl").write_text("".join(json.dumps(p) + "\n" for p in papers))
    inventory(run, tmp_path)
    run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
    (root / "main.tex").write_text("modified source")
    summary = run_recovery(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "recovery")
    records = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")
    assert summary["recovered_series"] == 1
    assert {r["paper_id"]: r["source_method"] for r in records} == {PAPER_ID: "unavailable", second_id: "original_source_data"}


def test_recovery_mixed_and_generic_override(inventory_run, tmp_path):
    body = r"""\begin{figure}
\subfloat[Histogram]{\begin{axis}\addplot coordinates {(1,2)};\end{axis}}
\subfloat[Detector schematic]{\begin{axis}\addplot coordinates {(9,99)};\end{axis}}
\caption{Combined}\end{figure}"""
    run, _ = inventory_run(body)
    inventory(run, tmp_path)
    run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
    args = dict(classified_dir=tmp_path / "classified", parsed_dir=run)
    summary = run_recovery(**args, output_dir=tmp_path / "recovery")
    records = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")
    assert summary["recovered_series"] == 1
    assert records[0]["classification"] == {"figure_type": "mixed", "panel_type": "plot", "plot_type": "histogram"}
    assert records[0]["panel_id"] is not None and records[0]["series"][0]["y"] == [2]
    override = run_recovery(**args, output_dir=tmp_path / "override", include_generic=True)
    assert override["recovered_series"] == 2


@pytest.mark.parametrize("mode", ["hepdata", "auto", "compare"])
def test_recovery_hepdata_modes_and_compare(native_fixture, tmp_path, mode):
    run, _, figures, _ = native_fixture(r"\addplot coordinates {(1,2) (2,4)};\addlegendentry{Signal}",
        axis_options=r"[xlabel={Energy},ylabel={Signal},x unit={GeV},y unit={pb}]")
    data_dir = tmp_path / "hepdata"
    data_dir.mkdir()
    write_json(data_dir / "table.json", {
        "independent_variables": [{"header": {"name": "Energy", "units": "GeV"}, "values": [{"value": 1}, {"value": 2}]}],
        "dependent_variables": [{"header": {"name": "Signal", "units": "pb"}, "values": [{"value": 2.5}, {"value": 4}]}]})
    write_json(data_dir / "mapping.json", {"schema_version": "1", "mappings": [
        {"paper_id": PAPER_ID, "figure_id": figures[0].figure_id, "record_id": "123", "table_name": "Table 7", "source_file": "table.json"}]})
    summary = run_recovery(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "recovery",
                           source_method=mode, hepdata_dir=data_dir)
    assert summary["source_method_counts"]["original_published_data"] == 1
    assert summary["recovered_series"] == (1 if mode == "hepdata" else 2)
    if mode == "compare":
        assert summary["comparisons"][0]["status"] == "compared"
        assert summary["comparisons"][0]["y_differences"] == ["-0.5", "0"]
        assert summary["comparisons"][0]["scientific_agreement"] == "manual_review_pending"
    else:
        assert summary["comparisons"] == []


def test_recovery_cli_and_offline_default(native_fixture, tmp_path, monkeypatch):
    import requests
    run, _, _, _ = native_fixture(r"\addplot coordinates {(1,2)};")
    def forbidden(*args, **kwargs):
        raise AssertionError("No network allowed")
    monkeypatch.setattr(requests, "get", forbidden)
    summary = run_recovery(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "offline",
                           source_method="auto")
    assert summary["recovered_series"] == 1
    command = [sys.executable, "scripts/extract_plots.py", "--mode", "recover", "--classified-dir", str(tmp_path / "classified"),
               "--parsed-dir", str(run), "--output-dir", str(tmp_path / "cli_recovery")]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read_manifest(tmp_path / "cli_recovery", "extract_plots")["summary"]["recovered_series"] == 1


@pytest.mark.parametrize("env,xscale,yscale", [("loglogaxis", "log", "log"), ("semilogxaxis", "log", None),
                                               ("semilogyaxis", None, "log")])
def test_native_log_axis_environments(inventory_run, tmp_path, env, xscale, yscale):
    run, root = inventory_run(r"\begin{figure}\begin{" + env + r"}\addplot coordinates {(1,2)};\end{" + env
                              + r"}\caption{Histogram}\end{figure}")
    inventory(run, tmp_path)
    run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
    run_recovery(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "recovery")
    data = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")[0]
    assert data["axes"]["x"]["scale"] == xscale and data["axes"]["y"]["scale"] == yscale


def test_native_asymmetric_uncertainty_and_included_source(inventory_run, tmp_path):
    run, root = inventory_run(r"\begin{figure}\input{plots/native}\caption{Histogram}\end{figure}", sources={
        "plots/native.tex": r"\begin{axis}\addplot[error bars/.cd,y dir=both,y explicit]"
                            r"table[x=x,y=y,y error plus=up,y error minus=down]{data.dat};\end{axis}"})
    (root / "plots" / "data.dat").write_text("x y up down\n1 2 0.3 0.2\n")
    inventory(run, tmp_path)
    run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
    run_recovery(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "recovery")
    data = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")[0]
    error = data["series"][0]["uncertainties"][0]
    assert error["representation"] == "asymmetric" and error["plus"] == [0.3] and error["minus"] == [0.2]
    assert data["association"]["source_origins"][0]["path"] == "extracted/plots/native.tex"


def test_native_plot_outside_all_figures_is_candidate(inventory_run, tmp_path):
    run, _ = inventory_run(r"\begin{axis}\addplot coordinates {(1,2)};\end{axis}")
    inventory(run, tmp_path)
    run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
    summary = run_recovery(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "recovery")
    assert summary["recovered_series"] == 0 and summary["association_counts"] == {"candidate": 1}
    record = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")[0]
    assert record["figure_id"] is None and record["source_method"] == "candidate_data" and record["series"] == []


def test_native_generic_figure_skipped_unless_overridden(native_fixture, tmp_path):
    run, _, _, _ = native_fixture(r"\addplot coordinates {(1,2)};", caption="Detector schematic")
    args = dict(classified_dir=tmp_path / "classified", parsed_dir=run)
    assert run_recovery(**args, output_dir=tmp_path / "skip")["records"] == 0
    assert run_recovery(**args, output_dir=tmp_path / "override", include_generic=True)["recovered_series"] == 1


def test_native_ambiguous_external_origin_is_not_guessed(inventory_run, tmp_path):
    command = r"\addplot table[x=x,y=y]{data.dat};"
    run, root = inventory_run(r"\begin{figure}\begin{axis}" + command + r"\end{axis}\caption{Histogram}\end{figure}",
                              sources={"unrelated/standalone.tex": r"\begin{axis}" + command + r"\end{axis}"})
    (root / "data.dat").write_text("x y\n1 2\n")
    inventory(run, tmp_path)
    run_classification(inventory_dir=tmp_path / "inventory", output_dir=tmp_path / "classified")
    summary = run_recovery(classified_dir=tmp_path / "classified", parsed_dir=run, output_dir=tmp_path / "recovery")
    assert summary["recovered_series"] == 0
    record = read_jsonl(tmp_path / "recovery" / "plot_data.jsonl")[0]
    assert record["association"]["state"] == "ambiguous" and record["source_method"] == "candidate_data"


def test_hepdata_bad_metadata_and_source_escape(native_fixture, tmp_path, hepdata_table):
    from kamiknows.kamiplotextractor.source_data import recover_hepdata
    run, _, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};")
    directory = tmp_path / "hepdata"
    directory.mkdir()
    hepdata_table["arxiv_id"] = "9999.99999"
    write_json(directory / "table.json", hepdata_table)
    entry = {"paper_id": PAPER_ID, "figure_id": figures[0].figure_id, "record_id": "123", "table_name": "Table 7", "source_file": "table.json"}
    args = dict(figures=figures, panels=panels, papers=read_jsonl(run / "papers.jsonl"), root=directory)
    records, _ = recover_hepdata(**args, mappings=[entry])
    assert records[0].association["state"] == "unmatched" and records[0].series == []
    entry["source_file"] = "../secret.json"
    records, _ = recover_hepdata(**args, mappings=[entry])
    assert records[0].warnings[0]["code"] == "unsafe_path"


def test_hepdata_missing_values_empty_sides_and_infinite_bins(hepdata_table):
    from kamiknows.kamiplotextractor.source_data import parse_hepdata_table
    hepdata_table["independent_variables"][0]["values"][0]["low"] = "-.inf"
    hepdata_table["dependent_variables"][0]["values"][0]["errors"] = [{"asymerror": {"plus": "", "minus": -0.3}}]
    hepdata_table["dependent_variables"][0]["values"][1]["value"] = "not measured"
    _, series, warnings = parse_hepdata_table(hepdata_table)
    assert series[0]["bins"][0]["raw"]["low"] == "-.inf"
    assert series[0]["uncertainties"][0]["components"][0]["asymerror"]["plus"]["state"] == "missing"
    assert series[0]["value_states"][1]["state"] == "nonnumeric" and warnings


def test_hepdata_rejects_different_column_lengths(hepdata_table):
    from kamiknows.kamiplotextractor.source_data import DataError, parse_hepdata_table
    hepdata_table["independent_variables"][0]["values"].pop()
    with pytest.raises(DataError, match="different row counts"):
        parse_hepdata_table(hepdata_table)


def test_hepdata_mapping_ids_do_not_depend_on_entry_order(native_fixture, tmp_path, hepdata_table):
    from kamiknows.kamiplotextractor.source_data import recover_hepdata
    run, _, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};")
    write_json(tmp_path / "table.json", hepdata_table)
    entries = [{"paper_id": PAPER_ID, "figure_id": figures[0].figure_id, "record_id": "123", "table_name": table, "source_file": "table.json"}
               for table in ("A", "B")]
    args = dict(figures=figures, panels=panels, papers=read_jsonl(run / "papers.jsonl"), root=tmp_path)
    forward, _ = recover_hepdata(**args, mappings=entries)
    reverse, _ = recover_hepdata(**args, mappings=list(reversed(entries)))
    assert [r.data_id for r in forward] == list(reversed([r.data_id for r in reverse]))


def test_compare_requires_exact_axes_units_and_decimal_coordinates(native_fixture, tmp_path):
    from kamiknows.kamiplotextractor.source_data import compare_sources, data_record, parse_hepdata_table
    run, root, figures, panels = native_fixture(r"\addplot coordinates {(1.00000000000000000001,2)};\addlegendentry{Signal}",
        axis_options=r"[xlabel={Energy},ylabel={Signal},x unit={GeV},y unit={pb}]")
    native, _ = native_extract(run, root, figures, panels)
    table = {"independent_variables": [{"header": {"name": "Energy", "units": "GeV"}, "values": [{"value": 1}]}],
             "dependent_variables": [{"header": {"name": "Signal", "units": "pb"}, "values": [{"value": 2}]}]}
    published = data_record(PAPER_ID, figures[0], None, "hepdata", "test", {"state": "verified"})
    published.source_method = "original_published_data"
    published.axes, published.series, _ = parse_hepdata_table(table)
    published.series[0]["series_id"] = "published-series"
    result = compare_sources([*native, published])[0]
    assert result["status"] == "not_comparable" and result["reason"] == "different_coordinates_or_missing_values"
    published.axes["independent_variables"][0]["units"] = "MeV"
    assert compare_sources([*native, published])[0]["reason"] == "no_unique_explicit_series_and_axes_match"


def test_hepdata_mocked_online_import_preserves_download(native_fixture, tmp_path, hepdata_table, monkeypatch):
    from kamiknows.kamiplotextractor import source_data
    run, _, figures, panels = native_fixture(r"\addplot coordinates {(1,2)};")
    content = json.dumps(hepdata_table).encode()
    monkeypatch.setattr(source_data, "fetch_hepdata", lambda *args: (content, "https://www.hepdata.net/download/table/mock/yaml"))
    records, inputs = source_data.recover_hepdata(figures=figures, panels=panels, papers=read_jsonl(run / "papers.jsonl"), root=tmp_path,
        enable_network=True, mappings=[{"paper_id": PAPER_ID, "figure_id": figures[0].figure_id, "record_id": "123", "table_name": "Table 7"}])
    assert records[0].source_method == "original_published_data" and inputs == {}
    assert records[0].raw_source["download_text"] == content.decode()
    assert records[0].provenance["sources"][0]["sha256"] == hashlib.sha256(content).hexdigest()


def test_recovery_hashes_invalid_data_before_decode(native_fixture, tmp_path):
    run, root, figures, panels = native_fixture(r"\addplot table{data.dat};")
    (root / "data.dat").write_bytes(b"\xff\xfe")
    records, inputs = native_extract(run, root, figures, panels)
    assert records[0].series == [] and records[0].extraction_status == "unavailable"
    assert records[0].provenance["sources"][-1]["sha256"] == hashlib.sha256(b"\xff\xfe").hexdigest()
    assert "data:extracted/data.dat" in inputs
