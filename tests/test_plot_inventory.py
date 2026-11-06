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
                                      run_classification, run_inventory)
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
