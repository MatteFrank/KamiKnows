"""Figure and explicit-panel inventory, independent of classification or digitization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kamiknows.parsing import SECTION_LEVELS, SECTION_RE, clean_latex_text

from .assets import SourceTree
from .latex import PANEL_ENVS, Command, Document, Environment, issue, mask_noncode, scan
from .records import FigureRecord, PanelRecord


def _inside(command: Command, span: tuple[int, int]) -> bool:
    return span[0] <= command.start < span[1]


def _raw_value(document: Document, command: Command) -> str:
    return document.raw[slice(*command.value_span)] if command.value_span else command.value


def _caption_text(raw: str) -> str:
    return clean_latex_text(mask_noncode(raw)[0])


class PanelDetector:
    """Detect explicit LaTeX panels; never infer graphical main/ratio regions."""

    def detect(self, document: Document, figure: Environment) -> list[dict[str, Any]]:
        panels = []
        for env in document.environments:
            if env.name in PANEL_ENVS and figure.start < env.start < env.end <= figure.end:
                panels.append({"kind": env.name, "span": (env.start, env.end), "caption_raw": None})
        for command in document.commands:
            if command.body_span and figure.start <= command.start < figure.end:
                caption = _raw_value(document, command) if command.name == "subcaptionbox" else (
                    document.raw[slice(*command.option_spans[-1])] if command.option_spans else None)
                panels.append({"kind": command.name, "span": (command.start, command.end),
                               "caption_raw": caption})
        return sorted(panels, key=lambda panel: panel["span"])


def _section_map(document: Document, sections: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Use saved section IDs only if the scanner reproduces the parser's headings."""
    begin = document.text.find("\\begin{document}")
    start = begin + len("\\begin{document}") if begin >= 0 else 0
    end = document.text.find("\\end{document}", start)
    end = end if end >= 0 else len(document.text)
    matches = list(SECTION_RE.finditer(document.text, start, end))
    expected = [(clean_latex_text(m[2]) or f"Section {i}", SECTION_LEVELS[m[1]])
                for i, m in enumerate(matches, 1)]
    saved = [(s.get("heading"), s.get("level")) for s in sections]
    if not matches and len(sections) == 1 and sections[0].get("heading") == "Full text":
        return [(start, sections[0]["section_id"])]
    if expected != saved:
        return []
    return [(m.start(), s["section_id"]) for m, s in zip(matches, sections)]


class FigureExtractor:
    """Build linked records using flat.tex for discovery and source files for resolution."""

    def extract(self, *, paper_id: str, flat_tex: str, source_root: Path,
                sections: list[dict[str, Any]]) -> tuple[list[FigureRecord], list[PanelRecord],
                                                        list[dict[str, Any]], dict[str, Path]]:
        document = scan(flat_tex)
        source = SourceTree(source_root)
        section_map = _section_map(document, sections)
        figures = []
        panels = []
        figure_envs = [env for env in document.environments if env.name in {"figure", "figure*"}]
        diagnostics = [w for w in document.warnings if not any(
            w.get("offset") is not None and env.start <= w["offset"] < env.end for env in figure_envs)]
        diagnostics.extend(source.warnings)
        assigned: set[int] = set()
        for index, env in enumerate(figure_envs, 1):
            figure_id = f"{paper_id}::figure::{index:04d}"
            warnings = [dict(w) for w in document.warnings
                        if w.get("offset") is not None and env.start <= w["offset"] < env.end]
            if any(other.start < env.start < other.end for other in figure_envs):
                warnings.append(issue("nested_figure", "Nested figure environment retained separately.", env.start))
            explicit = PanelDetector().detect(document, env)
            for number, panel in enumerate(explicit, 1):
                panel["panel_id"] = f"{figure_id}::panel::{number:02d}"

            def panel_for(command: Command) -> dict[str, Any] | None:
                containing = [panel for panel in explicit if _inside(command, panel["span"])]
                return min(containing, key=lambda p: p["span"][1] - p["span"][0]) if containing else None

            # Inner figures own their graphics, preventing duplicate asset assignment.
            nested = [other for other in figure_envs if env.start < other.start < env.end]
            commands = [c for c in document.commands if env.start <= c.start < env.end
                        and not any(other.start <= c.start < other.end for other in nested)]
            captions = [{"raw": _raw_value(document, c), "normalized": _caption_text(_raw_value(document, c)),
                         "short_raw": document.raw[slice(*c.option_spans[0])] if c.option_spans else None,
                         "starred": c.starred}
                        for c in commands if c.name == "caption" and panel_for(c) is None]
            labels = [c.value for c in commands if c.name == "label" and panel_for(c) is None]
            if not captions:
                warnings.append(issue("caption_missing", "Figure has no explicit caption.", env.start))
            if len(captions) > 1:
                warnings.append(issue("multiple_captions", "All captions retained; caption is the first.", env.start))
            assets = []
            for command in commands:
                if command.name != "includegraphics":
                    continue
                assigned.add(command.start)
                panel = panel_for(command)
                asset = source.resolve(document, env, command, f"{figure_id}::asset::{len(assets) + 1:02d}",
                                       panel["panel_id"] if panel else None)
                assets.append(asset)
                warnings.extend(asset["warnings"])
            if not assets:
                warnings.append(issue("no_graphics_references", "No explicit includegraphics command in figure.", env.start))
            section_id = next((identifier for offset, identifier in reversed(section_map)
                               if offset < env.start), None)
            if section_id is None:
                warnings.append(issue("section_unresolved", "No verified saved section for this figure.", env.start))
            for panel in explicit:
                panel_commands = [c for c in commands if panel_for(c) is panel]
                panel_labels = [c.value for c in panel_commands if c.name == "label"]
                panel_captions = [_raw_value(document, c) for c in panel_commands if c.name in {"caption", "subcaption"}]
                raw_caption = panel_captions[0] if panel_captions else panel["caption_raw"]
                if raw_caption is not None:
                    for command in scan(raw_caption).commands:
                        if command.name == "label" and command.value not in panel_labels:
                            panel_labels.append(command.value)
                panel_warnings = [w for w in warnings if w.get("offset") is not None
                                  and panel["span"][0] <= w["offset"] < panel["span"][1]]
                panels.append(PanelRecord(
                    panel_id=panel["panel_id"], figure_id=figure_id, paper_id=paper_id,
                    panel_label=panel_labels[0] if panel_labels else None,
                    source_region={"kind": "latex_span", "environment_or_command": panel["kind"],
                                   "flat_span": list(panel["span"]), "image_region": None},
                    source_asset_ids=[a["asset_id"] for a in assets if a["panel_id"] == panel["panel_id"]],
                    caption=_caption_text(raw_caption) if raw_caption is not None else None,
                    caption_raw=raw_caption, latex_labels=panel_labels, warnings=panel_warnings))
            figures.append(FigureRecord(
                paper_id=paper_id, figure_id=figure_id,
                caption=captions[0]["normalized"] if captions else None,
                caption_raw=captions[0]["raw"] if captions else None,
                latex_label=labels[0] if labels else None, latex_labels=labels, captions=captions,
                section_id=section_id, source_assets=assets,
                source_hashes={a["path"]: a["sha256"] for a in assets if a["path"] is not None},
                panels=[p["panel_id"] for p in explicit],
                provenance={"flat_file": f"papers/{paper_id}/flat.tex", "flat_span": [env.start, env.end],
                            "environment": env.name, "source_root": f"papers/{paper_id}/source",
                            "manual_review": "pending"},
                warnings=warnings, status="partial" if warnings else "success"))
            diagnostics.extend({**w, "figure_id": figure_id} for w in warnings)
        for command in document.commands:
            if command.name == "includegraphics" and command.start not in assigned:
                diagnostics.append(issue("graphics_outside_figure", "Graphic not enclosed in figure or figure*.",
                                         command.start, reference=command.value))
            elif command.name in {"input", "include"}:
                diagnostics.append(issue("unexpanded_input", "Input remains in flat.tex; it is not expanded again.",
                                         command.start, reference=command.value))
            elif command.name == "subcaption" and not any(
                    p.source_region["flat_span"][0] <= command.start < p.source_region["flat_span"][1]
                    for p in panels):
                diagnostics.append(issue("panel_boundary_unresolved", "Subcaption has no supported explicit panel boundary.",
                                         command.start))
        # The existing flattener replaces failed inputs with comments, not commands.
        for line_number, line in enumerate(flat_tex.splitlines(), 1):
            if line.lstrip().startswith("% unresolved input:"):
                diagnostics.append(issue("unexpanded_input", line.strip(), line_number=line_number))
        return figures, panels, diagnostics, source.inputs
