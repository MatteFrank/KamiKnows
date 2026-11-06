"""Match flattened graphics to original sources and resolve local assets safely."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kamiknows.common import sha256_file

from .latex import Command, Document, Environment, issue, option_values, scan

EXTENSIONS = (".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg")


def local_path(root: Path, path: Path) -> Path:
    """Check containment before any read, including symlink targets."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path outside source root: {path}")
    return resolved


def _literal_path(value: str) -> str | None:
    if not value:
        return "empty_reference"
    if ":" in value or value.startswith("/"):
        return "unsafe_path"
    if ".." in Path(value).parts:
        return "unsafe_path"
    if any(char in value for char in "\\{}#$~\x00"):
        return "unresolved_macro"
    return None


def _key(command: Command) -> tuple[str, tuple[str, ...], bool]:
    return command.value.strip(), tuple(command.options), command.starred


def _signature(document: Document, env: Environment) -> str:
    return re.sub(r"\s+", "", document.text[env.start:env.end])


@dataclass
class Origin:
    path: Path
    document: Document
    command: Command


class SourceTree:
    """Index source commands; filenames alone never establish figure provenance."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.inputs: dict[str, Path] = {}
        self.warnings: list[dict[str, Any]] = []
        self.origins: dict[tuple[str, tuple[str, ...], bool], list[Origin]] = {}
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
            for name in sorted(files):
                if Path(name).suffix.lower() != ".tex":
                    continue
                path = Path(directory) / name
                relative = path.relative_to(self.root).as_posix()
                try:
                    safe = local_path(self.root, path)
                    if not safe.is_file():
                        continue
                    document = scan(safe.read_text(encoding="utf-8", errors="replace"))
                    self.inputs[relative] = safe
                except (OSError, ValueError) as exc:
                    self.warnings.append(issue("source_unreadable", str(exc), source_file=relative))
                    continue
                for command in document.commands:
                    if command.name == "includegraphics":
                        self.origins.setdefault(_key(command), []).append(Origin(safe, document, command))

    def match(self, document: Document, figure: Environment, command: Command) -> list[Origin]:
        candidates = self.origins.get(_key(command), [])
        exact = []
        signature = _signature(document, figure)
        siblings = [c for c in document.commands if c.name == "includegraphics"
                    and figure.start <= c.start < figure.end and _key(c) == _key(command)]
        ordinal = siblings.index(command)
        for origin in candidates:
            for env in origin.document.environments:
                if (env.name in {"figure", "figure*"} and env.start <= origin.command.start < env.end
                        and _signature(origin.document, env) == signature):
                    matches = [c for c in origin.document.commands if c.name == "includegraphics"
                               and env.start <= c.start < env.end and _key(c) == _key(command)]
                    if ordinal < len(matches) and matches[ordinal] == origin.command:
                        exact.append(origin)
        return exact or candidates

    def resolve(self, document: Document, figure: Environment, command: Command,
                asset_id: str, panel_id: str | None) -> dict[str, Any]:
        origins = self.match(document, figure, command)
        warnings = []
        reference = command.value.strip()
        options = option_values(command.options[-1]) if command.options else {}
        record: dict[str, Any] = {
            "asset_id": asset_id, "panel_id": panel_id, "reference": reference,
            "path": None, "format": None, "sha256": None, "status": "unresolved",
            "flat_span": [command.start, command.end],
            "includegraphics_raw": document.raw[command.start:command.end],
            "options_raw": [document.raw[slice(*span)] for span in command.option_spans],
            "options": options, "starred": command.starred,
            "layout": {key: options.get(key) for key in
                       ("width", "height", "scale", "trim", "viewport", "clip", "angle", "origin", "page")},
            "source_origins": [{"path": o.path.relative_to(self.root).as_posix(),
                                "span": [o.command.start, o.command.end]} for o in origins],
            "graphicspath": list(command.graphicspath), "candidates": [], "warnings": warnings,
        }

        def warn(code: str, message: str) -> None:
            warnings.append(issue(code, message, command.start, asset_id=asset_id, reference=reference))

        invalid = _literal_path(reference)
        if invalid:
            warn(invalid, "Graphic reference is not a safe literal local path.")
            return record
        suffix = Path(reference).suffix.lower()
        if suffix and suffix not in EXTENSIONS:
            warn("unsupported_format", f"Unsupported graphics extension: {suffix}")
            return record
        if not origins:
            warn("source_origin_unresolved", "No matching includegraphics command in the original .tex files.")
        elif len(origins) > 1:
            warn("source_origin_ambiguous", "Multiple original command locations match; all are retained.")

        bases = {origin.path.parent for origin in origins}
        bases.add(self.root)
        extracted = self.root / "extracted"
        if extracted.is_dir():
            bases.add(extracted)
        graphics_paths = set(command.graphicspath)
        for origin in origins:
            graphics_paths.update(origin.command.graphicspath)
        search_dirs = set(bases)
        for entry in sorted(graphics_paths):
            invalid = _literal_path(entry)
            if invalid:
                warn(invalid, f"Ignored graphicspath entry: {entry!r}")
                continue
            search_dirs.update(base / entry for base in bases)
        matches: dict[str, Path] = {}
        for directory in sorted(search_dirs):
            try:
                requested = directory / reference
                parent = local_path(self.root, requested.parent)
                if not parent.is_dir():
                    continue
                # Actual directory entries avoid duplicate lower/uppercase candidates
                # on case-insensitive filesystems without assuming a TeX engine.
                candidates = [requested] if suffix else sorted(
                    entry for entry in parent.iterdir()
                    if entry.stem == requested.name and entry.suffix.lower() in EXTENSIONS)
                for candidate in candidates:
                    safe = local_path(self.root, candidate)
                    if safe.is_file():
                        matches[safe.relative_to(self.root).as_posix()] = safe
            except (ValueError, OSError) as exc:
                warn("unsafe_path", str(exc))
        record["candidates"] = sorted(matches)
        if not matches:
            warn("asset_missing", "No supported local asset resolves this reference.")
        elif len(matches) > 1:
            warn("asset_ambiguous", "Several assets resolve this reference; none was selected.")
        else:
            relative, path = next(iter(matches.items()))
            try:
                digest = sha256_file(path)
            except OSError as exc:
                warn("asset_unreadable", str(exc))
                return record
            record.update(path=relative, format=path.suffix.lower().lstrip("."),
                          sha256=digest, status="resolved")
            self.inputs[relative] = path
        return record
