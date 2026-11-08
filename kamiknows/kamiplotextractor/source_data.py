"""Explicit native/HEPData source recovery; no TeX execution or digitization."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from copy import deepcopy
from decimal import Decimal, localcontext
from platform import python_version
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import requests

from kamiknows.common import sha256_file

from .assets import _literal_path, local_path
from .latex import COMMAND, Document, group, issue, option_values, scan
from .records import FigureRecord, PanelRecord, PlotDataRecord

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
AXIS_ENVS = {"axis", "loglogaxis", "semilogxaxis", "semilogyaxis"}
MAX_SOURCE_BYTES = 8 * 1024 * 1024


class DataError(ValueError):
    """An auditable failure, rather than guessed plotted values."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def number(value: Any) -> int | float:
    text = str(value).strip()
    if isinstance(value, bool) or not NUMBER.fullmatch(text):
        code = "unresolved_macro" if "\\" in text else "invalid_numeric_value"
        if code != "unresolved_macro" and any(char in text for char in "*/^()"):
            code = "unsupported_expression"
        raise DataError(code, f"Not a literal finite decimal: {text!r}")
    decimal = Decimal(text)
    if decimal == decimal.to_integral() and abs(decimal) < Decimal("1e100"):
        return int(decimal)
    result = float(decimal)
    if not (-float("inf") < result < float("inf")) or (result == 0 and decimal != 0):
        raise DataError("invalid_numeric_value", "Value is outside the supported JSON numeric range; raw token retained.")
    return result


def _read_bytes(path: Path) -> bytes:
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise DataError("source_too_large", f"Source exceeds {MAX_SOURCE_BYTES} bytes: {path.name}")
    with path.open("rb") as stream:
        content = stream.read(MAX_SOURCE_BYTES + 1)
    if len(content) > MAX_SOURCE_BYTES:
        raise DataError("source_too_large", "Source grew beyond the size limit.")
    return content


def _id(paper: str | None, figure: str | None, panel: str | None, backend: str, source: Any) -> str:
    digest = hashlib.sha256(json.dumps([paper, figure, panel, backend, source], sort_keys=True).encode()).hexdigest()[:24]
    return f"{figure or paper or 'unassociated'}::data::{digest}"


def data_record(paper_id: str | None, figure: FigureRecord | None, panel: PanelRecord | None,
                backend: str, source: Any, association: dict[str, Any]) -> PlotDataRecord:
    return PlotDataRecord(
        paper_id=paper_id, figure_id=figure.figure_id if figure else None,
        panel_id=panel.panel_id if panel else None,
        data_id=_id(paper_id, figure.figure_id if figure else None, panel.panel_id if panel else None, backend, source),
        source_method="candidate_data" if association["state"] in {"candidate", "ambiguous"} else "unavailable",
        association=association,
        classification={"figure_type": figure.figure_type if figure else None,
                        "panel_type": panel.panel_type if panel else None,
                        "plot_type": panel.plot_type if panel else None},
        provenance={"sources": [], "source_identity": source}, backend=backend)


def _skip(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _unbrace(value: str | bool) -> str | bool:
    if isinstance(value, str) and value.startswith("{"):
        body, _, end = group(value, 0)
        if end == len(value):
            return body
    return value


def _options(text: str, pos: int) -> tuple[dict[str, Any], int]:
    pos = _skip(text, pos)
    if text[pos:pos + 1] != "[":
        return {}, pos
    value, _, end = group(text, pos, "[")
    return {k: _unbrace(v) for k, v in option_values(value).items()}, end


def plot_commands(document: Document) -> list[dict[str, Any]]:
    """Focused scan over masked text, preserving existing latex.scan() semantics."""
    result = []
    text = document.text
    pos = 0
    while match := COMMAND.search(text, pos):
        pos = match.end()
        if match[1] != "addplot":
            continue
        item: dict[str, Any] = {"start": match.start(), "options": {}, "table_options": {}, "kind": None}
        try:
            if text[pos:pos + 1] == "3":
                raise DataError("unsupported_data_format", "Three-dimensional addplot is not supported.")
            if text[pos:pos + 1] == "+":
                pos += 1
            item["options"], pos = _options(text, pos)
            pos = _skip(text, pos)
            kind = re.match(r"(?:coordinates|table)\b", text[pos:])
            if kind is None:
                raise DataError("unsupported_expression", "Only literal coordinates and explicit tables are supported.")
            item["kind"] = kind[0]
            pos += len(kind[0])
            if kind[0] == "table":
                item["table_options"], pos = _options(text, pos)
            item["body"], _, pos = group(text, pos)
            end = _skip(text, pos)
            if text[end:end + 1] != ";":
                raise DataError("unsupported_transformation", "Trailing plot paths or transformations are not supported.")
            item["end"] = end + 1
            pos = end + 1
        except (ValueError, DataError) as exc:
            item["error"] = (exc.code if isinstance(exc, DataError) else "unsupported_data_format", str(exc))
            # Do not consume another plot command when recovering from malformed input.
            end = text.find(";", pos)
            next_plot = text.find("\\addplot", pos)
            pos = min([n for n in (end + 1 if end >= 0 else len(text), next_plot) if n >= 0])
            item["end"] = max(pos, match.end())
        item["raw"] = document.raw[item["start"]:item["end"]]
        result.append(item)
    return result


# Unknown styles may transform numbers. Accept a small explicit visual/metadata
# vocabulary; failing closed avoids returning source columns as transformed data.
VISUAL_KEYS = {"width", "height", "title", "xlabel", "ylabel", "x unit", "y unit", "xmode", "ymode",
               "legend entries", "legend pos", "legend style", "grid", "xmin", "xmax", "ymin", "ymax",
               "x dir", "y dir", "compat", "name", "mark", "mark size", "mark options", "color", "draw",
               "fill", "opacity", "line width", "dashed", "dotted", "solid", "thick", "thin", "only marks",
               "sharp plot", "const plot", "smooth", "ybar", "xbar", "legend columns"}
ERROR_KEYS = {"error bars/.cd", "y dir", "y explicit", "y explicit relative", "error bars/y dir",
              "error bars/y explicit", "error bars/y explicit relative"}
TABLE_KEYS = {"x", "y", "x index", "y index", "col sep", "row sep", "header",
              "y error", "y error index", "y error plus", "y error minus",
              "y error plus index", "y error minus index"}


def _check_options(options: dict[str, Any], allowed: set[str]) -> None:
    for key in options:
        if "filter" in key or "trafo" in key:
            raise DataError("unsupported_transformation", f"Coordinate transformation {key!r} is not evaluated.")
        if "expr" in key:
            raise DataError("unsupported_expression", f"Expression option {key!r} is not evaluated.")
        if key not in allowed and not re.fullmatch(r"(?:red|blue|black|green|orange|gray)(?:!\d+)?", key):
            code = "unsupported_error_syntax" if "error" in key else "unsupported_transformation"
            raise DataError(code, f"Unsupported option or custom style: {key}")


def _coordinates(body: str) -> tuple[list[int | float], list[int | float], list[list[str]]]:
    x, y, raw = [], [], []
    pos = 0
    while (pos := _skip(body, pos)) < len(body):
        if body[pos] != "(":
            raise DataError("unresolved_macro" if "\\" in body[pos:] else "invalid_numeric_value", "Expected literal (x,y).")
        end = body.find(")", pos + 1)
        if end < 0:
            raise DataError("invalid_numeric_value", "Unclosed coordinate pair.")
        pair = [v.strip() for v in body[pos + 1:end].split(",")]
        if len(pair) != 2:
            raise DataError("invalid_numeric_value", "Only two-dimensional numeric pairs are supported.")
        x.append(number(pair[0]))
        y.append(number(pair[1]))
        raw.append(pair)
        pos = end + 1
    if not x:
        raise DataError("invalid_numeric_value", "Empty coordinate list.")
    return x, y, raw


def _table(text: str, options: dict[str, Any]) -> tuple[list[list[str]], list[str] | None]:
    _check_options(options, TABLE_KEYS)
    row_sep = options.get("row sep", "newline")
    if row_sep not in {"newline", "\\\\"}:
        raise DataError("unsupported_data_format", "Unsupported row separator.")
    if row_sep == "\\\\":
        text = text.replace("\\\\", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("#", "%"))]
    separator = options.get("col sep", "space")
    if separator == "space":
        rows = [line.split() for line in lines]
    elif separator in {"comma", "tab", "semicolon"}:
        rows = list(csv.reader(lines, delimiter={"comma": ",", "tab": "\t", "semicolon": ";"}[separator]))
        rows = [[cell.strip() for cell in row] for row in rows]
    else:
        raise DataError("unsupported_data_format", "Unsupported column separator.")
    if not rows or len({len(row) for row in rows}) != 1:
        raise DataError("ambiguous_columns", "Empty table or inconsistent row widths.")
    mode = options.get("header", "auto")
    if mode not in {"auto", "true", "false"}:
        raise DataError("unsupported_data_format", "Unsupported header setting.")
    header = None
    if mode == "true" or (mode == "auto" and any(not NUMBER.fullmatch(v) for v in rows[0])):
        if any("\\" in v for v in rows[0]):
            raise DataError("unresolved_macro", "Table header or first row contains macros.")
        if mode == "auto" and any(NUMBER.fullmatch(v) for v in rows[0]):
            raise DataError("invalid_numeric_value", "Mixed numeric/nonnumeric first row is not accepted as a header.")
        header = rows.pop(0)
        if len(set(header)) != len(header):
            raise DataError("ambiguous_columns", "Duplicate table column names.")
    if not rows:
        raise DataError("invalid_numeric_value", "No table data rows.")
    return rows, header


def _column(options: dict[str, Any], name: str, header: list[str] | None, width: int,
            default: int | None = None) -> tuple[int, str]:
    if name in options and name + " index" in options:
        raise DataError("ambiguous_columns", f"Conflicting selectors for {name}.")
    if name in options:
        selector = options[name]
        if not isinstance(selector, str) or "\\" in selector:
            raise DataError("unresolved_macro", f"Unresolved column selector: {selector}")
        if header is None or selector not in header:
            raise DataError("ambiguous_columns", f"Unknown named column: {selector}")
        return header.index(selector), selector
    if name + " index" in options:
        value = str(options[name + " index"])
        if not value.isdecimal():
            raise DataError("ambiguous_columns", f"Nonliteral column index: {value}")
        index = int(value)
    elif default is not None:
        index = default
    else:
        raise DataError("ambiguous_columns", f"Missing explicit column selector for {name}.")
    if index >= width:
        raise DataError("ambiguous_columns", f"Column index {index} outside table.")
    return index, header[index] if header else str(index)


def _table_series(text: str, options: dict[str, Any], plot_options: dict[str, Any]) -> dict[str, Any]:
    rows, header = _table(text, options)
    defaults = not any(key in options for key in ("x", "y", "x index", "y index"))
    xi, xname = _column(options, "x", header, len(rows[0]), 0 if defaults else None)
    yi, yname = _column(options, "y", header, len(rows[0]), 1 if defaults else None)
    result = {"x": [number(row[xi]) for row in rows], "y": [number(row[yi]) for row in rows],
              "raw_values": [[row[xi], row[yi]] for row in rows], "uncertainties": [], "bins": None,
              "columns": {"x": xname, "y": yname, "default_first_two": defaults}}
    has_error = any(key.startswith("y error") for key in options)
    explicit = any(key in plot_options for key in ("y explicit", "error bars/y explicit", "y explicit relative",
                                                   "error bars/y explicit relative"))
    if has_error and not explicit:
        raise DataError("unsupported_error_syntax", "Error columns require explicit PGFPlots error-bar semantics.")
    if explicit and not has_error:
        raise DataError("unsupported_error_syntax", "Explicit error bars have no supported error columns.")
    if has_error:
        symmetric = "y error" in options or "y error index" in options
        asymmetric = any(key.startswith(("y error plus", "y error minus")) for key in options)
        if symmetric and asymmetric:
            raise DataError("ambiguous_columns", "Both symmetric and asymmetric errors specified.")
        names = ["y error"] if symmetric else ["y error plus", "y error minus"]
        columns = [_column(options, name, header, len(rows[0])) for name in names]
        values = [[number(row[index]) for row in rows] for index, _ in columns]
        labels = [name for _, name in columns]
        # Column names are retained, but no stat/sys semantics are guessed from them.
        result["uncertainties"] = [{"label": None, "kind": None, "axis": "y", "columns": labels,
                                    "relative": any("relative" in key for key in plot_options),
                                    "representation": "symmetric" if symmetric else "asymmetric",
                                    "symerror": values[0] if symmetric else None,
                                    "plus": values[0] if asymmetric else None,
                                    "minus": values[1] if asymmetric else None,
                                    "raw_values": [[row[index] for index, _ in columns] for row in rows]}]
    return result


def _data_path(root: Path, reference: str, origins: list[Path]) -> Path:
    invalid = _literal_path(reference)
    if invalid:
        raise DataError(invalid, f"Unsafe or unresolved data reference: {reference}")
    if Path(reference).suffix.lower() not in {".csv", ".dat", ".txt"}:
        raise DataError("unsupported_data_format", "Only explicitly referenced CSV/DAT/TXT tables are supported.")
    bases = {root, root / "extracted", *(p.parent for p in origins)}
    found = set()
    for base in bases:
        try:
            candidate = local_path(root, base / reference)
        except ValueError as exc:
            raise DataError("unsafe_path", str(exc)) from exc
        if candidate.is_file():
            found.add(candidate)
    if not found:
        raise DataError("data_file_missing", f"No local table for {reference}")
    if len(found) != 1:
        raise DataError("ambiguous_columns", f"Several source files resolve {reference}; none selected.")
    return found.pop()


def recover_native(*, paper_id: str, flat_path: Path, source_root: Path, figures: list[FigureRecord],
                   panels: list[PanelRecord], include_generic: bool = False) -> tuple[list[PlotDataRecord], dict[str, Path]]:
    """Recover literal addplot commands linked by verified flat.tex character spans."""
    raw = _read_bytes(flat_path).decode("utf-8", errors="replace")
    document = scan(raw)
    commands = plot_commands(document)
    inputs = {"flat": flat_path}
    originals: list[tuple[Path, Document, list[dict[str, Any]]]] = []
    # Only .tex source commands can establish origin; arbitrary numeric files are
    # never discovered. Off-figure standalone plots remain candidates.
    for path in sorted(source_root.rglob("*.tex")):
        try:
            safe = local_path(source_root, path)
        except ValueError:
            continue
        if path.is_symlink() or not safe.is_file():
            continue
        source_doc = scan(_read_bytes(safe).decode("utf-8", errors="replace"))
        originals.append((safe, source_doc, plot_commands(source_doc)))
        inputs["source:" + safe.relative_to(source_root).as_posix()] = safe
    output = []
    seen_origins: set[tuple[Path, int]] = set()
    for command in commands:
        owners = [f for f in figures if f.provenance["flat_span"][0] <= command["start"] < f.provenance["flat_span"][1]]
        figure = min(owners, key=lambda f: f.provenance["flat_span"][1] - f.provenance["flat_span"][0]) if owners else None
        owned_panels = [p for p in panels if figure and p.figure_id == figure.figure_id
                        and p.source_region["flat_span"][0] <= command["start"] < p.source_region["flat_span"][1]]
        panel = min(owned_panels, key=lambda p: p.source_region["flat_span"][1] - p.source_region["flat_span"][0]) if owned_panels else None
        origin_matches = [(path, item) for path, _, items in originals for item in items if item["raw"] == command["raw"]]
        seen_origins.update((path, item["start"]) for path, item in origin_matches)
        if not include_generic and (figure and figure.figure_type == "generic" or panel and panel.panel_type == "generic"):
            continue
        association = {"state": "verified" if figure else "candidate", "kind": "flat_figure_span",
                       "flat_span": [command["start"], command["end"]],
                       "source_origins": [{"path": p.relative_to(source_root).as_posix(), "offset": c["start"]}
                                          for p, c in origin_matches]}
        record = data_record(paper_id, figure, panel, "native", {"flat_offset": command["start"]}, association)
        record.raw_source = {"command": command["raw"], "plot_options": command["options"],
                             "table_options": command["table_options"]}
        record.provenance["sources"] = [{"role": "flat", "path": f"papers/{paper_id}/flat.tex", "sha256": sha256_file(flat_path)}]
        record.provenance["sources"].extend({"role": "latex_origin", "path": path.relative_to(source_root).as_posix(),
                                             "sha256": sha256_file(path)} for path in sorted({p for p, _ in origin_matches}))
        try:
            if not figure:
                raise DataError("unverified_figure_link", "Plot outside a known figure; numerical data not promoted.")
            if "error" in command:
                raise DataError(*command["error"])
            axes = [e for e in document.environments if e.name in AXIS_ENVS and e.start < command["start"] < e.end]
            if len(axes) != 1 or not axes[0].closed:
                raise DataError("unsupported_data_format", "Plot requires one complete supported PGFPlots axis.")
            axis = axes[0]
            axis_options, _ = _options(document.text, axis.body_start)
            _check_options(axis_options, VISUAL_KEYS)
            _check_options(command["options"], VISUAL_KEYS | ERROR_KEYS)
            # Global style hooks can alter coordinates. Only harmless compat/size
            # declarations are accepted; TeX expansion is never attempted.
            for match in re.finditer(r"\\(?:pgfplotsset|tikzset)\b", document.text):
                settings, _, _ = group(document.text, match.end())
                _check_options(option_values(settings), {"compat", "width", "height"})
            if any(w["code"] == "conditional_not_evaluated" for w in document.warnings):
                raise DataError("unsupported_transformation", "Conditional source requires TeX evaluation.")
            axis_metadata = {}
            for name in ("x", "y"):
                scale = axis_options.get(name + "mode")
                if scale is None and (axis.name == "loglogaxis" or axis.name == f"semilog{name}axis"):
                    scale = "log"
                if scale is not None and scale not in {"linear", "log"}:
                    raise DataError("unresolved_macro", "Axis scale is not literal linear/log.")
                axis_metadata[name] = {"label": axis_options.get(name + "label"), "scale": scale,
                                       "unit": axis_options.get(name + " unit")}
            record.axes = axis_metadata
            record.raw_source["axis_options"] = axis_options
            record.provenance["axis_id"] = f"{figure.figure_id}::axis::{axis.start}"
            if command["kind"] == "coordinates":
                if any("error" in key or "explicit" in key for key in command["options"]):
                    raise DataError("unsupported_error_syntax", "Coordinate error-bar syntax is not yet supported.")
                x, y, values = _coordinates(command["body"])
                series = {"x": x, "y": y, "raw_values": values, "uncertainties": [], "bins": None}
            else:
                body = command["body"].strip()
                inline = "\n" in body or "\\\\" in body
                if inline:
                    table_text = body
                else:
                    if len({p.parent for p, _ in origin_matches}) > 1:
                        record.association["state"] = "ambiguous"
                        record.source_method = "candidate_data"
                        raise DataError("ambiguous_source_origin", "External table command has several possible source directories.")
                    path = _data_path(source_root, body, [p for p, _ in origin_matches])
                    content = _read_bytes(path)
                    relative = path.relative_to(source_root).as_posix()
                    inputs["data:" + relative] = path
                    record.provenance["sources"].append({"role": "table", "path": relative,
                                                         "sha256": hashlib.sha256(content).hexdigest()})
                    table_text = content.decode("utf-8-sig")
                record.raw_source["table_text"] = table_text
                series = _table_series(table_text, command["table_options"], command["options"])
            for name in ("x", "y"):
                if record.axes[name]["scale"] == "log" and any(v <= 0 for v in series[name]):
                    raise DataError("invalid_numeric_value", "Non-positive values on logarithmic axis would be discarded by PGFPlots.")
            siblings = [c for c in commands if axis.start < c["start"] < axis.end]
            order = siblings.index(command)
            legend = None
            next_start = siblings[order + 1]["start"] if order + 1 < len(siblings) else axis.end
            entries = list(re.finditer(r"\\addlegendentry\b", document.text[command["end"]:next_start]))
            if len(entries) == 1:
                legend, _, _ = group(document.text, command["end"] + entries[0].end())
            elif "legend entries" in axis_options:
                labels = list(option_values(str(axis_options["legend entries"])))
                if len(labels) == len(siblings):
                    legend = str(_unbrace(labels[order]))
            series.update(series_id=f"{record.data_id}::series::01", series_index=order, label=legend, qualifiers=[])
            record.series = [series]
            record.source_method = "original_source_data"
            record.extraction_status = "success"
            record.validation_status = "structural_checks_passed"
        except (ValueError, OSError) as exc:
            code = exc.code if isinstance(exc, DataError) else "unsupported_data_format"
            record.warnings.append(issue(code, str(exc), command["start"]))
        output.append(record)
    for path, _, items in originals:
        for command in items:
            if (path, command["start"]) in seen_origins:
                continue
            relative = path.relative_to(source_root).as_posix()
            record = data_record(paper_id, None, None, "native", {"path": relative, "offset": command["start"]},
                                 {"state": "candidate", "kind": "standalone_source_without_figure_link"})
            record.raw_source = {"command": command["raw"]}
            record.provenance["sources"] = [{"path": relative, "sha256": sha256_file(path)}]
            record.warnings = [issue("unverified_figure_link", "Standalone plot has no unambiguous flat figure association.")]
            output.append(record)
    return output, inputs


# HEPData submission-format JSON/YAML. Keep published variables and raw error
# components intact: percentages and signed variations are not absolute errors.
def decode_hepdata(content: bytes, suffix: str) -> tuple[dict[str, Any], str]:
    if len(content) > MAX_SOURCE_BYTES:
        raise DataError("source_too_large", "HEPData source exceeds the size limit.")
    text = content.decode("utf-8-sig")
    if suffix.lower() == ".json":
        data = json.loads(text, parse_float=str)
        version = "stdlib-json / Python " + python_version()
    elif suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise DataError("optional_dependency_missing", "Install optional PyYAML>=6 for YAML, or provide submission-format JSON.") from exc
        # Reject alias graphs before constructing objects; YAML is untrusted input.
        class ExactSafeLoader(yaml.SafeLoader):
            pass
        ExactSafeLoader.add_constructor("tag:yaml.org,2002:float", lambda loader, node: loader.construct_scalar(node))
        try:
            if any(isinstance(token, yaml.tokens.AliasToken) for token in yaml.scan(text)):
                raise DataError("unsupported_data_format", "YAML aliases are not accepted.")
            data = yaml.load(text, Loader=ExactSafeLoader)
        except yaml.YAMLError as exc:
            raise DataError("unsupported_data_format", str(exc)) from exc
        version = "PyYAML " + yaml.__version__
    else:
        raise DataError("unsupported_data_format", "HEPData requires submission-format JSON or YAML.")
    if not isinstance(data, dict):
        raise DataError("unsupported_data_format", "Expected one HEPData table or mapping object.")
    try:
        json.dumps(data, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise DataError("unsupported_data_format", "Non-JSON values in HEPData source.") from exc
    return data, version


def load_hepdata_mapping(path: Path) -> list[dict[str, Any]]:
    data, _ = decode_hepdata(_read_bytes(path), path.suffix)
    if data.get("schema_version") != "1" or not isinstance(data.get("mappings"), list):
        raise DataError("unsupported_data_format", "Mapping requires schema_version='1' and a mappings list.")
    seen = set()
    for entry in data["mappings"]:
        if not isinstance(entry, dict):
            raise DataError("unsupported_data_format", "Invalid mapping entry.")
        for key in ("paper_id", "arxiv_id", "figure_id", "latex_label", "panel_id", "record_id", "table_name", "source_file"):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                raise DataError("unsupported_data_format", f"Mapping {key} must be a string.")
        identity = json.dumps(entry, sort_keys=True)
        if identity in seen:
            raise DataError("ambiguous_mapping", "Duplicate HEPData mapping entry.")
        seen.add(identity)
    return data["mappings"]


def _published_value(value: Any) -> dict[str, Any]:
    raw = deepcopy(value)
    if value is None or value in ("", "-", "--"):
        return {"value": None, "state": "missing", "raw": raw}
    try:
        return {"value": number(value), "state": "numeric", "raw": raw}
    except DataError:
        return {"value": None, "state": "nonnumeric", "raw": raw}


def _published_error(value: Any) -> dict[str, Any]:
    percentage = isinstance(value, str) and value.strip().endswith("%")
    parsed = _published_value(value.strip()[:-1] if percentage else value)
    return {**parsed, "raw": deepcopy(value), "relative": percentage,
            "unit": "percent" if percentage else None}


def parse_hepdata_table(table: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain multidimensional coordinates and every published uncertainty component."""
    independent = table.get("independent_variables")
    dependent = table.get("dependent_variables")
    if not isinstance(independent, list) or not isinstance(dependent, list):
        raise DataError("unsupported_data_format", "Expected HEPData independent_variables and dependent_variables lists.")
    columns = independent + dependent
    for column in columns:
        if (not isinstance(column, dict) or not isinstance(column.get("header"), dict)
                or not isinstance(column.get("values"), list)
                or any(not isinstance(cell, dict) for cell in column["values"])):
            raise DataError("unsupported_data_format", "Malformed HEPData column.")
    if len({len(column["values"]) for column in columns}) > 1:
        raise DataError("invalid_numeric_value", "HEPData columns have different row counts.")
    axes = {"independent_variables": [deepcopy(c["header"]) for c in independent],
            "dependent_variables": [deepcopy(c["header"]) for c in dependent], "scales": None}
    coordinates = []
    warnings = []
    for column in independent:
        cells = []
        for cell in column["values"]:
            if ("low" in cell) != ("high" in cell):
                raise DataError("invalid_numeric_value", "HEPData bin must specify both limits.")
            cells.append({"value": _published_value(cell.get("value")),
                          "low": _published_value(cell.get("low")), "high": _published_value(cell.get("high")),
                          "raw": deepcopy(cell)})
        coordinates.append({"header": deepcopy(column["header"]), "values": cells})
    series = []
    for column in dependent:
        values = [_published_value(cell.get("value")) for cell in column["values"]]
        errors = []
        for row, cell in enumerate(column["values"]):
            components = []
            if not isinstance(cell.get("errors", []), list):
                raise DataError("unsupported_data_format", "HEPData errors must be a list.")
            for component in cell.get("errors", []):
                if not isinstance(component, dict):
                    raise DataError("unsupported_data_format", "Malformed uncertainty component.")
                if ("symerror" in component) == ("asymerror" in component):
                    raise DataError("unsupported_data_format", "Uncertainty must be symmetric or asymmetric.")
                label = component.get("label")
                kind = None
                if isinstance(label, str):
                    if re.match(r"^stat(?:istical)?\b", label, re.I):
                        kind = "statistical"
                    elif re.match(r"^sys(?:tematic)?\b", label, re.I):
                        kind = "systematic"
                parsed = {"label": label, "kind": kind, "raw": deepcopy(component)}
                if "symerror" in component:
                    parsed["symerror"] = _published_error(component["symerror"])
                else:
                    asymmetric = component["asymerror"]
                    if not isinstance(asymmetric, dict) or not {"plus", "minus"} <= asymmetric.keys():
                        raise DataError("unsupported_data_format", "Asymmetric error requires plus and minus values.")
                    parsed["asymerror"] = {key: _published_error(asymmetric[key]) for key in ("plus", "minus")}
                components.append(parsed)
            errors.append({"row": row, "components": components})
        if any(cell["state"] != "numeric" for cell in values):
            warnings.append(issue("missing_or_nonnumeric_published_value", "Published missing/textual values retained as explicit null states."))
        # Do not compute bin midpoints or project multi-dimensional measurements.
        x = [c["value"]["value"] for c in coordinates[0]["values"]] if len(coordinates) == 1 else None
        bins = ([{"low": c["low"]["value"], "high": c["high"]["value"], "raw": c["raw"]}
                 for c in coordinates[0]["values"]] if len(coordinates) == 1 else None)
        series.append({"label": column["header"].get("name"), "header": deepcopy(column["header"]),
                       "qualifiers": deepcopy(column.get("qualifiers", [])), "independent_variables": deepcopy(coordinates),
                       "x": x, "y": [v["value"] for v in values], "value_states": values, "bins": bins,
                       "uncertainties": errors})
    return axes, series, warnings


def _arxiv(value: str) -> str:
    # Version suffixes identify source revisions, not different publications.
    return re.sub(r"v\d+$", "", value.removeprefix("arXiv:").strip())


def recover_hepdata(*, figures: list[FigureRecord], panels: list[PanelRecord], papers: list[dict[str, Any]],
                    mappings: list[dict[str, Any]], root: Path, enable_network: bool = False,
                    include_generic: bool = False) -> tuple[list[PlotDataRecord], dict[str, Path]]:
    """Promote only explicit, unambiguous paper/figure/table mappings."""
    records = []
    inputs = {}
    paper_map = {p["paper_id"]: p for p in papers}
    eligible = [f for f in figures if include_generic or f.figure_type != "generic"]
    covered = set()
    for entry in mappings:
        candidates = [f for f in eligible if (not entry.get("figure_id") or f.figure_id == entry["figure_id"])
                      and (not entry.get("latex_label") or entry["latex_label"] in f.latex_labels
                           or entry["latex_label"] == f.latex_label)]
        if entry.get("paper_id"):
            candidates = [f for f in candidates if f.paper_id == entry["paper_id"]]
        if entry.get("arxiv_id"):
            candidates = [f for f in candidates if _arxiv(str(paper_map[f.paper_id].get("arxiv_id", ""))) == _arxiv(entry["arxiv_id"])]
        publication_known = bool(entry.get("paper_id") or entry.get("arxiv_id"))
        target_known = bool(entry.get("figure_id") or entry.get("latex_label"))
        if not candidates:
            state = "unmatched"
        elif not publication_known or not target_known:
            state = "candidate"
        elif len(candidates) != 1:
            state = "ambiguous"
        elif not entry.get("record_id") or not entry.get("table_name"):
            state = "candidate"
        else:
            state = "verified"
        figure = candidates[0] if len(candidates) == 1 else None
        panel = None
        if entry.get("panel_id"):
            matches = [p for p in panels if figure and p.figure_id == figure.figure_id and p.panel_id == entry["panel_id"]]
            if len(matches) == 1 and (include_generic or matches[0].panel_type != "generic"):
                panel = matches[0]
            else:
                state = "unmatched"
        paper_id = figure.paper_id if figure else entry.get("paper_id")
        identity = {key: entry.get(key) for key in ("record_id", "table_name", "source_file", "figure_id", "latex_label", "panel_id")}
        record = data_record(paper_id, figure, panel, "hepdata", identity,
                             {"state": state, "kind": "explicit_user_mapping", "mapping": deepcopy(entry),
                              "candidate_figure_ids": [f.figure_id for f in candidates]})
        if figure:
            covered.add(figure.figure_id)
        try:
            if state != "verified":
                raise DataError("hepdata_" + state, "Publication identity alone or ambiguous targets cannot verify a figure/table link.")
            if not re.fullmatch(r"(?:ins)?\d+", entry["record_id"]):
                record.association["state"] = "unmatched"
                raise DataError("invalid_hepdata_reference", "record_id must be a numeric HEPData ID or ins-prefixed INSPIRE ID.")
            if entry.get("source_file"):
                invalid = _literal_path(entry["source_file"])
                if invalid:
                    raise DataError(invalid, "Unsafe HEPData source path.")
                try:
                    path = local_path(root, root / entry["source_file"])
                except ValueError as exc:
                    raise DataError("unsafe_path", str(exc)) from exc
                content = _read_bytes(path)
                inputs[entry["source_file"]] = path
                origin = {"path": entry["source_file"], "sha256": hashlib.sha256(content).hexdigest()}
                record.provenance["sources"] = [origin]
                table, version = decode_hepdata(content, path.suffix)
            elif enable_network:
                content, url = fetch_hepdata(entry["record_id"], entry["table_name"])
                origin = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
                record.provenance["sources"] = [origin]
                record.raw_source["download_text"] = content.decode("utf-8")
                record.provenance["network_client"] = "requests " + requests.__version__
                table, version = decode_hepdata(content, ".yaml")
            else:
                raise DataError("network_disabled", "No local source_file; HEPData networking is disabled.")
            record.backend_version = version
            record.provenance["sources"] = [origin]
            record.raw_source["table"] = table
            if (table.get("arxiv_id") and _arxiv(str(table["arxiv_id"])) != _arxiv(str(paper_map[figure.paper_id].get("arxiv_id", "")))
                    or any(table.get(key) is not None and str(table[key]) != entry[key] for key in ("record_id", "table_name"))):
                record.association["state"] = "unmatched"
                raise DataError("hepdata_identity_mismatch", "Table metadata contradicts the explicit mapping.")
            axes, series, warnings = parse_hepdata_table(table)
            record.axes = axes
            record.series = [{**s, "series_id": f"{record.data_id}::series::{i + 1:02d}", "series_index": i}
                             for i, s in enumerate(series)]
            record.warnings.extend(warnings)
            if any(v is not None for s in series for v in s["y"]):
                record.source_method = "original_published_data"
                record.extraction_status = "partial" if warnings else "success"
                record.validation_status = "structural_checks_passed"
            else:
                record.warnings.append(issue("no_numeric_published_values", "Table contains no recoverable numeric dependent values."))
        except (ValueError, OSError) as exc:
            record.warnings.append(issue(exc.code if isinstance(exc, DataError) else "hepdata_input_error", str(exc)))
        records.append(record)
    for figure in eligible:
        if figure.figure_id not in covered:
            record = data_record(figure.paper_id, figure, None, "hepdata", "no_mapping",
                                 {"state": "unmatched", "kind": "no_explicit_mapping"})
            record.warnings = [issue("hepdata_unmatched", "No explicit HEPData table mapping for this figure.")]
            records.append(record)
    return records, inputs


def fetch_hepdata(record_id: str, table_name: str) -> tuple[bytes, str]:
    """Bounded opt-in download using the official record format/table endpoint."""
    if not re.fullmatch(r"(?:ins)?\d+", record_id) or not table_name or len(table_name) > 300:
        raise DataError("invalid_hepdata_reference", "Expected a numeric HEPData/ins record ID and explicit table name.")
    initial = "https://www.hepdata.net/record/" + record_id + "?" + urlencode({"format": "yaml", "table": table_name})
    last_error = ""
    for _ in range(3):
        url = initial
        try:
            for _ in range(4):
                parts = urlparse(url)
                if (parts.scheme != "https" or parts.hostname != "www.hepdata.net"
                        or parts.port not in {None, 443} or parts.username or parts.password):
                    raise DataError("unsafe_path", "HEPData redirect outside the allowed HTTPS host.")
                with requests.get(url, timeout=(10, 20), stream=True, allow_redirects=False) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location:
                            raise DataError("hepdata_backend_error", "Redirect has no Location.")
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise DataError("unsupported_data_format", "Received HTML instead of a HEPData table.")
                    length = response.headers.get("Content-Length")
                    if length and int(length) > MAX_SOURCE_BYTES:
                        raise DataError("source_too_large", "HEPData download exceeds the size limit.")
                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=65536):
                        content.extend(chunk)
                        if len(content) > MAX_SOURCE_BYTES:
                            raise DataError("source_too_large", "HEPData download exceeds the size limit.")
                    return bytes(content), url
            raise DataError("hepdata_backend_error", "Too many HEPData redirects.")
        except requests.RequestException as exc:
            last_error = str(exc)
    raise DataError("hepdata_backend_error", f"HEPData download failed after 3 attempts: {last_error}")


def compare_sources(records: list[PlotDataRecord]) -> list[dict[str, Any]]:
    """Compare only unique named 1D point series with explicit matching axes/units."""
    results = []
    published = [r for r in records if r.source_method == "original_published_data"]
    for native in (r for r in records if r.source_method == "original_source_data"):
        for series in native.series:
            candidates = []
            for other in published:
                if other.figure_id != native.figure_id or other.panel_id != native.panel_id:
                    continue
                independent = other.axes.get("independent_variables", [])
                for target in other.series:
                    header = target["header"]
                    if (not series.get("label") or series["label"] != target.get("label") or len(independent) != 1
                            or series.get("qualifiers") != target.get("qualifiers")
                            or any(not native.axes[axis].get(key) for axis in ("x", "y") for key in ("label", "unit"))):
                        continue
                    if (native.axes["x"]["label"] != independent[0].get("name")
                            or native.axes["x"]["unit"] != independent[0].get("units")
                            or native.axes["y"]["label"] != header.get("name")
                            or native.axes["y"]["unit"] != header.get("units")
                            or any(b["raw"].get("low") is not None or b["raw"].get("high") is not None for b in target["bins"] or [])):
                        continue
                    candidates.append((other, target))
            comparison: dict[str, Any] = {"native_data_id": native.data_id, "native_series_id": series["series_id"],
                                          "status": "not_comparable", "reason": "no_unique_explicit_series_and_axes_match"}
            same_named = [s for r in records if r.source_method == "original_source_data"
                          and r.figure_id == native.figure_id and r.panel_id == native.panel_id
                          for s in r.series if s.get("label") == series.get("label")]
            if len(candidates) == 1 and len(same_named) == 1:
                other, target = candidates[0]
                comparison.update(hepdata_data_id=other.data_id, hepdata_series_id=target["series_id"])
                raw_x = [str(c["value"]["raw"]) for c in target["independent_variables"][0]["values"]]
                if (len(series["x"]) != len(raw_x) or any(v is None for v in target["x"])
                        or any(Decimal(row[0]) != Decimal(x) for row, x in zip(series["raw_values"], raw_x))
                        or len(series["y"]) != len(target["y"])
                        or any(y is None for y in target["y"])):
                    comparison["reason"] = "different_coordinates_or_missing_values"
                else:
                    with localcontext() as context:
                        context.prec = max(50, max((len(str(s)) for s in series["raw_values"] + target["value_states"]), default=0) * 2)
                        differences = [Decimal(raw[1]) - Decimal(str(value["raw"]))
                                       for raw, value in zip(series["raw_values"], target["value_states"])]
                    comparison.update(status="compared", reason="exact_series_label_axes_units_and_coordinates",
                                      y_differences=[str(v) for v in differences],
                                      max_abs_y_difference=str(max(map(abs, differences), default=Decimal(0))),
                                      uncertainty_comparison="not_performed", scientific_agreement="manual_review_pending")
            results.append(comparison)
    return results
