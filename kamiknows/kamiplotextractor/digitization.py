"""Explicit source-coordinate calibration and optional visual extraction adapters."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kamiknows.common import sha256_file
from .assets import local_path

MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_POINTS = 100_000
BACKENDS = ("pymupdf", "plotdigitizer", "chartdetective", "plotsift")


class ExtractionError(ValueError):
    """Separate unsupported science/geometry from an execution failure."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numerical coordinate")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("coordinate must be finite")
    return result


def bounded_bytes(path: Path, root: Path) -> bytes:
    path = local_path(root, path)
    with path.open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("input exceeds 32 MiB limit")
    return content


def load_config(path: Path, key: str) -> list[dict]:
    obj = json.loads(bounded_bytes(path, path.parent))
    if not isinstance(obj, dict) or obj.get("schema_version") != "1" or not isinstance(obj.get(key), list):
        raise ValueError(f"expected schema_version '1' with {key} list")
    if any(not isinstance(entry, dict) for entry in obj[key]):
        raise ValueError(f"{key} entries must be objects")
    return obj[key]


@dataclass
class Axis:
    scale: str
    unit: str
    slope: float
    intercept: float
    residuals: list[float]

    def value(self, position: float) -> float:
        value = self.slope * finite(position) + self.intercept
        return finite(10 ** value if self.scale == "log10" else value)

    def position(self, value: float) -> float:
        value = finite(value)
        if self.scale == "log10":
            if value <= 0:
                raise ValueError("log-axis values must be positive")
            value = math.log10(value)
        return (value - self.intercept) / self.slope

    @classmethod
    def fit(cls, config: dict, interval: tuple[float, float]) -> Axis:
        scale = config.get("scale")
        if scale not in {"linear", "log10"} or not isinstance(config.get("unit"), str):
            raise ValueError("axis requires linear/log10 scale and explicit unit (empty string for dimensionless)")
        anchors = config.get("anchors", [])
        if not 2 <= len(anchors) <= 100:
            raise ValueError("axis requires 2–100 anchors")
        positions = [finite(a["position"]) for a in anchors]
        values = [finite(a["value"]) for a in anchors]
        if len(set(positions)) != len(positions) or min(positions) < interval[0] or max(positions) > interval[1]:
            raise ValueError("anchors must be distinct and within ROI")
        if scale == "log10":
            if min(values) <= 0:
                raise ValueError("log-axis anchors must be positive")
            values = [math.log10(v) for v in values]
        n = len(values)
        pmean, vmean = sum(positions) / n, sum(values) / n
        slope = sum((p - pmean) * (v - vmean) for p, v in zip(positions, values)) / sum((p - pmean)**2 for p in positions)
        if not math.isfinite(slope) or slope == 0:
            raise ValueError("zero/nonfinite calibration span")
        intercept = vmean - slope * pmean
        residuals = [v - (slope * p + intercept) for p, v in zip(positions, values)]
        tolerance = finite(config.get("max_residual", 1e-6))
        if tolerance < 0 or max(map(abs, residuals)) > tolerance:
            raise ValueError("inconsistent calibration anchors: residual exceeds max_residual")
        direction = "increasing" if slope > 0 else "decreasing"
        if config.get("direction", direction) != direction:
            raise ValueError("axis direction contradicts anchors")
        return cls(scale, config["unit"], slope, intercept, residuals)


class Calibration:
    """Coordinates refer to the original, unrotated source, top left origin, y down."""

    def __init__(self, config: dict):
        self.config = config
        roi = config.get("roi", [])
        if len(roi) != 4:
            raise ValueError("ROI requires [left, top, right, bottom]")
        self.roi = tuple(finite(v) for v in roi)
        l, t, r, b = self.roi
        if not (0 <= l < r and 0 <= t < b):
            raise ValueError("ROI must have positive spans and nonnegative coordinates")
        if config.get("coordinate_system") not in {"pdf_points", "pixels"}:
            raise ValueError("coordinate_system must be pdf_points or pixels")
        page = config.get("page", 1)
        if type(page) is not int or page < 1:
            raise ValueError("page must be a positive one-based integer")
        self.x = Axis.fit(config["x"], (l, r))
        self.y = Axis.fit(config["y"], (t, b))

    def bounds(self, width: float, height: float, system: str) -> None:
        if self.config["coordinate_system"] != system or self.roi[2] > width or self.roi[3] > height:
            raise ValueError("ROI/coordinate system does not match source dimensions")

    def contains(self, point: tuple) -> bool:
        x, y = map(finite, point)
        l, t, r, b = self.roi
        return l <= x <= r and t <= y <= b

    def convert(self, points: list) -> tuple[list, list]:
        if not points or len(points) > MAX_POINTS or any(not self.contains(p) for p in points):
            raise ExtractionError("needs_manual_review", "empty, excessive, or out-of-ROI selected geometry")
        return [self.x.value(p[0]) for p in points], [self.y.value(p[1]) for p in points]

    def metadata(self) -> dict:
        return {**self.config, "origin": "top_left", "y_direction": "down",
                "coordinate_reference": "original_unrotated_source",
                "fit": {name: {"slope": axis.slope, "intercept": axis.intercept,
                               "residuals": axis.residuals, "residual_unit": "log10(value)" if axis.scale == "log10" else axis.unit}
                        for name, axis in (("x", self.x), ("y", self.y))}}


def series_record(spec: dict, points: list, calibration: Calibration, geometry: str) -> dict:
    if not isinstance(spec.get("series_id"), str) or not spec["series_id"]:
        raise ValueError("explicit series_id required")
    x, y = calibration.convert(points)
    return {"series_id": spec["series_id"], "label": spec.get("label"),
            "observable": spec.get("observable"), "coordinate_definition": spec.get("coordinate_definition"),
            "units": {"x": calibration.x.unit, "y": calibration.y.unit}, "geometry": geometry,
            "x": x, "y": y, "source_coordinates": points,
            "value_states": ["visible" for _ in points], "displayed_uncertainties": [],
            "digitization_uncertainty": {"status": "not_estimated", "reason": "no independently calibrated localization error model"}}


def _polyline(path: dict) -> tuple[list, bool]:
    points = []
    sampled = False
    for item in path["items"]:
        op = item[0]
        if op not in {"l", "c"}:
            raise ExtractionError("unsupported", "line contains non-line/non-Bezier geometry")
        start = list(item[1])
        if points and math.dist(points[-1], start) > 1e-5:
            raise ExtractionError("needs_manual_review", "disconnected path cannot be treated as one curve")
        if not points:
            points.append(start)
        if op == "l":
            points.append(list(item[2]))
        else:
            sampled = True
            for i in range(1, 17):
                t = i / 16
                points.append([sum(w * p[d] for w, p in zip(
                    ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3), item[1:5])) for d in (0, 1)])
    return points, sampled


def _uncertainty(spec: dict) -> dict:
    semantic = spec.get("uncertainty_semantics", "unlabeled_displayed_error")
    if semantic not in {"unlabeled_displayed_error", "statistical", "systematic"}:
        raise ValueError("unsupported uncertainty semantics")
    evidence = spec.get("uncertainty_evidence")
    if semantic != "unlabeled_displayed_error" and (not isinstance(evidence, str) or not evidence.strip()):
        raise ValueError("statistical/systematic interpretation requires explicit textual evidence")
    return {"semantics": semantic, "evidence": evidence}


def vector_series(paths: list[dict], calibration: Calibration) -> list[dict]:
    """Only explicitly selected, distinct paths become data; axes are never guessed."""
    specs = calibration.config.get("series", [])
    if not specs:
        raise ExtractionError("needs_manual_review", "explicit series geometry and path_indices required; inspect raw_source.paths")
    results, used, names = [], set(), set()
    for spec in specs:
        indices = spec.get("path_indices", [])
        if (not indices or any(type(i) is not int or not 0 <= i < len(paths) for i in indices)
                or len(set(indices)) != len(indices) or used.intersection(indices)):
            raise ExtractionError("needs_manual_review", "invalid, duplicated or ambiguously shared path selection")
        used.update(indices)
        selected = [paths[i] for i in indices]
        geometry = spec.get("geometry")
        if spec.get("stacked") or geometry not in {"line", "scatter", "histogram", "band"}:
            raise ExtractionError("unsupported", "stacked/unknown geometry is unsupported")
        if geometry == "line":
            if len(selected) != 1:
                raise ExtractionError("needs_manual_review", "one connected path per line series required")
            points, sampled = _polyline(selected[0])
            result = series_record(spec, points, calibration, "rendered_curve_samples")
            result["sampling"] = {"kind": "bezier_16_segments" if sampled else "rendered_polyline_vertices",
                                  "original_measurement_points": False}
        elif geometry == "scatter":
            boxes = [p["rect"] for p in selected]
            max_size = finite(spec.get("max_marker_size", 12))
            if any(not (0 < b[2]-b[0] <= max_size and 0 < b[3]-b[1] <= max_size)
                   or not (p.get("closePath") or any(i[0] == "re" for i in p["items"]))
                   for b, p in zip(boxes, selected)):
                raise ExtractionError("needs_manual_review", "selected paths are not bounded closed markers")
            for path, box in zip(selected, boxes):
                rectangular = len(path["items"]) == 1 and path["items"][0][0] == "re"
                cubic = len(path["items"]) == 4 and all(item[0] == "c" for item in path["items"])
                if not rectangular and not cubic:
                    raise ExtractionError("unsupported", "markers must be rectangles or centrally symmetric four-Bezier shapes")
                if cubic:
                    control = [list(point) for item in path["items"] for point in item[1:]]
                    if any(not any(math.dist([box[0]+box[2]-p[0],box[1]+box[3]-p[1]],q) < 1e-4 for q in control) for p in control):
                        raise ExtractionError("needs_manual_review", "marker has no verified symmetry center")
                if not calibration.contains((box[0],box[1])) or not calibration.contains((box[2],box[3])):
                    raise ExtractionError("needs_manual_review", "marker crosses ROI; center may be clipped")
            if any(max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])
                   for i, a in enumerate(boxes) for b in boxes[i+1:]):
                raise ExtractionError("needs_manual_review", "overlapping marker boxes: hidden centers ambiguous; no coordinates emitted")
            points = [[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes]
            result = series_record(spec, points, calibration, "marker_centers")
        elif geometry == "histogram":
            if any(len(p["items"]) != 1 or p["items"][0][0] != "re" for p in selected):
                raise ExtractionError("unsupported", "histogram requires separate rectangular bars")
            baseline = finite(spec["baseline_position"])
            boxes = sorted([p["items"][0][1] for p in selected])
            if any(min(abs(b[1]-baseline), abs(b[3]-baseline)) > 1e-5 for b in boxes):
                raise ExtractionError("unsupported", "bars do not share explicit baseline; possible stack")
            if any(a[2] > b[0]+1e-5 for a, b in zip(boxes, boxes[1:])):
                raise ExtractionError("unsupported", "overlapping histogram bars")
            points = [[(b[0]+b[2])/2, b[1] if abs(b[3]-baseline) < 1e-5 else b[3]] for b in boxes]
            result = series_record(spec, points, calibration, "histogram")
            if any(not calibration.contains((b[d], b[e])) for b in boxes for d,e in ((0,1),(2,3))):
                raise ExtractionError("needs_manual_review", "bar boundaries outside ROI")
            result["bins"] = [sorted([calibration.x.value(b[0]), calibration.x.value(b[2])]) for b in boxes]
            result["baseline"] = calibration.y.value(baseline)
            result["visible_bars_only"] = True
        else:
            if len(selected) != 2:
                raise ExtractionError("unsupported", "band requires two explicitly selected envelope paths")
            first, sampled_a = _polyline(selected[0])
            second, sampled_b = _polyline(selected[1])
            if sampled_a or sampled_b or len(first) != len(second) or any(abs(a[0]-b[0]) > 1e-5 for a,b in zip(first, second)):
                raise ExtractionError("unsupported", "band envelopes must have matching explicit x vertices")
            x, ya = calibration.convert(first)
            _, yb = calibration.convert(second)
            result = series_record(spec, first, calibration, "band_envelopes")
            result["y"] = [None] * len(x)
            result["value_states"] = ["central_value_not_displayed"] * len(x)
            result["displayed_uncertainties"] = [{**_uncertainty(spec), "lower": min(a,b), "upper": max(a,b)} for a,b in zip(ya,yb)]
            result["source_envelopes"] = [first, second]
        if spec.get("error_path_indices"):
            if geometry != "scatter" or len(spec["error_path_indices"]) != len(result["x"]):
                raise ExtractionError("unsupported", "one explicitly paired vertical error segment per marker required")
            components = []
            for index, point in zip(spec["error_path_indices"], result["source_coordinates"]):
                if type(index) is not int or not 0 <= index < len(paths) or index in used:
                    raise ExtractionError("needs_manual_review", "ambiguous error segment selection")
                used.add(index)
                items = paths[index]["items"]
                if len(items) != 1 or items[0][0] != "l":
                    raise ExtractionError("unsupported", "error path must be one vertical segment without caps")
                ends = [list(items[0][1]), list(items[0][2])]
                if any(abs(p[0]-point[0]) > 1e-5 for p in ends) or not min(p[1] for p in ends) <= point[1] <= max(p[1] for p in ends):
                    raise ExtractionError("needs_manual_review", "error segment does not contain marker center")
                _, yy = calibration.convert(ends)
                components.append({**_uncertainty(spec), "lower": min(yy), "upper": max(yy), "source_endpoints": ends})
            result["displayed_uncertainties"] = components
        if result["series_id"] in names:
            raise ValueError("duplicate series_id")
        names.add(result["series_id"])
        results.append(result)
    return results


def read_pdf(path: Path, calibration: Calibration, dpi: int) -> tuple[list, dict, bytes]:
    import pymupdf

    with pymupdf.open(stream=bounded_bytes(path, path.parent), filetype="pdf") as document:
        if document.needs_pass or not 1 <= document.page_count <= 100:
            raise ExtractionError("unsupported", "encrypted PDF or page count outside 1–100")
        number = calibration.config.get("page", 1)
        if number > document.page_count:
            raise ValueError("selected page exceeds PDF page count")
        page = document[number - 1]
        rotation = page.rotation
        page.set_rotation(0)  # In-memory only; get_drawings uses unrotated coordinates.
        calibration.bounds(page.rect.width, page.rect.height, "pdf_points")
        if page.rect.width * page.rect.height * (dpi / 72)**2 > MAX_PIXELS:
            raise ExtractionError("unsupported", "PDF render exceeds pixel limit")
        raw = page.get_drawings()
        if len(raw) > 20_000 or sum(len(p["items"]) for p in raw) > MAX_POINTS:
            raise ExtractionError("unsupported", "PDF vector complexity limit exceeded")
        paths = []
        for p in raw:
            items = []
            for item in p["items"]:
                items.append([item[0], *[list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v for v in item[1:]]])
            paths.append({"items": items, "rect": list(p["rect"]), "color": p.get("color"),
                          "fill": p.get("fill"), "width": p.get("width"), "closePath": p.get("closePath", False)})
        metadata = {"paths": paths, "text_words": page.get_text("words")[:20_000],
                    "width": page.rect.width, "height": page.rect.height, "page": number,
                    "pdf_rotation": rotation, "cropbox": list(page.cropbox), "mediabox": list(page.mediabox),
                    "render_scale": dpi/72, "render_orientation": "unrotated", "dpi": dpi}
        preview = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72), alpha=False).tobytes("png")
    return paths, metadata, preview


def raster_image(path: Path, calibration: Calibration) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(bounded_bytes(path, path.parent))) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ExtractionError("unsupported", "image exceeds 16 million pixels")
        if source.getexif().get(274, 1) != 1:
            raise ExtractionError("needs_manual_review", "EXIF rotation requires a separately inventoried normalized source")
        calibration.bounds(source.width, source.height, "pixels")
        return source.convert("RGB")


def raster_series(path: Path, calibration: Calibration, work_dir: Path) -> tuple[list, dict]:
    specs = calibration.config.get("series", [])
    if len(specs) != 1 or specs[0].get("geometry") != "line" or not calibration.config.get("single_curve_evidence"):
        raise ExtractionError("unsupported", "plotdigitizer requires one line series and explicit single_curve_evidence")
    if any(specs[0].get(k) for k in ("stacked", "error_path_indices")):
        raise ExtractionError("unsupported", "raster errors/stacks unsupported")
    original = raster_image(path, calibration)
    l,t,r,b = calibration.roi
    box = (math.floor(l), math.floor(t), math.ceil(r), math.ceil(b))
    cropped = original.crop(box).convert("L")
    width,height = cropped.size
    if min(width,height) < 8:
        raise ExtractionError("unsupported", "raster ROI too small")
    prepared = work_dir / "preprocessed.png"
    cropped.save(prepared)
    # plotdigitizer 0.3.0 uses bottom-origin locations and space-delimited output.
    # Identity calibration yields bottom-origin pixel values; we convert afterward.
    command = [shutil.which("plotdigitizer"), str(prepared), "-p", "0,0", "-p", f"{width-1},0",
               "-p", f"0,{height-1}", "-l", "0,0", "-l", f"{width-1},0", "-l", f"0,{height-1}"]
    metadata = {"transforms": [{"operation": "crop", "box": list(box)}, {"operation": "grayscale", "mode": "L"}],
                "preprocessed_sha256": sha256_file(prepared), "width": original.width, "height": original.height,
                "render_scale": 1, "timeout_seconds": 45, "backend_axis_erasure_pixels": 3}
    with tempfile.TemporaryDirectory(prefix="kamiplot-") as tmp:
        temp = Path(tmp)
        output = temp / "trajectory.txt"
        command += ["--output", str(output)]
        # Log files avoid unbounded PIPE accumulation. Polling enforces aggregate file limits.
        with (temp / "stdout").open("wb") as stdout, (temp / "stderr").open("wb") as stderr:
            run_bounded(command, temp, stdout, stderr)
        metadata.update(stdout=(temp / "stdout").read_text(errors="replace")[:16_384],
                        stderr=(temp / "stderr").read_text(errors="replace")[:16_384], exit_code=0)
        rows = bounded_bytes(output, temp).decode("utf-8").splitlines()
        points = []
        for row in rows:
            values = row.split()
            if len(values) != 2:
                raise ExtractionError("backend_error", "invalid plotdigitizer output: expected two space-separated numbers")
            x,y = map(finite, values)
            points.append([x + box[0], height - y + box[1]])
        if len(points) != len({p[0] for p in points}):
            raise ExtractionError("backend_error", "duplicate trajectory x coordinates")
        result = series_record(specs[0], points, calibration, "rendered_curve_samples")
        result["sampling"] = {"kind": "raster_column_trajectory", "original_measurement_points": False}
    return [result], metadata


def run_bounded(command: list, directory: Path, stdout: Any, stderr: Any) -> None:
    import os
    import time

    def limits() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))

    with subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                          shell=False, preexec_fn=limits if os.name == "posix" else None) as process:
        started = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - started > 45 or sum(p.stat().st_size for p in directory.iterdir() if p.is_file()) > MAX_BYTES:
                process.kill()
                process.wait()
                raise ExtractionError("backend_error", "plotdigitizer timeout/output resource limit")
            time.sleep(0.05)
        if process.returncode:
            stderr.flush()
            message = (directory / "stderr").read_bytes()[:4096].decode(errors="replace")
            raise ExtractionError("backend_error", f"plotdigitizer exit {process.returncode}: {message}")


def import_table(root: Path, entry: dict, calibration: Calibration) -> tuple[dict, Path]:
    name = entry.get("file", "")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("import file must be a contained relative path")
    path = local_path(root, root / name)
    allowed = {".csv"} if entry["backend"] == "chartdetective" else {".csv", ".tsv", ".dat"}
    if path.suffix.lower() not in allowed:
        raise ValueError("unconfirmed import format")
    content = bounded_bytes(path, root)
    if hashlib.sha256(content).hexdigest() != entry.get("export_sha256"):
        raise ValueError("export SHA256 mismatch")
    if not isinstance(entry.get("backend_version"), str) or not entry["backend_version"]:
        raise ValueError("external application version/commit required")
    delimiter = entry.get("delimiter", ",")
    if delimiter not in {",", ";", "\t", "space"}:
        raise ValueError("unsupported delimiter")
    lines = content.decode("utf-8-sig").splitlines()
    if delimiter == "space":
        rows = [line.lstrip("# ").split() for line in lines if line.strip()]
    else:
        rows = list(csv.reader(lines, delimiter=delimiter))
    if entry.get("header", True):
        rows = rows[1:]
    columns = entry.get("columns", {})
    if not {"x", "y"} <= columns.keys() or any(type(i) is not int or i < 0 for i in columns.values()):
        raise ValueError("explicit zero-based x/y column indices required")
    if len(set(columns.values())) != len(columns):
        raise ValueError("ambiguous duplicate column mapping")
    if not rows or len(rows) > MAX_POINTS:
        raise ValueError("empty/excessive imported point count")
    def cell(row: list, key: str) -> float | None:
        index = columns.get(key)
        return None if index is None or index >= len(row) or not row[index].strip() else finite(row[index])
    points, physical, errors = [], [], []
    for row in rows:
        x,y = cell(row, "x"),cell(row, "y")
        if x is None or y is None:
            raise ValueError("missing imported coordinates; ambiguous rows cannot be mapped")
        points.append([calibration.x.position(x), calibration.y.position(y)])
        physical.append((x,y))
        error = cell(row, "y_error")
        lower,upper = cell(row, "y_lower"),cell(row, "y_upper")
        if error is not None and (error < 0 or lower is not None or upper is not None):
            raise ValueError("ambiguous/negative imported uncertainty")
        if error is not None:
            # PlotSift exports a magnitude; preserve it without inventing two observed endpoints.
            errors.append({**_uncertainty(entry), "magnitude": error, "lower": None, "upper": None,
                           "endpoint_state": "magnitude_only_export"})
        elif lower is not None or upper is not None:
            if lower is not None and lower > y or upper is not None and upper < y:
                raise ValueError("imported uncertainty endpoints do not bracket point")
            errors.append({**_uncertainty(entry), "lower": lower, "upper": upper})
        else:
            errors.append({"semantics": "unlabeled_displayed_error", "lower": None, "upper": None, "state": "not_exported"})
    result = series_record(entry, points, calibration, "externally_digitized_points")
    result["x"],result["y"] = map(list, zip(*physical))
    result["displayed_uncertainties"] = errors
    result["external_sheet"] = entry.get("sheet")
    return result, path


class Backend:
    """One interface; interactive tools expose import only, never fictional execution."""

    def __init__(self, name: str):
        if name not in BACKENDS:
            raise ValueError("unknown digitization backend")
        self.name = name

    def available(self) -> bool:
        if self.name in {"chartdetective", "plotsift"}:
            return True
        return importlib.util.find_spec(self.name) is not None and (self.name != "plotdigitizer" or shutil.which(self.name) is not None)

    def capabilities(self) -> dict:
        manual = self.name in {"chartdetective", "plotsift"}
        try:
            version = importlib.metadata.version(self.name) if not manual else "external_version_required_in_sidecar"
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {"backend_name": self.name, "version": version, "adapter_version": "1",
                "license": {"pymupdf": "AGPL-3.0-or-later OR commercial", "plotdigitizer": "LGPL-3.0-or-later",
                            "chartdetective": "BSD-2-Clause", "plotsift": "Apache-2.0"}[self.name],
                "available": self.available(), "execution": "import_only" if manual else "automatic_with_explicit_configuration",
                "formats": {"pymupdf": ["pdf"], "plotdigitizer": ["png", "jpg", "jpeg", "webp"],
                            "chartdetective": ["csv"], "plotsift": ["csv", "tsv", "dat"]}[self.name],
                "axis_scales": ["linear", "log10"], "calibration_provider": "shared_explicit_axis_layer",
                "multiple_series": self.name != "plotdigitizer", "uncertainties": self.name != "plotdigitizer",
                "limitations": "manual review pending; no tick inference; no hidden-point recovery; no stacked plots; "
                               + ("one grayscale trajectory; no experimental errors" if self.name == "plotdigitizer" else
                                  "explicit path selection or mapped external exports only")}

    def digitize(self, path: Path, calibration: Calibration, work_dir: Path, dpi: int = 144) -> tuple[list, dict]:
        if self.name in {"chartdetective", "plotsift"}:
            raise ExtractionError("needs_manual_review", "interactive backend requires mapped export")
        if not self.available():
            raise ExtractionError("backend_unavailable", f"optional {self.name} dependency/executable missing")
        if self.name == "plotdigitizer":
            return raster_series(path, calibration, work_dir)
        paths, metadata, preview = read_pdf(path, calibration, dpi)
        (work_dir / "source.png").write_bytes(preview)
        # Retain geometry even if subsequent reconstruction abstains.
        (work_dir / "geometry.json").write_text(json.dumps(metadata), encoding="utf-8")
        return vector_series(paths, calibration), metadata

    def import_results(self, root: Path, entry: dict, calibration: Calibration) -> tuple[dict, Path]:
        if self.name not in {"chartdetective", "plotsift"} or entry.get("backend") != self.name:
            raise ValueError("backend does not match external import")
        return import_table(root, entry, calibration)
