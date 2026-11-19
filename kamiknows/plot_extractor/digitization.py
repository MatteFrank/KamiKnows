"""Axis mappings and value extraction from native PDF geometry or calibrated pixels."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import local_path

MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_POINTS = 100_000
BACKENDS = ("pymupdf", "raster")


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
        return finite(10**value if self.scale == "log10" else value)

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
            raise ValueError(
                "axis requires linear/log10 scale and explicit unit (empty string for dimensionless)"
            )
        anchors = config.get("anchors", [])
        if not 2 <= len(anchors) <= 100:
            raise ValueError("axis requires 2–100 anchors")
        positions = [finite(a["position"]) for a in anchors]
        values = [finite(a["value"]) for a in anchors]
        if (
            len(set(positions)) != len(positions)
            or min(positions) < interval[0]
            or max(positions) > interval[1]
        ):
            raise ValueError("anchors must be distinct and within ROI")
        if scale == "log10":
            if min(values) <= 0:
                raise ValueError("log-axis anchors must be positive")
            values = [math.log10(v) for v in values]
        n = len(values)
        pmean, vmean = sum(positions) / n, sum(values) / n
        slope = sum((p - pmean) * (v - vmean) for p, v in zip(positions, values)) / sum(
            (p - pmean) ** 2 for p in positions
        )
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
        return {
            **self.config,
            "origin": "top_left",
            "y_direction": "down",
            "coordinate_reference": "original_unrotated_source",
            "fit": {
                name: {
                    "slope": axis.slope,
                    "intercept": axis.intercept,
                    "residuals": axis.residuals,
                    "residual_unit": "log10(value)" if axis.scale == "log10" else axis.unit,
                }
                for name, axis in (("x", self.x), ("y", self.y))
            },
        }


def series_record(spec: dict, points: list, calibration: Calibration, geometry: str) -> dict:
    if not isinstance(spec.get("series_id"), str) or not spec["series_id"]:
        raise ValueError("explicit series_id required")
    x, y = calibration.convert(points)
    return {
        "series_id": spec["series_id"],
        "label": spec.get("label"),
        "observable": spec.get("observable"),
        "coordinate_definition": spec.get("coordinate_definition"),
        "units": {"x": calibration.x.unit, "y": calibration.y.unit},
        "geometry": geometry,
        "x": x,
        "y": y,
        "source_coordinates": points,
        "value_states": ["visible" for _ in points],
        "displayed_uncertainties": [],
        "digitization_uncertainty": {
            "status": "not_estimated",
            "reason": "no independently calibrated localization error model",
        },
    }


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
                points.append(
                    [
                        sum(
                            w * p[d]
                            for w, p in zip(
                                ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t**3), item[1:5]
                            )
                        )
                        for d in (0, 1)
                    ]
                )
    return points, sampled


def _filled_outline(segments: list, max_drift: float) -> list:
    """Keep the way out of a filled histogram outline; the way back must be its horizontal baseline.

    A filled step histogram is drawn as one polygon that runs along the steps with x increasing and
    returns along a baseline. A returning edge at varying height (a band, an arbitrary shape) is refused.
    """
    turn = next((i for i, (a, b) in enumerate(segments) if b[0] - a[0] < -1e-5), len(segments))
    outbound, back = segments[:turn], segments[turn:]
    tolerance = max(1e-5, max_drift)
    if any(b[0] - a[0] > 1e-5 for a, b in back):
        raise ExtractionError("unsupported", "filled outline is not one step profile closed by a baseline")
    levels = [y for a, b in back if abs(b[0] - a[0]) > 1e-5 for y in (a[1], b[1])]
    steps = [y for a, b in outbound if b[0] - a[0] > 1e-5 for y in (a[1], b[1])]
    if levels and (max(levels) - min(levels) > tolerance or (steps and min(levels) < max(steps) - tolerance)):
        raise ExtractionError("unsupported", "filled outline returns at varying height: not a histogram fill")
    return outbound


def step_profile(path: dict, calibration: Calibration, max_drift: float) -> tuple[list, list]:
    """Displayed steps of a step-histogram path, stroked or filled.

    Plotting libraries break a stroked histogram into separate pieces where bins drop below the axis
    range, and close a filled one along a baseline outside the frame. Every piece must be axis-aligned
    and monotone in x, and later pieces may not go back in x. Steps drawn wholly below the frame are
    not displayed: their intervals are returned, never a value. A step above the frame or across its
    sides is refused.
    """
    if not path["items"] or any(item[0] != "l" for item in path["items"]):
        raise ExtractionError("unsupported", "step histogram requires straight segments only")
    segments = [(list(item[1]), list(item[2])) for item in path["items"]]
    if path.get("fill") is not None:
        segments = _filled_outline(segments, max_drift)
    reached = -math.inf
    for a, b in segments:
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx < -1e-5 or a[0] < reached - 1e-5 or (abs(dx) > 1e-5 and abs(dy) > max(1e-5, max_drift)):
            raise ExtractionError("unsupported", "step histogram must be monotone in source x and axis-aligned")
        reached = max(reached, b[0])
    left, top, right, bottom = calibration.roi
    visible, below = [], []
    for a, b in segments:
        if b[0] - a[0] <= 1e-5:
            continue
        if not (left <= a[0] <= right and left <= b[0] <= right):
            raise ExtractionError("needs_manual_review", "horizontal step crosses ROI; no clipped interval inferred")
        if calibration.contains(a) and calibration.contains(b):
            visible.append([a, b])
        elif min(a[1], b[1]) > bottom:
            below.append(sorted([calibration.x.value(a[0]), calibration.x.value(b[0])]))
        else:
            raise ExtractionError("needs_manual_review", "step above or across the plot frame; height not displayed")
    if not visible:
        raise ExtractionError("unsupported", "no visible horizontal steps")
    return visible, below


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
        raise ExtractionError(
            "needs_manual_review",
            "explicit series geometry and path_indices required; inspect raw_source.paths",
        )
    results, used, names = [], set(), set()
    for spec in specs:
        indices = spec.get("path_indices", [])
        if (
            not indices
            or any(type(i) is not int or not 0 <= i < len(paths) for i in indices)
            or len(set(indices)) != len(indices)
            or used.intersection(indices)
        ):
            raise ExtractionError(
                "needs_manual_review", "invalid, duplicated or ambiguously shared path selection"
            )
        used.update(indices)
        selected = [paths[i] for i in indices]
        geometry = spec.get("geometry")
        if spec.get("stacked") or geometry not in {"line", "scatter", "histogram", "step_histogram", "band"}:
            raise ExtractionError("unsupported", "stacked/unknown geometry is unsupported")
        if geometry == "step_histogram":
            if len(selected) != 1:
                raise ExtractionError("unsupported", "step histogram requires one path")
            max_drift = finite(spec.get("max_step_drift", 0))
            if max_drift < 0:
                raise ValueError("max_step_drift must be nonnegative")
            segments, below = step_profile(selected[0], calibration, max_drift)
            points = [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in segments]
            result = series_record(spec, points, calibration, "rendered_step_intervals")
            result["rendered_intervals"] = [
                sorted([calibration.x.value(a[0]), calibration.x.value(b[0])]) for a, b in segments
            ]
            result["source_horizontal_segments"] = segments
            result["rendered_height_ranges"] = [
                sorted([calibration.y.value(a[1]), calibration.y.value(b[1])]) for a, b in segments
            ]
            result["source_vertical_drifts"] = [abs(a[1] - b[1]) for a, b in segments]
            result["below_range_intervals"] = below
            result["original_bin_edges_verified"] = False
            result["sampling"] = {
                "kind": "visible_step_spans",
                "original_measurement_points": False,
                "max_step_drift_source_units": max_drift,
                "representative_position": "source_segment_midpoint",
                "limitations": "Equal-height original bins may share a plateau; vertical baseline closures are not data. No zero or hidden bins recovered; steps drawn below the plot frame are listed in below_range_intervals without values.",
            }
        elif geometry == "line":
            if len(selected) != 1:
                raise ExtractionError("needs_manual_review", "one connected path per line series required")
            points, sampled = _polyline(selected[0])
            result = series_record(spec, points, calibration, "rendered_curve_samples")
            result["sampling"] = {
                "kind": "bezier_16_segments" if sampled else "rendered_polyline_vertices",
                "original_measurement_points": False,
            }
        elif geometry == "scatter":
            boxes = [p["rect"] for p in selected]
            max_size = finite(spec.get("max_marker_size", 12))
            if any(
                not (0 < b[2] - b[0] <= max_size and 0 < b[3] - b[1] <= max_size)
                or not (p.get("closePath") or any(i[0] == "re" for i in p["items"]))
                for b, p in zip(boxes, selected)
            ):
                raise ExtractionError("needs_manual_review", "selected paths are not bounded closed markers")
            for path, box in zip(selected, boxes):
                rectangular = len(path["items"]) == 1 and path["items"][0][0] == "re"
                cubic = len(path["items"]) == 4 and all(item[0] == "c" for item in path["items"])
                if not rectangular and not cubic:
                    raise ExtractionError(
                        "unsupported", "markers must be rectangles or centrally symmetric four-Bezier shapes"
                    )
                if cubic:
                    control = [list(point) for item in path["items"] for point in item[1:]]
                    if any(
                        not any(
                            math.dist([box[0] + box[2] - p[0], box[1] + box[3] - p[1]], q) < 1e-4
                            for q in control
                        )
                        for p in control
                    ):
                        raise ExtractionError("needs_manual_review", "marker has no verified symmetry center")
                if not calibration.contains((box[0], box[1])) or not calibration.contains((box[2], box[3])):
                    raise ExtractionError("needs_manual_review", "marker crosses ROI; center may be clipped")
            if any(
                max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])
                for i, a in enumerate(boxes)
                for b in boxes[i + 1 :]
            ):
                raise ExtractionError(
                    "needs_manual_review",
                    "overlapping marker boxes: hidden centers ambiguous; no coordinates emitted",
                )
            points = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
            result = series_record(spec, points, calibration, "marker_centers")
        elif geometry == "histogram":
            if any(len(p["items"]) != 1 or p["items"][0][0] != "re" for p in selected):
                raise ExtractionError("unsupported", "histogram requires separate rectangular bars")
            baseline = finite(spec["baseline_position"])
            boxes = sorted([p["items"][0][1] for p in selected])
            if any(min(abs(b[1] - baseline), abs(b[3] - baseline)) > 1e-5 for b in boxes):
                raise ExtractionError("unsupported", "bars do not share explicit baseline; possible stack")
            if any(a[2] > b[0] + 1e-5 for a, b in zip(boxes, boxes[1:])):
                raise ExtractionError("unsupported", "overlapping histogram bars")
            points = [[(b[0] + b[2]) / 2, b[1] if abs(b[3] - baseline) < 1e-5 else b[3]] for b in boxes]
            result = series_record(spec, points, calibration, "histogram")
            if any(not calibration.contains((b[d], b[e])) for b in boxes for d, e in ((0, 1), (2, 3))):
                raise ExtractionError("needs_manual_review", "bar boundaries outside ROI")
            result["bins"] = [sorted([calibration.x.value(b[0]), calibration.x.value(b[2])]) for b in boxes]
            result["baseline"] = calibration.y.value(baseline)
            result["visible_bars_only"] = True
        else:
            if len(selected) != 2:
                raise ExtractionError("unsupported", "band requires two explicitly selected envelope paths")
            first, sampled_a = _polyline(selected[0])
            second, sampled_b = _polyline(selected[1])
            if (
                sampled_a
                or sampled_b
                or len(first) != len(second)
                or any(abs(a[0] - b[0]) > 1e-5 for a, b in zip(first, second))
            ):
                raise ExtractionError("unsupported", "band envelopes must have matching explicit x vertices")
            x, ya = calibration.convert(first)
            _, yb = calibration.convert(second)
            result = series_record(spec, first, calibration, "band_envelopes")
            result["y"] = [None] * len(x)
            result["value_states"] = ["central_value_not_displayed"] * len(x)
            result["displayed_uncertainties"] = [
                {**_uncertainty(spec), "lower": min(a, b), "upper": max(a, b)} for a, b in zip(ya, yb)
            ]
            result["source_envelopes"] = [first, second]
        if spec.get("error_path_indices"):
            if geometry != "scatter" or len(spec["error_path_indices"]) != len(result["x"]):
                raise ExtractionError(
                    "unsupported", "one explicitly paired vertical error segment per marker required"
                )
            components = []
            for index, point in zip(spec["error_path_indices"], result["source_coordinates"]):
                if type(index) is not int or not 0 <= index < len(paths) or index in used:
                    raise ExtractionError("needs_manual_review", "ambiguous error segment selection")
                used.add(index)
                items = paths[index]["items"]
                if len(items) != 1 or items[0][0] != "l":
                    raise ExtractionError(
                        "unsupported", "error path must be one vertical segment without caps"
                    )
                ends = [list(items[0][1]), list(items[0][2])]
                if any(abs(p[0] - point[0]) > 1e-5 for p in ends) or not min(p[1] for p in ends) <= point[
                    1
                ] <= max(p[1] for p in ends):
                    raise ExtractionError(
                        "needs_manual_review", "error segment does not contain marker center"
                    )
                _, yy = calibration.convert(ends)
                components.append(
                    {**_uncertainty(spec), "lower": min(yy), "upper": max(yy), "source_endpoints": ends}
                )
            result["displayed_uncertainties"] = components
        if result["series_id"] in names:
            raise ValueError("duplicate series_id")
        names.add(result["series_id"])
        results.append(result)
    return results


def prepare_pdf_orientation(page: Any, orientation: str = "unrotated") -> None:
    """Normalize only in memory; preserve the displayed crop for rotated proposals."""
    import pymupdf

    if orientation not in {"unrotated", "display"}:
        raise ValueError("unsupported PDF orientation")
    if orientation == "display" and page.rotation:
        crop = page.cropbox
        media = page.mediabox
        page.set_mediabox(pymupdf.Rect(crop.x0, media.y1 - crop.y1, crop.x1, media.y1 - crop.y0))
        page.remove_rotation()
    else:
        page.set_rotation(0)


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
        prepare_pdf_orientation(page, calibration.config.get("pdf_orientation", "unrotated"))
        calibration.bounds(page.rect.width, page.rect.height, "pdf_points")
        if page.rect.width * page.rect.height * (dpi / 72) ** 2 > MAX_PIXELS:
            raise ExtractionError("unsupported", "PDF render exceeds pixel limit")
        raw = page.get_drawings()
        if len(raw) > 20_000 or sum(len(p["items"]) for p in raw) > MAX_POINTS:
            raise ExtractionError("unsupported", "PDF vector complexity limit exceeded")
        paths = []
        for p in raw:
            items = []
            for item in p["items"]:
                items.append(
                    [
                        item[0],
                        *[
                            list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
                            for v in item[1:]
                        ],
                    ]
                )
            paths.append(
                {
                    "items": items,
                    "rect": list(p["rect"]),
                    "color": p.get("color"),
                    "fill": p.get("fill"),
                    "width": p.get("width"),
                    "closePath": p.get("closePath", False),
                }
            )
        metadata = {
            "paths": paths,
            "text_words": page.get_text("words")[:20_000],
            "width": page.rect.width,
            "height": page.rect.height,
            "page": number,
            "pdf_rotation": rotation,
            "cropbox": list(page.cropbox),
            "mediabox": list(page.mediabox),
            "render_scale": dpi / 72,
            "render_orientation": calibration.config.get("pdf_orientation", "unrotated"),
            "dpi": dpi,
        }
        preview = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False).tobytes("png")
    return paths, metadata, preview


def raster_image(path: Path, calibration: Calibration) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(bounded_bytes(path, path.parent))) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ExtractionError("unsupported", "image exceeds 16 million pixels")
        if source.getexif().get(274, 1) != 1:
            raise ExtractionError(
                "needs_manual_review", "EXIF rotation requires a separately inventoried normalized source"
            )
        calibration.bounds(source.width, source.height, "pixels")
        return source.convert("RGB")


class Backend:
    """Native PDF geometry or calibrated pixel extraction behind one interface."""

    def __init__(self, name: str):
        if name not in BACKENDS:
            raise ValueError("unknown digitization backend")
        self.name = name

    def available(self) -> bool:
        if self.name == "raster":
            return all(importlib.util.find_spec(name) is not None for name in ["cv2", "PIL", "numpy"])
        return importlib.util.find_spec(self.name) is not None

    def capabilities(self) -> dict:
        try:
            version = importlib.metadata.version(self.name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        if self.name == "raster":
            return {
                "backend_name": "raster",
                "version": "internal_raster_v1",
                "adapter_version": "1",
                "license": "project code; dependency licenses apply",
                "available": self.available(),
                "execution": "automatic_with_calibration_recipe",
                "formats": ["pdf", "png", "jpg", "jpeg", "webp"],
                "axis_scales": ["linear", "log10"],
                "calibration_provider": "shared_axis_layer_with_ocr_fallback",
                "multiple_series": True,
                "uncertainties": False,
                "limitations": "prototype: separable connected colored traces or constant-color blocks; no broken axes, texture or original-bin recovery",
            }
        return {
            "backend_name": "pymupdf",
            "version": version,
            "adapter_version": "1",
            "license": "AGPL-3.0-or-later OR commercial",
            "available": self.available(),
            "execution": "automatic_with_explicit_configuration",
            "formats": ["pdf"],
            "axis_scales": ["linear", "log10"],
            "calibration_provider": "shared_explicit_axis_layer",
            "multiple_series": True,
            "uncertainties": True,
            "limitations": "manual review pending; no tick inference; no hidden-point recovery; no stacked plots; "
            "explicit path selection only",
        }

    def digitize(
        self, path: Path, calibration: Calibration, work_dir: Path, dpi: int = 144
    ) -> tuple[list, dict]:
        if not self.available():
            raise ExtractionError(
                "backend_unavailable", f"optional {self.name} dependency/executable missing"
            )
        if self.name == "raster":
            return pixel_digitization(path, calibration, work_dir, dpi)
        paths, metadata, preview = read_pdf(path, calibration, dpi)
        (work_dir / "source.png").write_bytes(preview)
        # Retain geometry even if subsequent reconstruction abstains.
        (work_dir / "geometry.json").write_text(json.dumps(metadata), encoding="utf-8")
        if calibration.config.get("panels_1d"):
            raise ExtractionError(
                "needs_manual_review",
                "per-panel 1D digitization is required; do not use the first panel mapping for the whole asset",
            )
        try:
            if calibration.config.get("histogram2d_panels"):
                return histogram2d_series(path, calibration.config), metadata
            return vector_series(paths, calibration), metadata
        except ExtractionError as exc:
            if exc.status != "unsupported" or not calibration.config.get("automatic_raster_fallback"):
                raise
            fallback = {
                **calibration.config,
                "raster_extraction": {
                    "kind": (
                        "histogram_2d" if calibration.config.get("histogram2d_panels") else "colored_traces"
                    ),
                    "method": "pixel_geometry_v1",
                },
            }
            series, info = pixel_digitization(path, Calibration(fallback), work_dir, dpi)
            info.update(
                backend_requested="pymupdf",
                backend_used="raster",
                native_fallback_reason=str(exc),
                effective_calibration=fallback,
            )
            return series, info


def pdf_cell_geometry(page: Any, roi: list[float]) -> tuple[list[dict], dict]:
    """Read visible rectangular cells; do not equate image pixels with original bins."""
    import pymupdf

    l, t, r, b = roi
    area = (r - l) * (b - t)
    cells = []
    for index, p in enumerate(page.get_drawings()):
        if p.get("fill") is None or len(p["items"]) != 1 or p["items"][0][0] != "re":
            continue
        box = list(p["items"][0][1])
        x0, y0, x1, y1 = box
        if not (l - 0.01 <= x0 < x1 <= r + 0.01 and t - 0.01 <= y0 < y1 <= b + 0.01):
            continue
        if (x1 - x0) * (y1 - y0) >= area * 0.5:
            continue
        if p.get("fill_opacity", 1) != 1:
            raise ExtractionError("needs_manual_review", "transparent vector cells")
        cells.append({"rect": box, "color": list(p["fill"]), "source_path": index})
    if len(cells) >= 4:
        ordered = sorted(cells, key=lambda c: c["rect"][0])
        for i, a in enumerate(ordered):
            for other in ordered[i + 1 :]:
                if other["rect"][0] >= a["rect"][2] - 0.01:
                    break
                if min(a["rect"][3], other["rect"][3]) - max(a["rect"][1], other["rect"][1]) > 0.01:
                    raise ExtractionError("needs_manual_review", "overlapping cell geometry")
        return cells, {"kind": "vector_rectangles", "original_bin_edges_verified": False}
    # Embedded nearest-neighbour heatmaps may contain uniform rectangular blocks.
    images = [
        im for im in page.get_image_info(xrefs=True) if all(abs(a - b) < 3 for a, b in zip(im["bbox"], roi))
    ]
    if len(images) != 1:
        raise ExtractionError("needs_manual_review", "no_unique_rectangular_cell_grid")
    im = images[0]
    transform = im.get("transform", (1, 0, 0, 1, 0, 0))
    if abs(transform[1]) > 1e-6 or abs(transform[2]) > 1e-6 or transform[0] <= 0 or transform[3] <= 0:
        raise ExtractionError("unsupported", "rotated_or_skewed_raster_grid")
    info = page.parent.extract_image(im["xref"])
    pix = pymupdf.Pixmap(page.parent, im["xref"])
    if pix.n != 3 or pix.width * pix.height > 4_000_000:
        raise ExtractionError("unsupported", "unsupported raster grid size/colors")
    import numpy as np

    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    if info.get("smask", 0):
        mask = pymupdf.Pixmap(page.parent, info["smask"])
        if mask.n != 1 or (mask.width, mask.height) != (pix.width, pix.height):
            raise ExtractionError("unsupported", "unsupported raster alpha mask")
        alpha = np.frombuffer(mask.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        if np.any((alpha != 0) & (alpha != 255)):
            raise ExtractionError("needs_manual_review", "partially_transparent_cell_colors")
    grid = np.concatenate([np.where(alpha[:, :, None] > 0, rgb, 0), alpha[:, :, None]], axis=2)
    xe = [0, *list(np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=(0, 2))) + 1), pix.width]
    ye = [0, *list(np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=(1, 2))) + 1), pix.height]
    if (
        not 2 <= len(xe) - 1 <= 512
        or not 2 <= len(ye) - 1 <= 512
        or min(np.diff(xe)) < 2
        or min(np.diff(ye)) < 2
    ):
        raise ExtractionError("needs_manual_review", "raster_has_no_unambiguous_constant_cell_blocks")
    il, it, ir, ib = im["bbox"]
    clipped = 0
    masked = 0
    for y0, y1 in zip(ye, ye[1:]):
        for x0, x1 in zip(xe, xe[1:]):
            if alpha[y0, x0] == 0:
                masked += 1
                continue
            raw = [
                il + x0 / pix.width * (ir - il),
                it + y0 / pix.height * (ib - it),
                il + x1 / pix.width * (ir - il),
                it + y1 / pix.height * (ib - it),
            ]
            rect = [max(l, raw[0]), max(t, raw[1]), min(r, raw[2]), min(b, raw[3])]
            if rect[0] >= rect[2] or rect[1] >= rect[3]:
                continue
            clipped += int(rect != raw)
            cells.append({"rect": rect, "color": (rgb[y0, x0].astype(float) / 255).tolist()})
    if not cells:
        raise ExtractionError("needs_manual_review", "no_visible_raster_cells")
    return cells, {
        "masked_blocks_omitted": masked,
        "alpha_policy": "opaque_only_transparent_is_missing",
        "kind": "rendered_raster_blocks",
        "xref": im["xref"],
        "edge_blocks_clipped": clipped,
        "original_bin_edges_verified": False,
    }


def colorbar_palette(page: Any, roi: list[float]) -> list[dict]:
    """Sample the displayed vertical colorbar; preserve quantized value ranges."""
    import pymupdf

    l, t, r, b = roi
    if r - l < 2 or b - t < 20:
        raise ExtractionError("needs_manual_review", "colorbar_too_small")
    pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(roi), alpha=False)
    samples = []
    for j in range(pix.height):
        color = list(pix.pixel(pix.width // 2, j)[:3])
        position = t + (j + 0.5) / pix.height * (b - t)
        samples.append(
            {
                "position": position,
                "color": [v / 255 for v in color],
                "position_range": [t + j / pix.height * (b - t), t + (j + 1) / pix.height * (b - t)],
            }
        )
    if len({tuple(s["color"]) for s in samples}) < 4:
        raise ExtractionError("needs_manual_review", "colorbar_not_quantitatively_resolved")
    return samples


def histogram2d_series(path: Path, config: dict) -> list[dict]:
    import pymupdf

    results = []
    with pymupdf.open(stream=bounded_bytes(path, path.parent), filetype="pdf") as document:
        page = document[config.get("page", 1) - 1]
        prepare_pdf_orientation(page, config.get("pdf_orientation", "unrotated"))
        for index, panel in enumerate(config["histogram2d_panels"], 1):
            roi = panel["roi"]
            cal = Calibration({**panel, "coordinate_system": "pdf_points", "page": 1})
            bar = panel["colorbar_roi"]
            zaxis = Axis.fit(panel["z"], (bar[1], bar[3]))
            if not page.rect.contains(pymupdf.Rect(bar)):
                raise ExtractionError("needs_manual_review", "colorbar_outside_page")
            cells, source = pdf_cell_geometry(page, roi)
            palette = colorbar_palette(page, bar)
            series = {
                "series_id": f"panel_{index}_cells",
                "label": None,
                "observable": "unreviewed_2d_displayed_values",
                "coordinate_definition": "visible cell/block bounds with colorbar-inverted Z estimates",
                "geometry": "histogram_2d_cells",
                "units": {"x": cal.x.unit, "y": cal.y.unit, "z": zaxis.unit},
                "x": [],
                "y": [],
                "z": [],
                "z_ranges": [],
                "cell_bounds": [],
                "source_cell_rects": [],
                "source_coordinates": [],
                "color_distances": [],
                "displayed_uncertainties": [],
                "original_bin_edges_verified": False,
                "cell_source": source,
                "axis_calibration": panel,
                "palette_matching_policy": {
                    "max_rgb_distance": 0.025,
                    "max_disconnected_local_span_fraction": 0.02,
                },
                "value_states": [],
                "digitization_uncertainty": {
                    "status": "not_estimated",
                    "reason": "color quantization ranges are not experimental uncertainties",
                },
            }
            color_cache = {}
            for cell in cells:
                color_key = tuple(cell["color"])
                if color_key not in color_cache:
                    color_cache[color_key] = [math.dist(cell["color"], s["color"]) for s in palette]
                distances = color_cache[color_key]
                best = min(distances)
                if best > 0.025:
                    raise ExtractionError("needs_manual_review", "cell_color_not_on_colorbar")
                matches = [i for i, d in enumerate(distances) if d <= best + 1e-5]
                if max(matches) - min(matches) + 1 != len(matches) and palette[max(matches)][
                    "position"
                ] - palette[min(matches)]["position"] > 0.02 * (bar[3] - bar[1]):
                    raise ExtractionError("needs_manual_review", "non_unique_colorbar_inverse")
                low = palette[min(matches)]["position_range"][0]
                high = palette[max(matches)]["position_range"][1]
                a, b, c, d = cell["rect"]
                xb = sorted([cal.x.value(a), cal.x.value(c)])
                yb = sorted([cal.y.value(b), cal.y.value(d)])
                series["x"].append(cal.x.value((a + c) / 2))
                series["y"].append(cal.y.value((b + d) / 2))
                series["z"].append(zaxis.value((low + high) / 2))
                series["z_ranges"].append(sorted([zaxis.value(low), zaxis.value(high)]))
                series["cell_bounds"].append({"x": xb, "y": yb})
                series["source_cell_rects"].append(cell["rect"])
                series["source_coordinates"].append([(a + c) / 2, (b + d) / 2])
                series["color_distances"].append(best)
                series["value_states"].append(
                    "colorbar_endpoint_saturation_possible"
                    if min(matches) == 0 or max(matches) == len(palette) - 1
                    else "color_inverted_estimate"
                )
            results.append(series)
    return results


def inspect_representation(path: Path) -> dict:
    """Inspect PDF contents rather than equating the extension with vector geometry."""
    if path.suffix.lower() != ".pdf":
        return {"container": path.suffix.lower().lstrip("."), "representation": "raster"}
    import pymupdf

    with pymupdf.open(path) as doc:
        page = doc[0]
        paths = len(page.get_drawings())
        images = len(page.get_images())
        return {
            "container": "pdf",
            "pages": doc.page_count,
            "vector_paths": paths,
            "embedded_images": images,
            "native_text_available": bool(page.get_text().strip()),
            "representation": (
                "hybrid"
                if images and paths
                else "raster" if images else "vector" if paths else "text_or_empty"
            ),
        }


def _pixel_source(path: Path, calibration: Calibration, dpi: int) -> tuple[Any, float]:
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            if doc.page_count != 1:
                raise ExtractionError("unsupported", "raster fallback requires a single PDF page")
            page = doc[0]
            prepare_pdf_orientation(page, "display")
            calibration.bounds(page.rect.width, page.rect.height, "pdf_points")
            scale = dpi / 72
            if page.rect.width * page.rect.height * scale * scale > MAX_PIXELS:
                raise ExtractionError("unsupported", "raster rendering exceeds pixel limit")
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples), scale
    return raster_image(path, calibration), 1.0


def _pixel_traces(image: Any, calibration: Calibration, scale: float) -> list[dict]:
    """Extract long, thin, connected colored traces; ambiguous/filled components abstain."""
    import cv2
    import numpy as np

    rgb = np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    l, t, r, b = [int(round(v * scale)) for v in calibration.roi]
    margin = 2
    crop = hsv[t + margin : b - margin, l + margin : r - margin]
    if crop.size == 0:
        raise ExtractionError("needs_manual_review", "raster plotting area is empty")
    colored = (crop[:, :, 1] >= 80) & (crop[:, :, 2] >= 60)
    bins = crop[:, :, 0] // 10
    results = []
    rejected = 0
    for hue in np.unique(bins[colored]):
        mask = ((bins == hue) & colored).astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if width < (r - l) * 0.35 or area < 20:
                continue
            points = []
            ambiguous = 0
            mask_component = labels[y : y + height, x : x + width] == label
            for column in range(width):
                hits = np.flatnonzero(mask_component[:, column])
                if not len(hits):
                    continue
                if np.ptp(hits) > max(8, 4 * scale):
                    ambiguous += 1
                    continue
                points.append(
                    [
                        (l + margin + x + column + 0.5) / scale,
                        (t + margin + y + float(np.median(hits)) + 0.5) / scale,
                    ]
                )
            if len(points) < max(10, width * 0.7) or ambiguous > width * 0.2:
                rejected += 1
                continue
            spec = {
                "series_id": f"color_{int(hue)}_component_{label}",
                "label": None,
                "observable": "unreviewed_displayed_raster_trace",
                "coordinate_definition": "pixel trace centers mapped through automatically fitted axes",
            }
            result = series_record(spec, points, calibration, "rendered_curve_samples")
            result["sampling"] = {"kind": "connected_color_pixels", "original_measurement_points": False}
            result["raster_color_rgb"] = (
                np.median(
                    rgb[t + margin + y : t + margin + y + height, l + margin + x : l + margin + x + width][
                        mask_component
                    ],
                    axis=0,
                )
                / 255
            ).tolist()
            result["digitization_uncertainty"] = {
                "status": "not_estimated",
                "reason": "pixel samples do not establish experimental uncertainty",
            }
            results.append(result)
    if not results:
        raise ExtractionError(
            "needs_manual_review",
            "no long thin separable colored traces; filled, dashed or overlapping geometry may require another method",
        )
    if rejected:
        raise ExtractionError(
            "needs_manual_review",
            "some long colored components are ambiguous; do not silently publish only easy traces",
        )
    return results


def _pixel_cells(image: Any, config: dict, scale: float) -> list[dict]:
    """Recover displayed constant-color blocks, never claim original bin boundaries."""
    import numpy as np

    rgb = np.asarray(image, dtype=float) / 255
    results = []
    for index, panel in enumerate(config["histogram2d_panels"], 1):
        cal = Calibration({**panel, "coordinate_system": config["coordinate_system"], "page": 1})
        cal.bounds(image.width / scale, image.height / scale, config["coordinate_system"])
        l, t, r, b = [int(round(v * scale)) for v in panel["roi"]]
        l += 2
        t += 2
        r -= 2
        b -= 2
        crop = rgb[t:b, l:r]
        if crop.size == 0:
            raise ExtractionError("needs_manual_review", "raster cell area is empty")
        # Consecutive identical scanlines define visible blocks; smooth imagery abstains.
        split = np.flatnonzero(np.any(np.abs(np.diff(crop, axis=0)) > 1 / 255, axis=(1, 2))) + 1
        edges = np.r_[0, split, crop.shape[0]]
        blocks = []
        for top, bottom in zip(edges[:-1], edges[1:]):
            if bottom - top < 2:
                continue
            row = np.median(crop[top:bottom], axis=0)
            xs = np.r_[
                0, np.flatnonzero(np.any(np.abs(np.diff(row, axis=0)) > 1 / 255, axis=1)) + 1, len(row)
            ]
            for left, right in zip(xs[:-1], xs[1:]):
                if right - left < 2:
                    continue
                color = np.median(row[left:right], axis=0)
                blocks.append(
                    (
                        [
                            float((l + left) / scale),
                            float((t + top) / scale),
                            float((l + right) / scale),
                            float((t + bottom) / scale),
                        ],
                        color,
                    )
                )
                if len(blocks) > 5000:
                    raise ExtractionError("needs_manual_review", "raster block complexity limit exceeded")
        covered = sum((c - a) * (d - bb) * scale * scale for (a, bb, c, d), color in blocks)
        if not blocks or len(blocks) > 5000 or covered < crop.shape[0] * crop.shape[1] * 0.8:
            raise ExtractionError(
                "needs_manual_review",
                "no sufficiently complete constant-color raster block layout; interpolation or texture is unsupported",
            )
        cl, ct, cr, cb = panel["colorbar_roi"]
        zaxis = Axis.fit(panel["z"], (ct, cb))
        if not (0 <= cl < cr <= image.width / scale and 0 <= ct < cb <= image.height / scale):
            raise ExtractionError("needs_manual_review", "raster colorbar is outside source")
        px0 = int((cl + (cr - cl) * 0.3) * scale)
        px1 = max(px0 + 1, int((cl + (cr - cl) * 0.7) * scale))
        py0 = int(ct * scale) + 1
        py1 = int(cb * scale) - 1
        palette = np.median(rgb[py0:py1, px0:px1], axis=1)
        if not len(palette):
            raise ExtractionError("needs_manual_review", "raster colorbar is too small")
        series = {
            "series_id": f"panel_{index}_pixel_blocks",
            "label": None,
            "observable": "unreviewed_displayed_raster_blocks",
            "coordinate_definition": "visible pixel blocks with colorbar-inverted Z estimates",
            "geometry": "histogram_2d_cells",
            "units": {"x": cal.x.unit, "y": cal.y.unit, "z": zaxis.unit},
            "x": [],
            "y": [],
            "z": [],
            "z_ranges": [],
            "cell_bounds": [],
            "source_cell_rects": [],
            "source_coordinates": [],
            "color_distances": [],
            "value_states": [],
            "displayed_uncertainties": [],
            "original_bin_edges_verified": False,
            "cell_source": {
                "kind": "detected_raster_blocks",
                "original_grid_verified": False,
                "retained_area_fraction": covered / (crop.shape[0] * crop.shape[1]),
                "border_trim_pixels": 2,
            },
            "axis_calibration": panel,
            "digitization_uncertainty": {
                "status": "not_estimated",
                "reason": "color quantization is not experimental uncertainty",
            },
        }
        for rect, color in blocks:
            distances = np.linalg.norm(palette - color, axis=1)
            best = float(distances.min())
            if best > 0.04:
                raise ExtractionError(
                    "needs_manual_review", "raster block color cannot be matched to colorbar"
                )
            matches = np.flatnonzero(distances <= best + 1e-5)
            if np.any(np.diff(matches) > 1) and np.ptp(matches) > 0.02 * len(palette):
                raise ExtractionError("needs_manual_review", "raster colorbar inverse is nonunique")
            low = (py0 + float(min(matches))) / scale
            high = (py0 + float(max(matches)) + 1) / scale
            a, bb, c, d = rect
            series["x"].append(cal.x.value((a + c) / 2))
            series["y"].append(cal.y.value((bb + d) / 2))
            series["z"].append(zaxis.value((low + high) / 2))
            series["z_ranges"].append(sorted([zaxis.value(low), zaxis.value(high)]))
            series["cell_bounds"].append(
                {
                    "x": sorted([cal.x.value(a), cal.x.value(c)]),
                    "y": sorted([cal.y.value(bb), cal.y.value(d)]),
                }
            )
            series["source_cell_rects"].append(rect)
            series["source_coordinates"].append([(a + c) / 2, (bb + d) / 2])
            series["color_distances"].append(best)
            series["value_states"].append(
                "colorbar_endpoint_saturation_possible"
                if min(matches) == 0 or max(matches) == len(palette) - 1
                else "color_inverted_estimate"
            )
        results.append(series)
    return results


def pixel_digitization(path: Path, calibration: Calibration, work_dir: Path, dpi: int) -> tuple[list, dict]:
    recipe = calibration.config.get("raster_extraction")
    if not recipe:
        raise ExtractionError(
            "needs_calibration",
            "raster extraction requires an automatically generated raster calibration recipe",
        )
    if recipe.get("kind") not in {"histogram_2d", "colored_traces"}:
        raise ExtractionError("unsupported", "unknown raster extraction recipe")
    image, scale = _pixel_source(path, calibration, dpi)
    image.save(work_dir / "source.png")
    series = (
        _pixel_cells(image, calibration.config, scale)
        if recipe["kind"] == "histogram_2d"
        else _pixel_traces(image, calibration, scale)
    )
    if not series:
        raise ExtractionError("needs_manual_review", "pixel extraction produced no supported series")
    return series, {
        "width": image.width / scale,
        "height": image.height / scale,
        "render_scale": scale,
        "dpi": dpi if scale != 1 else None,
        "raster_method": recipe["kind"],
        "paths": [],
    }
