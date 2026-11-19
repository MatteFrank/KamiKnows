"""Conservative PDF-axis and series proposals, independent of original scientific data."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from kamiknows.common import (
    create_output_dir,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
    write_manifest,
)
from .assets import local_path
from .digitization import (
    Axis,
    Calibration,
    _polyline,
    inspect_representation,
    prepare_pdf_orientation,
    step_profile,
)

METHOD = "axes_auto"


def _near(a: float, b: float, tolerance: float = 0.3) -> bool:
    return abs(a - b) <= tolerance


def _black(color: Any) -> bool:
    return color is not None and max(color) - min(color) < 0.05 and max(color) < 0.5


def frames(paths: list[dict], min_width: float = 60, min_height: float = 50) -> list[list[float]]:
    """Require a visible rectangular frame; do not guess axes from arbitrary curves."""
    candidates = []
    horizontal, vertical = [], []
    for path in paths:
        if not _black(path.get("color")):
            continue
        items = path["items"]
        if len(items) >= 4 and all(item[0] == "l" for item in items):
            points = [list(item[1]) for item in items] + [list(items[-1][2])]
            rect = [
                min(p[0] for p in points),
                min(p[1] for p in points),
                max(p[0] for p in points),
                max(p[1] for p in points),
            ]
            corners = [(rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[1]), (rect[2], rect[3])]
            if (
                math.dist(points[0], points[-1]) < 0.3
                and rect[2] - rect[0] > min_width
                and rect[3] - rect[1] > min_height
                and all(_near(a[0], b[0]) or _near(a[1], b[1]) for a, b in zip(points, points[1:]))
                and all(any(math.dist(c, p) < 0.3 for p in points) for c in corners)
            ):
                candidates.append(rect)
        for item in path["items"]:
            if item[0] == "re":
                r = list(item[1])
                if r[2] - r[0] > min_width and r[3] - r[1] > min_height:
                    candidates.append(r)
            if item[0] != "l":
                continue
            a, b = item[1:]
            if _near(a.y, b.y) and abs(a.x - b.x) > min_width:
                horizontal.append((min(a.x, b.x), max(a.x, b.x), a.y))
            if _near(a.x, b.x) and abs(a.y - b.y) > min_height:
                vertical.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    for left in vertical:
        for right in vertical:
            if (
                right[0] - left[0] <= min_width
                or not _near(left[1], right[1])
                or not _near(left[2], right[2])
            ):
                continue
            if all(
                any(_near(h[0], left[0]) and _near(h[1], right[0]) and _near(h[2], y) for h in horizontal)
                for y in [left[1], left[2]]
            ):
                candidates.append([left[0], left[1], right[0], left[2]])
    unique = []
    for r in candidates:
        if not any(all(_near(a, b) for a, b in zip(r, q)) for q in unique):
            unique.append(r)
    return unique


def _numeric_labels(words: list, paths: list[dict]) -> list[dict]:
    """Handle literal numbers and PDF-split 10/exponent labels, including vector minus glyphs."""
    labels = []
    used = set()
    for i, w in enumerate(words):
        if i in used:
            continue
        text = w[4].replace("−", "-")
        literal_power = re.fullmatch(r"10\^([-+]?\d{1,2})", text)
        if literal_power:
            text = "1e" + literal_power[1]
        multiplier_prefix = text.startswith(("×10", "x10", "X10"))
        if multiplier_prefix:
            text = text[1:]
        if re.fullmatch(r"[-+]?\d+", text) and any(
            v[5] == w[5]
            and "[" in v[4]
            and re.search("[A-Za-z]", v[4])
            and w[3] - w[1] < 0.75 * (v[3] - v[1])
            for v in words
        ):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            signs = [
                v
                for v in words
                if v[4] in {"−", "-"}
                and -2 <= w[0] - v[2] < (w[3] - w[1]) * 0.35
                and abs((v[1] + v[3] - w[1] - w[3]) / 2) < (w[3] - w[1]) * 0.25
            ]
            if len(signs) == 1:
                text = "-" + text
                sign = signs[0]
                w = (min(w[0], sign[0]), min(w[1], sign[1]), w[2], max(w[3], sign[3]), *w[4:])
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            height = w[3] - w[1]
            bars = [
                list(p["rect"])
                for p in paths
                if p.get("fill") is not None
                and _black(p.get("fill"))
                and "rect" in p
                and 0 < p["rect"].height <= height * 0.16
                and height * 0.15 <= p["rect"].width <= height * 0.8
                and -2 <= w[0] - p["rect"].x1 < height * 0.35
                and abs((p["rect"].y0 + p["rect"].y1 - w[1] - w[3]) / 2) < height * 0.25
            ]
            if len(bars) == 1:
                text = "-" + text
                bar = bars[0]
                w = (min(w[0], bar[0]), w[1], w[2], w[3], *w[4:])
        exponent = None
        if text == "10":
            nearby = [
                (j, v)
                for j, v in enumerate(words)
                if j != i
                and re.fullmatch(r"[-+]?\d{1,2}", v[4])
                and -2 <= v[0] - w[2] < max(25, (w[3] - w[1]) * 0.5)
                and abs(v[1] - w[1]) < max(6, (w[3] - w[1]) * 0.4)
                and v[3] - v[1] < 0.9 * (w[3] - w[1])
            ]
            if len(nearby) == 1:
                j, v = nearby[0]
                exponent = int(v[4])
                minus = any(
                    item[0] == "l"
                    and _near(item[1].y, item[2].y)
                    and w[2] <= min(item[1].x, item[2].x) < max(item[1].x, item[2].x) <= v[0] + 1
                    and v[1] <= item[1].y <= v[3]
                    for p in paths
                    for item in p["items"]
                )
                if minus:
                    exponent = -abs(exponent)
                used.add(j)
        try:
            value = 10.0 ** exponent if exponent is not None else float(text)
        except (ValueError, OverflowError):
            continue
        if math.isfinite(value):
            labels.append(
                {
                    "box": [w[0], w[1], max(w[2], v[2]) if exponent is not None else w[2], w[3]],
                    "text": text if exponent is None else f"10^{exponent}",
                    "value": value,
                    "power10": exponent is not None or literal_power is not None,
                    "multiplier_prefix": multiplier_prefix,
                    "text_segmentation": w[8] if len(w) > 8 else None,
                }
            )
    # PDF overprinting may repeat the same glyphs in several drawing layers.
    # Equal labels at equal positions are one observation, not competing ticks.
    return list(
        {(label["text"], tuple(round(v, 2) for v in label["box"])): label for label in labels}.values()
    )


def _axis_words(page: Any) -> list:
    """Undo word merging of superscript powers using PDF font-size evidence."""
    words = page.get_text("words")
    spans = [
        s
        for b in page.get_text("dict", flags=0)["blocks"]
        for line in b.get("lines", [])
        for s in line["spans"]
    ]
    for base in spans:
        if base["text"].strip() not in {"10", "×10", "x10", "X10"}:
            continue
        x0, y0, x1, y1 = base["bbox"]
        exponents = [
            s
            for s in spans
            if re.fullmatch(r"[-+]?\d{1,2}", s["text"].strip())
            and s["size"] < base["size"] * 0.85
            and -2 <= s["bbox"][0] - x1 < (y1 - y0) * 0.5
            and abs(s["bbox"][1] - y0) < (y1 - y0) * 0.4
        ]
        if len(exponents) != 1:
            continue
        exponent = exponents[0]
        ex0, ey0, ex1, ey1 = exponent["bbox"]
        words = [
            w
            for w in words
            if not (
                x0 - 0.5 <= w[0]
                and w[2] <= ex1 + 0.5
                and min(y0, ey0) - 0.5 <= w[1]
                and w[3] <= max(y1, ey1) + 0.5
            )
        ]
        words.extend(
            [
                (*base["bbox"], base["text"].strip(), 0, 0, 0),
                (*exponent["bbox"], exponent["text"].strip(), 0, 0, 0),
            ]
        )
    # Split adjacent numeric labels only at explicit glyph-position gaps.
    for block in page.get_text("rawdict", flags=0)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                chars = span["chars"]
                text = "".join(c["c"] for c in chars)
                if not re.fullmatch(r"\d{3,}", text):
                    continue
                groups = [[]]
                for char in chars:
                    if groups[-1] and char["bbox"][0] - groups[-1][-1]["bbox"][2] > max(
                        0.5, span["size"] * 0.03
                    ):
                        groups.append([])
                    groups[-1].append(char)
                if len(groups) < 2:
                    continue
                matching = [
                    w
                    for w in words
                    if w[4] == text and all(abs(a - b) < 1 for a, b in zip(w[:4], span["bbox"]))
                ]
                if len(matching) != 1:
                    continue
                original = matching[0]
                words.remove(original)
                for group in groups:
                    box = [
                        min(c["bbox"][0] for c in group),
                        min(c["bbox"][1] for c in group),
                        max(c["bbox"][2] for c in group),
                        max(c["bbox"][3] for c in group),
                    ]
                    words.append(
                        (
                            *box,
                            "".join(c["c"] for c in group),
                            *original[5:8],
                            {"source_word": text, "split_reason": "positioned_character_gap"},
                        )
                    )
    return words


def _tick_lengths(paths: list[dict], roi: list[float], axis: str) -> dict[float, float]:
    l, t, r, b = roi
    lengths = {}
    tick_limit = max(10, min(40, ((r - l) + (b - t)) * 0.025))
    if axis == "z":
        tick_limit = max(tick_limit, (r - l) * 0.8)
    for path in paths:
        if not _black(path.get("color")):
            continue
        for item in path["items"]:
            if item[0] != "l":
                continue
            a, z = item[1:]
            if (
                axis == "x"
                and _near(a.x, z.x)
                and 1 <= abs(a.y - z.y) <= tick_limit
                and (_near(a.y, b) or _near(z.y, b))
            ):
                if l <= a.x <= r:
                    key = max(l, min(r, round(a.x, 5)))
                    lengths[key] = max(lengths.get(key, 0), abs(a.y - z.y))
            if (
                axis in {"y", "z"}
                and _near(a.y, z.y)
                and 1 <= abs(a.x - z.x) <= tick_limit
                and (_near(a.x, r if axis == "z" else l) or _near(z.x, r if axis == "z" else l))
            ):
                if t <= a.y <= b:
                    key = max(t, min(b, round(a.y, 5)))
                    lengths[key] = max(lengths.get(key, 0), abs(a.x - z.x))
    return lengths


def _ticks(paths: list[dict], roi: list[float], axis: str) -> list[float]:
    return sorted(_tick_lengths(paths, roi, axis))


def _multiplier_labels(labels: list[dict], words: list) -> list[dict]:
    result = []
    for label in labels:
        if not label["power10"]:
            continue
        x0, y0, x1, y1 = label["box"]
        height = y1 - y0
        if label.get("multiplier_prefix") or any(
            w[4] in {"×", "x", "X"}
            and -2 <= x0 - w[2] < height * 0.6
            and abs((w[1] + w[3] - y0 - y1) / 2) < height * 0.5
            for w in words
        ):
            result.append(label)
    return result


def _log_minor_tick_support(
    paths: list[dict], roi: list[float], axis: str, fit: Axis, anchors: list[dict]
) -> int:
    """Two printed log labels need independent minor-tick geometry, not just notation."""
    low, high = sorted(a["position"] for a in anchors)
    matches = 0
    total = 0
    for position in _ticks(paths, roi, axis):
        if not low + 0.5 < position < high - 0.5:
            continue
        value = fit.value(position)
        decade = 10 ** math.floor(math.log10(value))
        mantissa = value / decade
        total += 1
        matches += int(2 <= round(mantissa) <= 9 and abs(mantissa - round(mantissa)) < 0.04)
    return matches if total >= 3 and matches / total >= 0.8 else 0


def infer_axis(
    paths: list[dict], words: list, roi: list[float], axis: str
) -> tuple[dict | None, dict]:
    l, t, r, b = roi
    labels = _numeric_labels(words, paths)
    anchors = []
    evidence = []
    multipliers = _multiplier_labels(labels, words)
    factors = [
        v
        for v in multipliers
        if (
            (v["box"][0] >= r - 40 and abs((v["box"][1] + v["box"][3]) / 2 - b) < 40)
            if axis == "x"
            else (abs((v["box"][1] + v["box"][3]) / 2 - t) < 40 and abs(v["box"][0] - l) < 90)
        )
    ]
    labels = [v for v in labels if v not in multipliers]
    lengths = _tick_lengths(paths, roi, axis)
    ticks = sorted(lengths)
    # Longer groups with enough labels distinguish major ticks from minor ticks.
    # All ticks remain available to the independent logarithmic-spacing check.
    for size in sorted({round(v, 1) for v in lengths.values()}, reverse=True):
        group = sorted(v for v, length in lengths.items() if abs(length - size) < 0.11)
        matched = []
        for label in labels:
            x0, y0, x1, y1 = label["box"]
            height = y1 - y0
            beside = (
                (b - height * 0.3 <= y0 < b + max(40, height) and (y0 + y1) / 2 > b)
                if axis == "x"
                else (
                    (r <= x0 < r + 90 and t - height < y0 < b)
                    if axis == "z"
                    else (l - 90 < x1 <= l and t - height < y0 < b)
                )
            )
            center = (x0 + x1) / 2 if axis == "x" else (y0 + y1) / 2
            if beside and group and min(abs(center - v) for v in group) <= min(12, height * 0.5):
                matched.append(label)
        if len(matched) >= 3 or (len(matched) >= 2 and any(v["power10"] for v in matched)):
            ticks = group
            break
    for tick in ticks:
        possible = []
        for label in labels:
            x0, y0, x1, y1 = label["box"]
            height = y1 - y0
            distance = abs((x0 + x1) / 2 - tick) if axis == "x" else abs((y0 + y1) / 2 - tick)
            nearest = min(abs((x0 + x1) / 2 - v) if axis == "x" else abs((y0 + y1) / 2 - v) for v in ticks)
            if distance > nearest + 1e-5:
                continue
            beside = (
                (b - height * 0.3 <= y0 < b + max(40, height) and (y0 + y1) / 2 > b)
                if axis == "x"
                else (
                    (r <= x0 < r + 90 and t - height < y0 < b)
                    if axis == "z"
                    else (l - 90 < x1 <= l and t - height < y0 < b)
                )
            )
            if beside and distance <= min(12, height * 0.5):
                possible.append((distance, label))
        possible.sort(key=lambda v: v[0])
        if not possible:
            continue
        if len(possible) > 1 and possible[1][0] - possible[0][0] < 2:
            continue
        label = possible[0][1]
        if any(e["box"] == label["box"] for e in evidence):
            continue
        anchors.append({"position": tick, "value": label["value"]})
        evidence.append(label)
    details = {"anchors": anchors, "labels": evidence}
    if len(anchors) < 2:
        return None, {**details, "reason": "fewer_than_three_unambiguous_ticks"}
    values = [a["value"] for a in anchors]
    if len(set(values)) != len(values) or not (
        all(a < b for a, b in zip(values, values[1:])) or all(a > b for a, b in zip(values, values[1:]))
    ):
        return None, {**details, "reason": "nonmonotone_or_duplicate_labels"}
    interval = (l, r) if axis == "x" else (t, b)
    fits = []
    for scale in ["linear", "log10"]:
        tolerance = (max(values) - min(values)) * 0.002 if scale == "linear" else 0.005
        config = {"scale": scale, "unit": "", "anchors": anchors, "max_residual": max(tolerance, 1e-8)}
        try:
            fit = Axis.fit(config, interval)
        except ValueError:
            continue
        if len(anchors) == 2:
            if scale != "log10" or not any(v["power10"] for v in evidence):
                continue
            support = _log_minor_tick_support(paths, roi, axis, fit, anchors)
            if not support:
                continue
            details["log_minor_tick_support"] = support
        fits.append((config, fit))
    if len(fits) != 1:
        return None, {
            **details,
            "reason": (
                "two_label_scale_requires_confirmation"
                if len(anchors) == 2 and not fits
                else "ambiguous_or_inconsistent_scale"
            ),
            "candidate_scales": [c["scale"] for c, f in fits],
        }
    units = []
    axis_words = []
    for w in words:
        beside = (
            (w[1] > b and l <= w[0] < r)
            if axis == "x"
            else (
                (r + 20 < w[0] < r + 140 and t <= (w[1] + w[3]) / 2 < b)
                if axis == "z"
                else (l - 120 < w[2] < l - 20 and t <= w[1] < b)
            )
        )
        if not beside:
            continue
        axis_words.append(w[4].lower())
        m = re.fullmatch(r"[\[(]([A-Za-zµμ%][A-Za-z0-9µμ%./^ -]*)[\])]", w[4])
        if m:
            units.append((m[1], w[4]))
        elif w[4].lower() in {"a.u.", "a.u", "%"}:
            units.append((w[4], w[4]))
    if not units and "cell" in axis_words and "id" in axis_words:
        units.append(("cell_index", "Cell ID (explicit index coordinate)"))
    units = list(dict.fromkeys(units))
    config, fit = fits[0]
    if len({v["value"] for v in factors}) > 1:
        return None, {**details, "reason": "ambiguous_axis_multiplier"}
    multiplier = factors[0]["value"] if factors else 1.0
    if multiplier != 1:
        details["raw_tick_values"] = [a["value"] for a in anchors]
        anchors = [{**a, "value": a["value"] * multiplier} for a in anchors]
        config["anchors"] = anchors
        details["anchors"] = anchors
        if config["scale"] == "linear":
            config["max_residual"] *= abs(multiplier)
        fit = Axis.fit(config, interval)
    config["display_multiplier"] = multiplier
    details["display_multiplier"] = multiplier
    details["multiplier_evidence"] = factors
    details.update(scale=config["scale"], residuals=fit.residuals, unit_candidates=units)
    config["unit"] = units[0][0] if len(units) == 1 else "unknown"
    config["unit_status"] = "explicit" if len(units) == 1 else "missing" if not units else "ambiguous"
    details["unit_status"] = config["unit_status"]
    return config, {**details, "reason": "axis_ticks_consistent"}


STEP_DRIFT = 0.1


def _step_candidate(path: dict, calibration: Calibration) -> frozenset | None:
    """Visible steps of a path that reads as a step histogram: two or more steps over a quarter of the frame."""
    try:
        visible, _ = step_profile(path, calibration, STEP_DRIFT)
    except (ValueError, TypeError):
        return None
    span = max(b[0] for _, b in visible) - min(a[0] for a, _ in visible)
    if len(visible) < 2 or span < (calibration.roi[2] - calibration.roi[0]) * 0.25:
        return None
    return frozenset(tuple(round(v, 1) for v in a + b) for a, b in visible)


def infer_series(paths: list[dict], calibration: Calibration) -> list[dict]:
    """Select data paths: step histograms (stroked, in pieces, or filled) and connected lines inside the frame.

    A fill and its outline drawn with the same steps are one series; the outline is kept.
    """
    steps, series = {}, {}
    for i, path in enumerate(paths):
        if len(path["items"]) < 3:
            continue
        candidate = _step_candidate(path, calibration)
        if candidate is not None:
            steps[i] = candidate
            continue
        if path.get("fill") is not None or path.get("closePath"):
            continue
        try:
            points, _ = _polyline({"items": path["items"]})
        except (ValueError, TypeError):
            continue
        if (
            math.dist(points[0], points[-1]) < 1e-5
            or max(p[0] for p in points) - min(p[0] for p in points)
            < (calibration.roi[2] - calibration.roi[0]) * 0.25
            or not all(calibration.contains(p) for p in points)
        ):
            continue
        series[i] = "line"

    def rank(i: int) -> tuple:
        return (paths[i].get("fill") is not None, i)

    for i, own in steps.items():
        if not any(j != i and own <= other and (own < other or rank(j) < rank(i)) for j, other in steps.items()):
            series[i] = "step_histogram"
    selected = []
    for i in sorted(series):
        spec = {
            "series_id": f"path_{i}",
            "label": None,
            "geometry": series[i],
            "path_indices": [i],
            "observable": "unreviewed_displayed_series",
            "coordinate_definition": "displayed source geometry mapped through proposed axis ticks",
        }
        if series[i] == "step_histogram":
            spec["max_step_drift"] = STEP_DRIFT
        selected.append(spec)
    return selected


def _with_vector_series(path: Path, config: dict) -> dict:
    """OCR may supply only the axes: when the curves are vector paths, select them natively instead of pixels."""
    import pymupdf

    if config.get("coordinate_system") != "pdf_points" or config["raster_extraction"]["kind"] != "colored_traces":
        return config
    with pymupdf.open(path) as document:
        page = document[0]
        prepare_pdf_orientation(page, config.get("pdf_orientation", "display"))
        series = infer_series(page.get_drawings(), Calibration(config))
    if not series:
        return config
    native = {key: value for key, value in config.items() if key != "raster_extraction"}
    return {**native, "series": series, "digitization_readiness": "ready"}


def _propose_1d(path: Path) -> tuple[dict | None, dict]:
    import pymupdf

    with pymupdf.open(path) as document:
        if document.page_count != 1:
            return None, {"reason": "multiple_pdf_pages"}
        page = document[0]
        prepare_pdf_orientation(page, "display")
        paths = page.get_drawings()
        words = _axis_words(page)
        if len(paths) > 20000:
            return None, {"reason": "vector_complexity_limit"}
        boxes = frames(paths)
        evidence = {"pdf_orientation": "display", "frame_candidates": boxes, "text_word_count": len(words)}
        evidence["native_numeric_candidates"] = len(_numeric_labels(words, paths))
        if not boxes:
            return None, {**evidence, "reason": "no_unique_single_panel_frame"}
        if any(
            min(a[2], b[2]) - max(a[0], b[0]) > 0.3 and min(a[3], b[3]) - max(a[1], b[1]) > 0.3
            for i, a in enumerate(boxes)
            for b in boxes[i + 1 :]
        ):
            return None, {
                **evidence,
                "reason": "no_unique_single_panel_frame",
                "frame_issue": "nested or overlapping candidates",
            }
        configs = []
        panels = []
        for index, roi in enumerate(sorted(boxes, key=lambda r: (r[1], r[0])), 1):
            x, xe = infer_axis(paths, words, roi, "x")
            y, ye = infer_axis(paths, words, roi, "y")
            panel = {"roi": roi, "x": xe, "y": ye, "missing_requirements": []}
            if x is None or y is None:
                if x is None:
                    panel["missing_requirements"].append("X: " + xe["reason"])
                if y is None:
                    panel["missing_requirements"].append("Y: " + ye["reason"])
                panels.append(panel)
                continue
            config = {
                "roi": roi,
                "coordinate_system": "pdf_points",
                "pdf_orientation": "display",
                "page": 1,
                "panel_index": index,
                "x": x,
                "y": y,
                "calibration_source": "axis_ticks",
                "proposal_method": METHOD,
                "review_status": "pending",
                "manual_interventions": [],
            }
            config["series"] = infer_series(paths, Calibration(config))
            config["digitization_readiness"] = "ready" if config["series"] else "series_selection_required"
            panel["digitization_readiness"] = config["digitization_readiness"]
            panels.append(panel)
            configs.append(config)
        if len(boxes) == 1:
            evidence.update(x=panels[0]["x"], y=panels[0]["y"])
        else:
            evidence["panels"] = panels
        if not configs:
            missing = [f"Panel {i}: " + m for i, p in enumerate(panels, 1) for m in p["missing_requirements"]]
            if evidence["native_numeric_candidates"] == 0:
                missing = [
                    "No native numeric text candidates were extracted. Labels may be outlines or pixels; OCR or manual anchors are required. This does not establish visual absence."
                ]
            return None, {
                **evidence,
                "reason": (
                    "native_numeric_text_unavailable"
                    if evidence["native_numeric_candidates"] == 0
                    else "axes_need_review"
                ),
                "missing_requirements": missing,
            }
        if len(boxes) == 1:
            config = configs[0]
            evidence["digitization_readiness"] = config["digitization_readiness"]
            return config, {
                **evidence,
                "reason": (
                    "proposal_generated" if config["series"] else "axes_calibrated_series_selection_required"
                ),
                "selected_paths": [s["path_indices"] for s in config["series"]],
            }
        roi = [
            min(p["roi"][0] for p in configs),
            min(p["roi"][1] for p in configs),
            max(p["roi"][2] for p in configs),
            max(p["roi"][3] for p in configs),
        ]
        config = {
            **configs[0],
            "roi": roi,
            "series": [],
            "panels_1d": configs,
            "panel_count": len(boxes),
            "calibration_completeness": "complete" if len(configs) == len(boxes) else "partial",
            "uncalibrated_panels": [
                {"panel_index": i, **p} for i, p in enumerate(panels, 1) if p["missing_requirements"]
            ],
            "digitization_readiness": "panel_digitization_required",
        }
        evidence["digitization_readiness"] = config["digitization_readiness"]
        return config, {
            **evidence,
            "reason": (
                "multipanel_1d_axes_calibrated"
                if len(configs) == len(boxes)
                else "multipanel_1d_axes_partially_calibrated"
            ),
        }


def propose_pdf(path: Path, expected_type: str | None = None) -> tuple[dict | None, dict]:
    """Try quantitative recipes, accepting alternatives only with PDF-backed axes."""
    if expected_type == "histogram_2d":
        config, evidence = _propose_2d(path)
        if config is not None:
            return config, evidence
        if evidence["reason"] == "histogram2d_missing_frame_or_colorbar":
            alternative, details = _propose_1d(path)
            if alternative is not None and alternative.get("series"):
                kind = (
                    "histogram_1d"
                    if all(s["geometry"] == "step_histogram" for s in alternative["series"])
                    else "line_plot"
                )
                details.update(
                    calibration_type=kind,
                    suggested_type=expected_type,
                    routing_reason="Single PDF frame with verified X/Y axes and selected vector series; no numeric colorbar",
                )
                return alternative, details
        return config, evidence
    if expected_type in {"unknown", "scatter_plot"}:
        config, evidence = _propose_2d(path)
        if config is not None:
            evidence.update(
                calibration_type="histogram_2d",
                suggested_type=expected_type,
                routing_reason="PDF panels with independently verified X/Y/Z axes",
            )
            return config, evidence
    config, evidence = _propose_1d(path)
    if (
        (config is None or config.get("panels_1d"))
        and expected_type in {"scatter_plot", "line_plot", "histogram_1d"}
        and (config is not None or evidence["reason"] == "no_unique_single_panel_frame")
    ):
        alternative, details = _propose_2d(path)
        if alternative is not None:
            details.update(
                calibration_type="histogram_2d",
                suggested_type=expected_type,
                routing_reason="PDF panels with independently verified X/Y/Z axes",
            )
            return alternative, details
    if expected_type == "unknown":
        if config is None:
            return None, evidence
        selected = config.get("series") or [
            s for panel in config.get("panels_1d", []) for s in panel.get("series", [])
        ]
        kind = (
            ("histogram_1d" if all(s["geometry"] == "step_histogram" for s in selected) else "line_plot")
            if selected
            else "xy_numeric_axes"
        )
        if not selected:
            config["calibration_scope"] = "xy_axes_only; representation unresolved"
        evidence.update(
            calibration_type=kind,
            suggested_type=expected_type,
            routing_reason="Unknown visual class; PDF verifies numeric axes and selected vector series",
        )
    return config, evidence


def _palette_strip_candidates(paths: list[dict]) -> list[list[float]]:
    """Find contiguous colored strips; numeric Z ticks must verify each candidate."""
    groups: dict[tuple, list] = {}
    for path in paths:
        if path.get("fill") is None or max(path["fill"]) - min(path["fill"]) < 0.05:
            continue
        for item in path["items"]:
            if item[0] != "re":
                continue
            box = list(item[1])
            if box[2] - box[0] < 4 or box[3] - box[1] <= 0:
                continue
            groups.setdefault((round(box[0], 1), round(box[2], 1)), []).append(box)
    result = []
    for strips in groups.values():
        strips = list({tuple(round(v, 2) for v in box): box for box in strips}.values())
        strips.sort(key=lambda box: box[1])
        if len(strips) < 4 or any(abs(a[3] - b[1]) > 0.3 for a, b in zip(strips, strips[1:])):
            continue
        box = [strips[0][0], strips[0][1], strips[0][2], strips[-1][3]]
        if box[3] - box[1] > 50 and (box[2] - box[0]) / (box[3] - box[1]) < 0.25:
            result.append(box)
    return result


def _propose_2d(path: Path) -> tuple[dict | None, dict]:
    import pymupdf
    from .digitization import pdf_cell_geometry, colorbar_palette

    with pymupdf.open(path) as document:
        if document.page_count != 1:
            return None, {"reason": "multiple_pdf_pages", "missing_requirements": ["single PDF page"]}
        page = document[0]
        prepare_pdf_orientation(page, "display")
        paths = page.get_drawings()
        words = _axis_words(page)
        if len(paths) > 20000:
            return None, {
                "reason": "vector_complexity_limit",
                "missing_requirements": ["bounded vector geometry"],
            }
        boxes = frames(paths, min_width=4, min_height=20)
        bars = [box for box in boxes if (box[2] - box[0]) / (box[3] - box[1]) < 0.25]
        data_frames = [
            box for box in boxes if box[2] - box[0] > 60 and box[3] - box[1] > 50 and box not in bars
        ]
        # Avoid interpreting arbitrary dense cell geometry as hundreds of palettes.
        # Closed bars already have sufficient geometry; this is a bounded fallback.
        strip_candidates = _palette_strip_candidates(paths) if not bars and 0 < len(data_frames) <= 12 else []
        for box in strip_candidates:
            if not any(
                0 <= box[0] - roi[2] < (roi[2] - roi[0]) * 0.5
                and abs(box[1] - roi[1]) < (roi[3] - roi[1]) * 0.1
                and abs(box[3] - roi[3]) < (roi[3] - roi[1]) * 0.1
                for roi in data_frames
            ):
                continue
            if not any(all(_near(a, b) for a, b in zip(box, bar)) for bar in bars):
                z, _ = infer_axis(paths, words, box, "z")
                if z is not None:
                    bars.append(box)
        data = [box for box in boxes if box[2] - box[0] > 60 and box[3] - box[1] > 50 and box not in bars]
        evidence = {
            "pdf_orientation": "display",
            "frame_candidates": data,
            "colorbar_candidates": bars,
            "panels": [],
            "missing_requirements": [],
        }
        if not data or not bars:
            return None, {
                **evidence,
                "reason": "histogram2d_missing_frame_or_colorbar",
                "missing_requirements": [
                    name for name, values in [("data frame", data), ("colorbar", bars)] if not values
                ],
            }
        configs = []
        used_bars = []
        union = [
            min(r[0] for r in data),
            min(r[1] for r in data),
            max(r[2] for r in data),
            max(r[3] for r in data),
        ]
        height = union[3] - union[1]
        shared_bar = (
            bars[0]
            if len(data) > 1
            and len(bars) == 1
            and 0 <= bars[0][0] - union[2] < (union[2] - union[0]) * 0.5
            and abs(bars[0][1] - union[1]) < height * 0.15
            and abs(bars[0][3] - union[3]) < height * 0.15
            else None
        )
        for roi in data:
            panel = {"roi": roi}
            candidates = [
                bar
                for bar in bars
                if 0 <= bar[0] - roi[2] < (roi[2] - roi[0]) * 0.5
                and abs(bar[1] - roi[1]) < (roi[3] - roi[1]) * 0.1
                and abs(bar[3] - roi[3]) < (roi[3] - roi[1]) * 0.1
            ]
            if shared_bar is not None:
                candidates = [shared_bar]
            if len(candidates) != 1 or (shared_bar is None and candidates[0] in used_bars):
                panel.update(
                    reason="ambiguous_colorbar_association",
                    missing_requirements=["unique panel/colorbar pairing"],
                )
                evidence["panels"].append(panel)
                continue
            bar = candidates[0]
            used_bars.append(bar)
            panel["colorbar_roi"] = bar
            panel["colorbar_shared"] = shared_bar is not None
            x, xe = infer_axis(paths, words, roi, "x")
            y, ye = infer_axis(paths, words, roi, "y")
            z, ze = infer_axis(paths, words, bar, "z")
            panel.update(x=xe, y=ye, z=ze)
            missing = []
            for axis, config in [("X", x), ("Y", y), ("Z", z)]:
                if config is None:
                    missing.append(axis + ": " + {"X": xe, "Y": ye, "Z": ze}[axis]["reason"])
            geometry_missing = []
            if not missing:
                try:
                    cells, source = pdf_cell_geometry(page, roi)
                    colorbar_palette(page, bar)
                    panel.update(cell_count=len(cells), cell_source=source)
                except (ValueError, RuntimeError) as exc:
                    geometry_missing.append(str(exc))
            panel["geometry_requirements"] = geometry_missing
            panel["digitization_readiness"] = (
                "geometry_review_required"
                if geometry_missing
                else "ready" if not missing else "axes_not_calibrated"
            )
            panel["missing_requirements"] = missing
            panel["reason"] = "requirements_missing" if missing else "panel_proposed"
            evidence["panels"].append(panel)
            if not missing:
                configs.append({"roi": roi, "colorbar_roi": bar, "x": x, "y": y, "z": z})
        for panel in evidence["panels"]:
            evidence["missing_requirements"].extend(panel.get("missing_requirements", []))
        # Do not silently publish only the easy panels of a compound asset.
        if not configs or len(configs) != len(data):
            return None, {**evidence, "reason": "histogram2d_requirements_missing"}
        roi = [
            min(c["roi"][0] for c in configs),
            min(c["roi"][1] for c in configs),
            max(c["roi"][2] for c in configs),
            max(c["roi"][3] for c in configs),
        ]
        config = {
            "roi": roi,
            "coordinate_system": "pdf_points",
            "pdf_orientation": "display",
            "page": 1,
            "x": configs[0]["x"],
            "y": configs[0]["y"],
            "histogram2d_panels": configs,
            "calibration_source": "axis_ticks",
            "proposal_method": METHOD,
            "review_status": "pending",
            "manual_interventions": [],
        }
        pending = any(p.get("geometry_requirements") for p in evidence["panels"])
        config["digitization_readiness"] = "geometry_review_required" if pending else "ready"
        evidence["digitization_readiness"] = config["digitization_readiness"]
        return config, {
            **evidence,
            "reason": (
                "axes_calibrated_geometry_review_required" if pending else "histogram2d_proposal_generated"
            ),
        }


def visual_numeric_gate(regions: list[dict], expected: str | None) -> tuple[str | None, list[str]]:
    """Summarize visual numeric-label presence separately from independent PDF verification."""
    quantitative = {"line_plot", "scatter_plot", "histogram_1d", "histogram_2d", "bar_chart"}
    if expected not in quantitative:
        return "needs_review_axes", ["Gemma did not establish a quantitative class"]
    if not regions:
        return "needs_review_axes", ["Axis observations are unavailable for this asset"]
    absent = []
    unclear = []
    required = ["x", "y"] + (["colorbar"] if expected == "histogram_2d" else [])
    for index, region in enumerate(regions, 1):
        if region.get("analysis_type") != expected:
            unclear.append(f"Region {index}: class conflicts with the figure class")
        for name in required:
            observation = region.get("axes", {}).get(name, {})
            prefix = f"Region {index}, " + ("Z (colorbar)" if name == "colorbar" else name.upper())
            labels = observation.get(
                "numeric_labels",
                {"readable": "present", "absent": "absent"}.get(observation.get("numeric_ticks"), "unclear"),
            )
            if observation.get("presence") == "absent" or labels == "absent":
                absent.append(prefix + ": numeric labels absent according to Gemma")
            elif observation.get("presence") != "present" or labels != "present":
                unclear.append(prefix + ": numeric-label presence uncertain according to Gemma")
    if absent:
        return "excluded_no_numbers", absent + unclear
    if unclear:
        return "needs_review_axes", unclear
    return None, []


@lru_cache(maxsize=1)
def _ocr_engine():
    """Load CPU OCR lazily; package models are local, with no runtime download."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("OCR requires rapidocr-onnxruntime; install requirements.txt") from exc
    return RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)


def _ocr_model_inputs() -> dict[str, Path]:
    try:
        import rapidocr_onnxruntime
    except ImportError as exc:
        raise RuntimeError("OCR requires rapidocr-onnxruntime; install requirements.txt") from exc
    return {
        "ocr_model:" + model.name: model
        for model in sorted((Path(rapidocr_onnxruntime.__file__).parent / "models").glob("*.onnx"))
    }


def _ocr_power_notation(image: Any, box: list[float], text: str) -> str:
    """Recover powers only when suffix glyphs are visibly raised above the base."""
    import cv2

    match = re.fullmatch(r"10([-+−]?\d{1,2})", text.replace(" ", ""))
    if not match:
        return text
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = image[max(0, y0) : y1, max(0, x0) : x1]
    if not crop.size:
        return text
    mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 120)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    glyphs = sorted([s for s in stats[1:] if s[4] >= 4], key=lambda s: s[0])
    if len(glyphs) < 3:
        return text
    max_height = max(s[3] for s in glyphs)
    # A cropped axis spine can cross the top of a printed power label.
    glyphs = [s for s in glyphs if not (s[2] > max_height * 1.5 and s[3] < max_height * 0.25)]
    if len(glyphs) < 3:
        return text
    first, second, *suffix = glyphs
    height = max(first[3], second[3])
    baseline = max(first[1] + first[3], second[1] + second[3])
    if min(first[3], second[3]) < height * 0.7:
        return text
    if all(
        g[0] >= second[0] + second[2] - 1 and g[1] + g[3] < baseline - height * 0.15 and g[3] < height * 0.9
        for g in suffix
    ):
        return "10^" + match[1].replace("−", "-")
    return text


def _raster_axis_ticks(
    mask: Any, boxes: list[list[float]], tick_mask: Any | None = None
) -> tuple[list, list]:
    """Snap thick raster spines to pixel centers and retain connected outward ticks."""
    import numpy as np
    import pymupdf

    tick_mask = mask if tick_mask is None else tick_mask
    normalized = []
    paths = []

    def center(values):
        return float(np.mean(values))

    for box in boxes:
        l, t, r, b = map(int, box)

        def snap(position, lower, upper, horizontal):
            margin = max(20, min(30, int(((r - l) + (b - t)) * 0.01)))
            candidates = range(
                max(0, position - margin), min(mask.shape[0 if horizontal else 1], position + margin + 1)
            )
            scores = [
                (p, int(np.count_nonzero(mask[p, lower:upper] if horizontal else mask[lower:upper, p])))
                for p in candidates
            ]
            peak = max(s for p, s in scores)
            return int(round(center([p for p, s in scores if s >= peak * 0.99])))

        top = snap(t, l, r, True)
        bottom = snap(b, l, r, True)
        left = snap(l, t, b, False)
        right = snap(r, t, b, False)
        roi = [left, top, right, bottom]
        if any(
            all(abs(a - b) < max(5, min(right - left, bottom - top) * 0.01) for a, b in zip(roi, other))
            for other in normalized
        ):
            continue
        normalized.append(roi)
        limit = max(10, min(40, int(((right - left) + (bottom - top)) * 0.025)))
        for axis in ["x", "y", "z"]:
            if axis == "x":
                strip = tick_mask[bottom + 2 : min(mask.shape[0], bottom + 2 + limit), left:right] > 0
                lengths = np.cumprod(strip, axis=0).sum(axis=0) if strip.size else []
                offset = left
            elif axis == "y":
                indices = list(range(left - 2, max(-1, left - 2 - limit), -1))
                strip = tick_mask[top:bottom, indices] > 0
                lengths = np.cumprod(strip, axis=1).sum(axis=1) if strip.size else []
                offset = top
            else:
                strip = tick_mask[top:bottom, right + 2 : min(mask.shape[1], right + 2 + limit)] > 0
                lengths = np.cumprod(strip, axis=1).sum(axis=1) if strip.size else []
                offset = top
            positions = [i for i, length in enumerate(lengths) if length >= 3]
            groups = []
            for i in positions:
                if not groups or i > groups[-1][-1] + 1:
                    groups.append([])
                groups[-1].append(i)
            for group in groups:
                position = offset + center(group)
                length = float(min(limit, max(lengths[i] for i in group) + 2))
                a, z = (
                    ((position, bottom), (position, bottom + length))
                    if axis == "x"
                    else (
                        ((left, position), (left - length, position))
                        if axis == "y"
                        else ((right, position), (right + length, position))
                    )
                )
                paths.append({"color": (0, 0, 0), "items": [("l", pymupdf.Point(*a), pymupdf.Point(*z))]})
    return normalized, paths


def _ocr_pixel_geometry(image: Any) -> tuple[list, list, list]:
    """Detect frames and ticks from pixels, including bitmap-only PDFs."""
    import cv2
    import numpy as np
    import pymupdf

    # Colored data crossing a frame must not merge with the black spine contour.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 20)
    paths = []
    # Bridge short occlusions from strokes crossing the frame, only for detection.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(polygon)
        rectangular = len(polygon) == 4 and cv2.contourArea(polygon) / (w * h) > 0.97
        narrow = w / h < 0.2 and cv2.contourArea(contour) / (w * h) > 0.65
        if w > 4 and h > 50 and (rectangular or narrow):
            paths.append({"color": (0, 0, 0), "items": [("re", pymupdf.Rect(x, y, x + w - 1, y + h - 1), 1)]})
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=15, minLineLength=2, maxLineGap=1)
    for line in ([] if lines is None else lines[:1000]):
        x0, y0, x1, y1 = map(float, line[0])
        if abs(x0 - x1) <= 1 or abs(y0 - y1) <= 1:
            paths.append({"color": (0, 0, 0), "items": [("l", pymupdf.Point(x0, y0), pymupdf.Point(x1, y1))]})
    spine_mask = cv2.inRange(gray, 0, 20)
    raw_bars = [
        box
        for box in frames(paths, min_width=4, min_height=50)
        if (box[2] - box[0]) / (box[3] - box[1]) < 0.2
    ]
    tick_mask = cv2.inRange(gray, 0, 100)
    bars, bar_ticks = _raster_axis_ticks(spine_mask, raw_bars, tick_mask)
    boxes, pixel_ticks = _raster_axis_ticks(spine_mask, frames(paths), tick_mask)
    boxes = [box for box in boxes if not any(all(abs(a - b) < 5 for a, b in zip(box, bar)) for bar in bars)]
    return boxes, bars, paths + pixel_ticks + bar_ticks


def ocr_axis_candidates(path: Path) -> dict:
    """Keep OCR observations and candidate fits unverified, outside consumable proposals."""
    import cv2
    import numpy as np
    import pymupdf
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as document:
            if document.page_count != 1:
                raise ValueError("OCR fallback requires a single PDF page")
            page = document[0]
            prepare_pdf_orientation(page, "display")
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))[:, :, ::-1].copy()
            paths = page.get_drawings()
            factor = 2.0
    else:
        image = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        paths = []
        factor = 1.0
    boxes = frames(paths)
    bars = []
    if factor == 1 or not boxes:
        boxes, bars, pixel_paths = _ocr_pixel_geometry(image)
        if factor != 1:
            boxes = [[v / factor for v in box] for box in boxes]
            bars = [[v / factor for v in box] for box in bars]
            for path_item in pixel_paths:
                converted = []
                for item in path_item["items"]:
                    if item[0] == "re":
                        converted.append(("re", pymupdf.Rect(*[v / factor for v in item[1]]), item[2]))
                    else:
                        converted.append(
                            (
                                "l",
                                pymupdf.Point(item[1].x / factor, item[1].y / factor),
                                pymupdf.Point(item[2].x / factor, item[2].y / factor),
                            )
                        )
                path_item["items"] = converted
        paths = pixel_paths
    results, _ = _ocr_engine()(image)
    results = list(results or [])
    # Re-read axis bands at their original resolution; full-image resizing can
    # erase small zeros and join adjacent labels. Coordinates always use pixels.
    axis_bands = []
    for roi in boxes[:8]:
        l, t, r, b = [v * factor for v in roi]
        margin = max(40, (b - t) * 0.15)
        axis_bands.extend(
            [
                (
                    max(0, l - margin),
                    b,
                    min(image.shape[1], r + margin * 0.3),
                    min(image.shape[0], b + margin),
                ),
                (max(0, l - margin), max(0, t - margin * 0.2), l, min(image.shape[0], b + margin * 0.2)),
                (max(0, l - margin), b, min(image.shape[1], l + margin), min(image.shape[0], b + margin)),
            ]
        )
    if not boxes:
        height, width = image.shape[:2]
        axis_bands = [(0, int(height * 0.7), int(width * 0.35), height)]
    for band in axis_bands:
        x0, y0, x1, y1 = map(int, band)
        crop = image[y0:y1, x0:x1]
        if not crop.size:
            continue
        extra, _ = _ocr_engine()(crop)
        for polygon, text, score in extra or []:
            shifted = [[float(x + x0), float(y + y0)] for x, y in polygon]
            # Keep the original observation when an overlapping pass agrees.
            if any(
                text == v[1]
                and abs(min(p[0] for p in shifted) - min(p[0] for p in v[0])) < 10
                and abs(min(p[1] for p in shifted) - min(p[1] for p in v[0])) < 10
                for v in results
            ):
                continue
            results.append((shifted, text, score))
    observations = []
    words = []
    for polygon, text, score in results or []:
        pixel_box = [
            min(p[0] for p in polygon),
            min(p[1] for p in polygon),
            max(p[0] for p in polygon),
            max(p[1] for p in polygon),
        ]
        normalized = _ocr_power_notation(image, pixel_box, text)
        box = [
            min(p[0] for p in polygon) / factor,
            min(p[1] for p in polygon) / factor,
            max(p[0] for p in polygon) / factor,
            max(p[1] for p in polygon) / factor,
        ]
        observations.append(
            {
                "box": box,
                "text": normalized,
                "raw_text": text,
                "score": float(score),
                "verified": False,
                "normalization": "raised_suffix_geometry" if normalized != text else None,
            }
        )
        # Independent OCR detections are separate text blocks. Reusing block 0
        # makes a tall unit label incorrectly suppress ticks as unit exponents.
        if score >= 0.8:
            words.append((*box, normalized, len(words), 0, 0))
    candidates = []
    for box in boxes[:20]:
        xc, x = infer_axis(paths, words, box, "x")
        yc, y = infer_axis(paths, words, box, "y")
        panel = {"roi": box, "x": x, "y": y, "axis_configs": {"x": xc, "y": yc}}
        nearby = [
            bar
            for bar in bars
            if 0 < bar[0] - box[2] < (box[2] - box[0]) * 0.5
            and abs(bar[1] - box[1]) < (box[3] - box[1]) * 0.1
            and abs(bar[3] - box[3]) < (box[3] - box[1]) * 0.1
        ]
        if len(nearby) == 1:
            zc, z = infer_axis(paths, words, nearby[0], "z")
            panel["axis_configs"]["z"] = zc
            panel.update(colorbar_roi=nearby[0], z=z)
        candidates.append(panel)
    import importlib.metadata

    return {
        "backend": "rapidocr-onnxruntime",
        "backend_version": importlib.metadata.version("rapidocr-onnxruntime"),
        "review_status": "pending",
        "numeric_verified": False,
        "coordinate_system": "pdf_points" if factor == 2 else "pixels",
        "observations": observations,
        "frame_candidates": boxes[:20],
        "panels": candidates,
        "limitation": "OCR can lose minus signs or superscripts. Automatic recipes require consistent fits on at least three labels per required axis; scientific review remains pending. Broken axes require separate mappings.",
    }


def raster_calibration_proposal(evidence: dict, expected: str | None) -> dict | None:
    """Publish a raster recipe only for a single frame with at least three consistent labels per required axis."""
    if expected not in {"line_plot", "histogram_1d", "histogram_2d"} or len(evidence.get("panels", [])) != 1:
        return None
    panel = evidence["panels"][0]
    axes = ["x", "y"] + (["z"] if expected == "histogram_2d" else [])
    if any(
        panel.get(a, {}).get("reason") != "axis_ticks_consistent"
        or len(panel[a].get("anchors", [])) < 3
        or not panel.get("axis_configs", {}).get(a)
        for a in axes
    ):
        return None
    config = {
        "roi": panel["roi"],
        "coordinate_system": evidence["coordinate_system"],
        "page": 1,
        "pdf_orientation": "display",
        "x": panel["axis_configs"]["x"],
        "y": panel["axis_configs"]["y"],
        "calibration_source": "ocr_axis_ticks",
        "proposal_method": METHOD,
        "review_status": "pending",
        "manual_interventions": [],
        "raster_extraction": {
            "kind": "histogram_2d" if expected == "histogram_2d" else "colored_traces",
            "method": "pixel_geometry_v1",
        },
        "digitization_readiness": "prototype_raster_backend",
    }
    if expected == "histogram_2d":
        config["histogram2d_panels"] = [
            {"roi": panel["roi"], "colorbar_roi": panel["colorbar_roi"], **panel["axis_configs"]}
        ]
    return config


def run_calibration(
    *,
    classified_dir: Path,
    parsed_dir: Path,
    output_dir: Path,
    inventory_dir: Path | None = None,
    figure_ids: list[str] | None = None,
    asset_ids: list[str] | None = None,
    visual_predictions: Path | None = None,
) -> dict[str, Any]:
    """Propose axis mappings automatically: native PDF text first, lazy OCR fallback.

    ``visual_predictions`` replaces the classified Gemma predictions with a sidecar
    JSONL; tests use it to exercise routing without a model server.
    """
    from .validation import run_calibration_review
    from .pipeline import _recovery_inputs, _verify_paper_sources

    classified_dir, parsed_dir = Path(classified_dir).resolve(), Path(parsed_dir).resolve()
    figures, panels, papers, inputs, original, upstream = _recovery_inputs(
        classified_dir, parsed_dir, inventory_dir
    )
    if figure_ids is not None:
        if (
            not figure_ids
            or len(set(figure_ids)) != len(figure_ids)
            or not set(figure_ids) <= {f.figure_id for f in figures}
        ):
            raise ValueError("figure_ids must contain unique existing figure IDs")
        figures = [f for f in figures if f.figure_id in figure_ids]
    if asset_ids is not None:
        available = {a["asset_id"] for f in figures for a in f.source_assets}
        if not asset_ids or len(set(asset_ids)) != len(asset_ids) or not set(asset_ids) <= available:
            raise ValueError("asset_ids must contain unique existing asset IDs")
        figures = [f for f in figures if any(a["asset_id"] in asset_ids for a in f.source_assets)]
    output_dir = Path(output_dir).resolve()
    if any(output_dir.is_relative_to(p) for p in [classified_dir, parsed_dir, upstream]):
        raise ValueError("output overlaps input run")
    hints = {}
    if visual_predictions is not None:
        visual_predictions = Path(visual_predictions).resolve()
        inputs["visual_predictions"] = visual_predictions
        for r in read_jsonl(visual_predictions):
            if r["figure_id"] in hints:
                raise ValueError("duplicate visual figure ID")
            hints[r["figure_id"]] = r
    output_dir = create_output_dir(output_dir)
    configs = []
    reports = []
    roots = {}
    errors = {}
    for paper in papers:
        try:
            _, roots[paper["paper_id"]] = _verify_paper_sources(
                parsed_dir, paper["paper_id"], original["inputs"], inputs
            )
        except (ValueError, OSError) as exc:
            errors[paper["paper_id"]] = str(exc)
    for figure in figures:
        for asset in figure.source_assets:
            if asset_ids is not None and asset["asset_id"] not in asset_ids:
                continue
            report = {
                "figure_id": figure.figure_id,
                "asset_id": asset["asset_id"],
                "panel_id": asset.get("panel_id"),
                "status": "needs_review",
                "review_status": "pending",
                "visual_hint": hints.get(figure.figure_id),
                "visual_hint_usage": "routes representation; never supplies numeric anchors",
            }
            sidecar = hints.get(figure.figure_id, {})
            asset_predictions = {
                **figure.classification_evidence.get("asset_predictions", {}),
                **sidecar.get("asset_predictions", {}),
            }
            asset_result = asset_predictions.get(asset["asset_id"]) or {}
            hint = (
                (asset_result.get("prediction") or {})
                if asset_predictions
                else (
                    (sidecar.get("prediction") or {})
                    if sidecar
                    else figure.classification_evidence.get("analysis") or {}
                )
            )
            report["visual_hint"] = hint or None
            number = next(
                i for i, a in enumerate(figure.source_assets, 1) if a["asset_id"] == asset["asset_id"]
            )
            report["visual_axis_observations"] = [
                r for r in hint.get("regions", []) if r.get("asset_number") == number and r.get("axes")
            ]
            expected = hint.get("analysis_type") or (
                figure.analysis_type
                if not asset_predictions and not sidecar and figure.analysis_type != "unknown"
                else None
            )
            report["expected_type"] = expected
            report["hint_source"] = "gemma" if hint else "classified_record"
            try:
                if figure.paper_id in errors:
                    raise ValueError(errors[figure.paper_id])
                if asset.get("status") != "resolved":
                    raise ValueError("unresolved asset")
                path = local_path(roots[figure.paper_id], roots[figure.paper_id] / asset["path"])
                inputs[asset["asset_id"]] = path
                if sha256_file(path) != asset["sha256"]:
                    raise ValueError("asset hash mismatch")
                report["source_inspection"] = inspect_representation(path)
                report["routing"] = {
                    "preferred": (
                        "pymupdf"
                        if report["source_inspection"]["representation"] in {"vector", "hybrid"}
                        else "raster"
                    ),
                    "attempts": [],
                }
                report["asset_role"] = hint.get("asset_role", "unknown")
                report["visual_description"] = hint.get("visual_description")
                quantitative = expected in {
                    "line_plot",
                    "scatter_plot",
                    "histogram_1d",
                    "histogram_2d",
                    "bar_chart",
                }
                non_quantitative_role = report["asset_role"] in {
                    "visual_comparison",
                    "legend",
                    "diagram",
                    "image",
                    "mixed",
                }
                report["class_role_conflict"] = quantitative and non_quantitative_role
                model_skip = expected in {"diagram", "image", "mixed"} or (
                    non_quantitative_role and not quantitative
                )
                report["model_skip_suggestion"] = model_skip
                audited = None
                if expected == "mixed":
                    # A model's mixed label must not hide independently recoverable axes.
                    if path.suffix.lower() == ".pdf":
                        audited = propose_pdf(path, "unknown")
                    if audited is None or audited[0] is None:
                        report.update(
                            reason="explicit_mixed_class_requires_review",
                            status="needs_review_classification",
                        )
                        reports.append(report)
                        continue
                    report["classification_review_required"] = True
                if path.suffix.lower() != ".pdf":
                    report.update(
                        reason="raster_requires_ocr_or_manual_calibration",
                        status="needs_raster_calibration",
                        missing_requirements=[
                            "Native-PDF calibration cannot process raster pixels. OCR or manual anchors are required."
                        ],
                    )
                    inputs.update(_ocr_model_inputs())
                    report["ocr_evidence"] = ocr_axis_candidates(path)
                    report.update(status="needs_review_ocr", reason="raster_ocr_candidates_unverified")
                    report["missing_requirements"] = [
                        "OCR has been run. Candidate axis labels and coordinate mappings require verification."
                    ]
                    config = raster_calibration_proposal(report["ocr_evidence"], expected)
                    report["routing"].update(
                        attempts=["ocr"], selected="raster", fallback_reason="raster_source"
                    )
                    if config is not None:
                        config.update(
                            figure_id=figure.figure_id,
                            asset_id=asset["asset_id"],
                            panel_id=asset.get("panel_id"),
                        )
                        if report["class_role_conflict"]:
                            config["classification_review_required"] = True
                        configs.append(config)
                        report.update(
                            status="proposed_raster",
                            reason="automatic_raster_calibration",
                            digitization_readiness=config["digitization_readiness"],
                            calibration_status="ocr_axes_calibrated_pending_visual_review",
                            missing_requirements=[],
                        )
                    reports.append(report)
                    continue
                if model_skip:
                    candidate, candidate_evidence = propose_pdf(path, "unknown")
                    report["skip_audit"] = {
                        "attempted": True,
                        "numeric_candidate_found": candidate is not None,
                        "reason": candidate_evidence["reason"],
                    }
                    if candidate is None:
                        report.update(
                            status=(
                                "skipped_class"
                                if report["asset_role"] == "visual_comparison"
                                else "needs_review_classification"
                            ),
                            reason=(
                                "qualitative_visual_comparison"
                                if report["asset_role"] == "visual_comparison"
                                else "model_class_not_independently_established"
                            ),
                            evidence=candidate_evidence,
                        )
                        reports.append(report)
                        continue
                    audited = (candidate, candidate_evidence)
                    report["class_role_conflict"] = True
                    report["classification_review_required"] = True
                if (
                    (hint or figure.classification_method.startswith("vision_"))
                    and expected
                    not in {
                        "line_plot",
                        "scatter_plot",
                        "histogram_1d",
                        "histogram_2d",
                        "bar_chart",
                        "unknown",
                    }
                    and audited is None
                ):
                    report.update(status="needs_review_axes", reason="quantitative_class_unresolved")
                    reports.append(report)
                    continue
                gate, visual_missing = visual_numeric_gate(report["visual_axis_observations"], expected)
                report["visual_numeric_assessment"] = {
                    "status": gate or "numbers_indicated",
                    "details": visual_missing,
                }
                if report["source_inspection"]["representation"] == "raster":
                    config, evidence = None, {"reason": "raster_pdf_requires_ocr", "frame_candidates": []}
                    report["routing"]["attempts"].append("source_inspection")
                else:
                    config, evidence = (
                        audited
                        if audited is not None
                        else propose_pdf(path, expected_type=expected)
                    )
                    report["routing"]["attempts"].append("native_pdf")
                if config is not None and not config.get("panels_1d"):
                    if (
                        config.get("histogram2d_panels")
                        and config.get("digitization_readiness") == "geometry_review_required"
                    ):
                        config["raster_extraction"] = {
                            "kind": "histogram_2d",
                            "method": "pixel_geometry_v1",
                            "reason": "native_cell_geometry_unavailable",
                        }
                    elif not config.get("histogram2d_panels") and not config.get("series"):
                        config["raster_extraction"] = {
                            "kind": "colored_traces",
                            "method": "pixel_geometry_v1",
                            "reason": "native_trace_selection_unavailable",
                        }
                report["evidence"] = evidence
                report["reason"] = evidence["reason"]
                fallback = evidence["reason"] == "native_numeric_text_unavailable" or (
                    report["source_inspection"]["representation"] in {"raster", "hybrid"} and config is None
                )
                if (
                    config is None
                    and fallback
                    and expected in {"line_plot", "histogram_1d", "histogram_2d"}
                ):
                    inputs.update(_ocr_model_inputs())
                    report["ocr_evidence"] = ocr_axis_candidates(path)
                    report["routing"].update(
                        attempts=report["routing"]["attempts"] + ["ocr"],
                        selected="raster",
                        fallback_reason=evidence["reason"],
                    )
                    config = raster_calibration_proposal(report["ocr_evidence"], expected)
                    if config is not None and report["source_inspection"]["representation"] != "raster":
                        config = _with_vector_series(path, config)
                        if not config.get("raster_extraction"):
                            report["routing"]["selected"] = "pymupdf"
                elif config is not None:
                    report["routing"]["selected"] = "raster" if config.get("raster_extraction") else "pymupdf"
                report["calibration_type"] = evidence.get("calibration_type", expected)
                report["routing_reason"] = evidence.get("routing_reason")
                panels = evidence.get("panels") or [evidence]
                axes = ["x", "y"] + (
                    ["z"] if evidence.get("calibration_type", expected) == "histogram_2d" else []
                )
                checks = [
                    {
                        "panel": i,
                        "axis": axis,
                        "matched_labels": len(panel.get(axis, {}).get("anchors", [])),
                        "attempted": axis in panel,
                        "verified": panel.get(axis, {}).get("reason") == "axis_ticks_consistent",
                        "reason": panel.get(axis, {}).get("reason", "not_established"),
                        "scale_evidence": panel.get(axis, {}).get("scale_evidence", "independent_pdf_fit"),
                    }
                    for i, panel in enumerate(panels, 1)
                    for axis in axes
                ]
                verified = bool(checks) and all(c["verified"] for c in checks)
                report["pdf_numeric_assessment"] = {"verified": verified, "axes": checks}
                report["gemma_pdf_disagreement"] = bool(gate and verified)
                if config is not None:
                    config.update(
                        figure_id=figure.figure_id, asset_id=asset["asset_id"], panel_id=asset.get("panel_id")
                    )
                    if not config.get("raster_extraction"):
                        config["automatic_raster_fallback"] = True
                    if report["class_role_conflict"]:
                        config["classification_review_required"] = True
                    configs.append(config)
                    report["status"] = (
                        "partial_proposed"
                        if config.get("calibration_completeness") == "partial"
                        else "proposed_class_conflict" if audited is not None else "proposed"
                    )
                    report["calibration_status"] = (
                        "some_panel_axes_calibrated"
                        if config.get("calibration_completeness") == "partial"
                        else "numeric_axes_calibrated"
                    )
                    if config.get("raster_extraction"):
                        # A class conflict stays visible: it, not the raster route, blocks digitization.
                        report.update(
                            status="proposed_class_conflict" if audited is not None else "proposed_raster",
                            reason="automatic_raster_calibration",
                            calibration_status="axes_calibrated_raster_extraction_required",
                            digitization_readiness="prototype_raster_backend",
                        )
                    report["digitization_readiness"] = config.get("digitization_readiness", "not_assessed")
                elif expected:
                    no_labels = any(
                        c["matched_labels"] == 0 and c["reason"] == "fewer_than_three_unambiguous_ticks"
                        for c in checks
                    )
                    report["status"] = "needs_review_axes" if not verified else "hint_mismatch"
                    if report["asset_role"] == "visual_comparison" and not verified:
                        report.update(status="skipped_class", reason="qualitative_visual_comparison")
                    if no_labels:
                        report["verification_limit"] = (
                            "No matched labels does not establish visual absence; outlined/raster labels may require OCR or manual review."
                        )
                    report["missing_requirements"] = (
                        evidence.get("missing_requirements")
                        or [
                            axis.upper() + ": " + evidence[axis]["reason"]
                            for axis in ("x", "y")
                            if evidence.get(axis, {}).get("reason") != "axis_ticks_consistent"
                            and evidence.get(axis, {}).get("reason")
                        ]
                        or [evidence["reason"]]
                    )
                    if report.get("ocr_evidence"):
                        report.update(
                            status="needs_review_ocr",
                            reason="outlined_label_ocr_candidates_unverified",
                            missing_requirements=[
                                "Native text is unavailable. OCR candidate labels and scale fits are shown below and require verification."
                            ],
                        )
            except (ValueError, OSError, RuntimeError) as exc:
                report.update(status="error", reason=str(exc))
            reports.append(report)
    write_json(
        output_dir / "calibration_proposals.json",
        {
            "schema_version": "1",
            "plots": configs,
            "classifications": [
                {
                    "figure_id": r["figure_id"],
                    "asset_id": r["asset_id"],
                    "panel_id": r.get("panel_id"),
                    "analysis_type": r["expected_type"],
                    "hint_source": r["hint_source"],
                }
                for r in reports
            ],
            "classified_manifest_sha256": sha256_file(inputs["classified_manifest"]),
        },
    )
    write_jsonl(output_dir / "calibration_diagnostics.jsonl", reports)
    summary = {
        "mode": "calibrate",
        "method": METHOD,
        "figures": len(figures),
        "assets": len(reports),
        "proposals": len(configs),
        "decision_mode": "automatic",
        "status_counts": dict(Counter(r["status"] for r in reports)),
        "reason_counts": dict(Counter(r["reason"] for r in reports)),
        "manual_review": "pending",
        "html_report": "review.html",
    }
    write_json(output_dir / "calibration_summary.json", summary)
    write_manifest(
        output_dir,
        step="extract_plots",
        params={
            "mode": "calibrate",
            "method": METHOD,
            "classified_dir": classified_dir,
            "parsed_dir": parsed_dir,
            "figure_ids": figure_ids,
            "asset_ids": asset_ids,
            "visual_predictions": visual_predictions,
            "max_step_drift_pdf_points": 0.1,
            "minimum_axis_ticks": 3,
        },
        inputs=inputs,
        summary=summary,
    )
    run_calibration_review(calibration_dir=output_dir, output_dir=output_dir, _inside_run=True)
    return summary
