"""Source overlays, source-render checks and independent-data comparisons."""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Any
from statistics import median

from .digitization import Axis, Calibration, finite


def structural_check(series: dict) -> None:
    x, y = series.get("x", []), series.get("y", [])
    if not x or len(x) != len(y) or not series.get("series_id"):
        raise ValueError("empty/misaligned numerical series")
    for value in [*x, *y]:
        if value is not None:
            finite(value)
    if any(v is None for v in x):
        raise ValueError("x coordinates must be explicit")
    if any(v is None for v in y) and series.get("geometry") != "band_envelopes":
        raise ValueError("missing central values require explicit band geometry")
    if series.get("geometry") == "histogram_2d_cells":
        for field in ("z", "z_ranges", "cell_bounds", "source_cell_rects"):
            if len(series.get(field, [])) != len(x):
                raise ValueError("misaligned 2D cells")
        for z in series["z"]:
            finite(z)
        for cell in series["cell_bounds"]:
            for axis in ("x", "y"):
                if not finite(cell[axis][0]) < finite(cell[axis][1]):
                    raise ValueError("invalid 2D cell bounds")
    for field in ("bins", "rendered_intervals", "displayed_uncertainties", "value_states"):
        if series.get(field) and len(series[field]) != len(x):
            raise ValueError(f"misaligned {field}")
    for bounds in series.get("bins") or []:
        if len(bounds) != 2 or finite(bounds[0]) >= finite(bounds[1]):
            raise ValueError("invalid bin boundaries")
    for bounds in series.get("rendered_intervals") or []:
        if len(bounds) != 2 or finite(bounds[0]) >= finite(bounds[1]):
            raise ValueError("invalid rendered interval")
    for component in series.get("displayed_uncertainties", []):
        for key in ("lower", "upper", "magnitude"):
            if component.get(key) is not None:
                finite(component[key])


def overlay(record: dict, work_dir: Path) -> dict:
    """SVG embeds a local source raster; marks are in original source coordinates."""
    from PIL import Image

    source = work_dir / "source.png"
    if not source.is_file():
        return {"status": "unavailable", "reason": "source rendering unavailable; no overlay claimed"}
    with Image.open(source) as image:
        width, height = image.size
    scale = record["raw_source"].get("render_scale", 1)
    data = base64.b64encode(source.read_bytes()).decode("ascii")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<title>{html.escape(record["data_id"])} — manual review pending</title>',
        f'<image width="{width}" height="{height}" href="data:image/png;base64,{data}"/>',
    ]

    def trace(points: list, color: str) -> None:
        encoded = " ".join(f"{finite(x)*scale},{finite(y)*scale}" for x, y in points)
        elements.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="1.5"/>')

    for i, series in enumerate(record["series"]):
        points = series.get("source_coordinates", [])
        color = ("#ff00ff", "#00aacc", "#dd5500")[i % 3]
        if series["geometry"] == "histogram_2d_cells":
            for left, top, right, bottom in series["source_cell_rects"]:
                trace([[left, top], [right, top], [right, bottom], [left, bottom], [left, top]], color)
        elif series["geometry"] == "rendered_step_intervals":
            for segment in series["source_horizontal_segments"]:
                trace(segment, color)
        elif series["geometry"] == "rendered_curve_samples":
            trace(points, color)
        elif series["geometry"] == "band_envelopes":
            for envelope in series["source_envelopes"]:
                trace(envelope, color)
        else:
            for x, y in points:
                elements.append(
                    f'<circle cx="{finite(x)*scale}" cy="{finite(y)*scale}" r="3" fill="none" stroke="{color}"/>'
                )
        cal = Calibration(record["calibration"])
        for j, component in enumerate(series.get("displayed_uncertainties", [])):
            if component.get("lower") is not None and component.get("upper") is not None:
                x = cal.x.position(series["x"][j])
                trace(
                    [[x, cal.y.position(component["lower"])], [x, cal.y.position(component["upper"])]], color
                )
        for j, bounds in enumerate(series.get("bins", [])):
            left, right = map(cal.x.position, bounds)
            top, bottom = cal.y.position(series["y"][j]), cal.y.position(series["baseline"])
            trace([[left, bottom], [left, top], [right, top], [right, bottom]], color)
    elements.append("</svg>")
    target = work_dir / "overlay.svg"
    target.write_text("\n".join(elements), encoding="utf-8")
    return {
        "status": "generated_manual_review_pending",
        "path": str(target.name),
        "kind": "source_geometry_overlay",
        "scientific_validity": "not_established",
        "data_id": record["data_id"],
    }


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": median(ordered) if ordered else None,
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None,
        "max": max(ordered) if ordered else None,
    }


def compatible(result: dict, reference: dict) -> bool:
    return all(
        result.get(key) is not None and result.get(key) == reference.get(key)
        for key in ("series_id", "observable", "coordinate_definition", "units")
    )


def compare_series(result: dict, reference: dict, comparison: dict) -> dict:
    """No row alignment: explicit point identity, unique nearest points or stated interpolation."""
    structural_check(result)
    structural_check(reference)
    if any(s.get("geometry") == "histogram_2d_cells" for s in (result, reference)):
        return {"status": "not_comparable", "reason": "independent 2D cell/Z benchmark not implemented"}
    if not compatible(result, reference):
        return {
            "status": "not_comparable",
            "reason": "observable, units, series or coordinate definition differs/missing",
        }
    if bool(result.get("bins")) != bool(reference.get("bins")):
        return {"status": "not_comparable", "reason": "binned and unbinned observables cannot be paired"}
    tolerance = comparison.get("tolerance", {})
    tx, ty = finite(tolerance.get("x", 0)), finite(tolerance.get("y", 0))
    if min(tx, ty) <= 0:
        raise ValueError("comparison requires positive physical x/y tolerances")
    method = comparison.get("method", "unique_nearest")
    pairs = []
    ambiguous = 0
    rx, ry = result["x"], result["y"]
    gx, gy = reference["x"], reference["y"]
    interpolated = None
    if method == "point_ids":
        a, b = result.get("point_ids", []), reference.get("point_ids", [])
        if len(a) != len(rx) or len(b) != len(gx) or len(set(a)) != len(a) or len(set(b)) != len(b):
            raise ValueError("point_ids matching requires complete unique known identities")
        lookup = {v: j for j, v in enumerate(b)}
        pairs = [(i, lookup[v]) for i, v in enumerate(a) if v in lookup]
    elif method == "linear_interpolation":
        domain = comparison.get("domain", [])
        if len(domain) != 2 or not finite(domain[0]) < finite(domain[1]):
            raise ValueError("curve comparison requires explicit common domain")
        if (
            min(len(rx), len(gx)) < 2
            or any(y is None for y in [*ry, *gy])
            or len(set(rx)) != len(rx)
            or len(set(gx)) != len(gx)
        ):
            raise ValueError("interpolation requires single-valued curves")
        if domain[0] < max(min(rx), min(gx)) or domain[1] > min(max(rx), max(gx)):
            raise ValueError("curve domain would require extrapolation")
        count = comparison.get("samples", 101)
        if type(count) is not int or not 2 <= count <= 10000:
            raise ValueError("curve samples must be 2–10000")

        def sample(xs: list, ys: list, points: list) -> list:
            from bisect import bisect_right

            ordered = sorted(zip(xs, ys))
            xx, yy = zip(*ordered)
            values = []
            for p in points:
                i = max(0, min(len(xx) - 2, bisect_right(xx, p) - 1))
                values.append(yy[i] + (yy[i + 1] - yy[i]) * (p - xx[i]) / (xx[i + 1] - xx[i]))
            return values

        grid = [domain[0] + i * (domain[1] - domain[0]) / (count - 1) for i in range(count)]
        ry, gy = sample(rx, ry, grid), sample(gx, gy, grid)
        rx = gx = grid
        pairs = list(zip(range(count), range(count)))
        interpolated = {
            "method": "piecewise_linear_in_physical_coordinates",
            "domain": domain,
            "samples": count,
            "point_detection_metrics_applicable": False,
        }
    elif method == "unique_nearest":
        if len(rx) * len(gx) > 2_000_000:
            return {
                "status": "not_comparable",
                "reason": "matching resource limit; use explicit IDs or curve domain",
            }
        # Reciprocal unique nearest neighbors abstain on ties, avoiding row-order matches.
        edges = []
        for i, (x, y) in enumerate(zip(rx, ry)):
            for j, (xx, yy) in enumerate(zip(gx, gy)):
                if y is None or yy is None:
                    if (
                        result.get("geometry") != reference.get("geometry")
                        or result.get("geometry") != "band_envelopes"
                    ):
                        continue
                    dy = 0
                else:
                    dy = abs(y - yy) / ty
                dx = (
                    max(abs(a - b) for a, b in zip(result["bins"][i], reference["bins"][j])) / tx
                    if result.get("bins")
                    else abs(x - xx) / tx
                )
                if max(dx, dy) <= 1:
                    edges.append((dx * dx + dy * dy, i, j))
        best_a = {}
        best_b = {}
        for score, i, j in sorted(edges):
            best_a.setdefault(i, []).append((score, j))
            best_b.setdefault(j, []).append((score, i))
        for i, candidates in best_a.items():
            score, j = candidates[0]
            reverse = best_b[j]
            tie = (len(candidates) > 1 and math.isclose(score, candidates[1][0], abs_tol=1e-14)) or (
                len(reverse) > 1 and math.isclose(reverse[0][0], reverse[1][0], abs_tol=1e-14)
            )
            if tie:
                ambiguous += 1
            elif reverse[0][1] == i:
                pairs.append((i, j))
    else:
        raise ValueError("unsupported comparison method")
    absolute = {"x": [], "y": []}
    relative = {"x": [], "y": []}
    near_zero = {"x": 0, "y": 0}
    endpoint = {"lower": [], "upper": []}
    bin_errors = []
    correct = 0
    for i, j in pairs:
        good = True
        for axis, a, b, tol in (("x", rx[i], gx[j], tx), ("y", ry[i], gy[j], ty)):
            if a is None or b is None:
                continue
            error = abs(a - b)
            absolute[axis].append(error)
            good = good and error <= tol
            if abs(b) > 1e-12:
                relative[axis].append(error / abs(b))
            else:
                near_zero[axis] += 1
        correct += int(good)
        if interpolated is None:
            ea, eb = result.get("displayed_uncertainties", []), reference.get("displayed_uncertainties", [])
            if ea and eb and ea[i].get("semantics") == eb[j].get("semantics"):
                for key in endpoint:
                    if ea[i].get(key) is not None and eb[j].get(key) is not None:
                        endpoint[key].append(abs(ea[i][key] - eb[j][key]))
            if result.get("bins") and reference.get("bins"):
                bin_errors.extend(abs(a - b) for a, b in zip(result["bins"][i], reference["bins"][j]))
    return {
        "status": "compared",
        "method": method,
        "pairs": pairs,
        "ambiguous_detections": ambiguous,
        "matched": len(pairs),
        "reference_points": len(gx),
        "detected_points": len(rx),
        "correct_fraction": correct / len(gx),
        "false_detections": len(rx) - len(pairs) if interpolated is None else None,
        "missing_fraction": 1 - len(pairs) / len(gx) if interpolated is None else None,
        "matching_bin_fraction": len(pairs) / len(gx) if bin_errors else None,
        "missing_bin_fraction": 1 - len(pairs) / len(gx) if bin_errors else None,
        "absolute_errors": {k: _summary(v) for k, v in absolute.items()},
        "relative_errors": {k: _summary(v) for k, v in relative.items()},
        "near_zero_relative_omitted": near_zero,
        "near_zero_policy": "absolute error only when abs(reference)<=1e-12",
        "endpoint_errors": {k: _summary(v) for k, v in endpoint.items()},
        "bin_edge_errors": _summary(bin_errors),
        "interpolation": interpolated,
        "manual_review": "pending",
    }


def original_series(record: dict, series: dict, mapping: dict) -> dict:
    """Normalize only verified 1D originals; retain their explicit units and error conventions."""
    from copy import deepcopy

    result = deepcopy(series)
    bins = result.get("bins") or []
    if record["backend"] == "native":
        units = {axis: record.get("axes", {}).get(axis, {}).get("unit") for axis in ("x", "y")}
    else:
        independent = series.get("independent_variables", [])
        if len(independent) != 1:
            raise ValueError("benchmark requires a one-dimensional original series")
        units = {
            "x": independent[0].get("header", {}).get("units"),
            "y": series.get("header", {}).get("units"),
        }
        if bins and all(b.get("low") is not None and b.get("high") is not None for b in bins):
            result["bins"] = [[b["low"], b["high"]] for b in bins]
            # Used only to position bin comparisons; matching is on both edges, not inferred measurements.
            result["x"] = [(a + b) / 2 for a, b in result["bins"]]
            result["coordinate_assumption"] = "bin midpoint for matching/display only"
        elif any(b.get("low") is not None or b.get("high") is not None for b in bins):
            raise ValueError("incomplete original bin boundaries")
        else:
            result["bins"] = None
    if units != mapping.get("units"):
        raise ValueError(
            "mapping units differ from explicitly published/native units; no implicit conversion"
        )
    result.update(
        series_id=mapping["series_id"],
        observable=mapping["observable"],
        units=units,
        coordinate_definition=mapping["coordinate_definition"],
    )
    components = series.get("uncertainties", [])
    result["displayed_uncertainties"] = []
    if components and record["backend"] == "native":
        if len(components) != 1:
            raise ValueError("select an unambiguous native error component")
        error = components[0]
        for i, y in enumerate(result["y"]):
            scale = abs(y) if error.get("relative") else 1
            lower = error["symerror"][i] if error.get("representation") == "symmetric" else error["minus"][i]
            upper = error["symerror"][i] if error.get("representation") == "symmetric" else error["plus"][i]
            result["displayed_uncertainties"].append(
                {
                    "lower": y - abs(lower) * scale,
                    "upper": y + abs(upper) * scale,
                    "semantics": "unlabeled_displayed_error",
                }
            )
    elif components:
        selected = mapping.get("uncertainty_label")
        for row, y in zip(components, result["y"]):
            choices = [c for c in row["components"] if c.get("label") == selected]
            if len(choices) != 1:
                result["displayed_uncertainties"] = []
                result["uncertainty_comparison"] = "not_selected_or_ambiguous_component"
                break
            error = choices[0]

            def delta(component: dict) -> float:
                value = component.get("value")
                if value is None:
                    raise ValueError("nonnumeric published error cannot be benchmarked")
                return finite(value) * (abs(y) / 100 if component.get("relative") else 1)

            if "symerror" in error:
                upper = abs(delta(error["symerror"]))
                lower = -upper
            else:
                lower = delta(error["asymerror"]["minus"])
                upper = delta(error["asymerror"]["plus"])
            result["displayed_uncertainties"].append(
                {
                    "lower": y + lower,
                    "upper": y + upper,
                    "semantics": error.get("kind") or "unlabeled_displayed_error",
                }
            )
    structural_check(result)
    return result


def _apply_source_axis_limits(ax: Any, calibration: Calibration) -> None:
    """Preserve displayed axis direction, including image coordinates increasing downwards."""
    left, top, right, bottom = calibration.config["roi"]
    ax.set_xlim(calibration.x.value(left), calibration.x.value(right))
    ax.set_ylim(calibration.y.value(bottom), calibration.y.value(top))


def _embed_review_resources(page: str, directory: Path) -> str:
    """Keep reports usable in IDE previews and when copied without companion folders."""
    import mimetypes
    import re
    from .assets import local_path

    def replace(match):
        attribute, url = match.groups()
        if url.startswith(("data:", "#")):
            return match.group(0)
        target = local_path(directory, directory / url)
        if not target.is_file():
            raise ValueError("missing report resource: " + url)
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
        download = (
            f' download="{html.escape(target.parent.name+"_"+target.name,quote=True)}"'
            if attribute == "href"
            else ""
        )
        return f'{attribute}="data:{mime};base64,{encoded}"' + download

    return re.sub(r'(src|href)="([^"]+)"', replace, page)


def run_digitization_review(*, digitized_dir: Path, output_dir: Path) -> dict:
    """Show source, numeric reconstruction and visual overlays without surrogate benchmarks."""
    from .records import series_axis_metadata
    import csv
    import json
    import xml.etree.ElementTree as ET
    import zipfile
    import numpy as np
    import pymupdf
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap, LogNorm, Normalize
    from PIL import Image
    from kamiknows.common import create_output_dir, read_manifest, sha256_file, write_json, write_manifest
    from .assets import local_path

    digitized_dir = Path(digitized_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(digitized_dir):
        raise ValueError("review output must be outside digitization run")
    manifest = read_manifest(digitized_dir, "extract_plots")
    data = digitized_dir / "digitized_data.jsonl"
    if manifest["params"].get("mode") != "digitize":
        raise ValueError("expected digitization run")
    if sha256_file(data) != manifest["summary"]["data_sha256"]:
        raise ValueError("digitized data hash mismatch")
    records = [json.loads(line) for line in data.read_text().splitlines()]
    output_dir = create_output_dir(output_dir)
    cards = []
    assets = []
    errors = []
    inputs = {"digitization_manifest": digitized_dir / "manifest.json", "digitized_data": data}
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 11})
    for record in records:
        identifier = record["data_id"]
        asset = record["provenance"]["asset_id"]
        if record["extraction_status"] != "success":
            errors.append(
                {
                    "asset_id": asset,
                    "status": record["extraction_status"],
                    "warnings": record.get("warnings", []),
                }
            )
            continue
        directory = local_path(output_dir, output_dir / identifier)
        directory.mkdir()
        try:
            source = local_path(digitized_dir, digitized_dir / "overlays" / identifier / "source.png")
            inputs["preview:" + identifier] = source
            with Image.open(source) as original:
                image = np.asarray(original.convert("RGB"), dtype=float) / 255
                original.convert("RGB").save(directory / "source.png")
                width, height = original.size
            scale = record["raw_source"]["render_scale"]
            cal = Calibration(record["calibration"])
            is2d = all(s["geometry"] == "histogram_2d_cells" for s in record["series"])
            fig, axes = plt.subplots(
                1,
                len(record["series"]) if is2d else 1,
                figsize=(7 * (len(record["series"]) if is2d else 1), 5),
                squeeze=False,
            )
            rows = []
            series_info = []
            numeric_svg = [
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><image width="{width}" height="{height}" href="data:image/png;base64,{base64.b64encode(source.read_bytes()).decode()}"/>'
            ]
            for index, series in enumerate(record["series"]):
                sid = series["series_id"]
                axis_metadata = series_axis_metadata(record, series)
                start = len(rows)
                mark_color = ["#00bf9a", "#f000d0", "#ff9900"][index % 3]
                if is2d:
                    cfg = series["axis_calibration"]
                    pc = Calibration(
                        {**cfg, "coordinate_system": record["calibration"]["coordinate_system"], "page": 1}
                    )
                    cl, ct, cr, cb = cfg["colorbar_roi"]
                    zaxis = Axis.fit(cfg["z"], (ct, cb))
                    palette = image[
                        int(ct * scale) + 1 : int(cb * scale) - 1, int(round((cl + cr) / 2 * scale)), :3
                    ]
                    if zaxis.value(ct) > zaxis.value(cb):
                        palette = palette[::-1]
                    low, high = sorted([zaxis.value(ct), zaxis.value(cb)])
                    norm = LogNorm(low, high) if cfg["z"]["scale"] == "log10" else Normalize(low, high)
                    vertices = []
                    for i, (bounds, z) in enumerate(zip(series["cell_bounds"], series["z"])):
                        xl, xr = bounds["x"]
                        yl, yh = bounds["y"]
                        vertices.append([(xl, yl), (xr, yl), (xr, yh), (xl, yh)])
                        rows.append(
                            {
                                "series_id": sid,
                                "element_index": i,
                                "kind": "displayed_cell",
                                "x": series["x"][i],
                                "y": series["y"][i],
                                "x_left": xl,
                                "x_right": xr,
                                "y_low": yl,
                                "y_high": yh,
                                "z": z,
                                "z_color_range_low": series["z_ranges"][i][0],
                                "z_color_range_high": series["z_ranges"][i][1],
                                "value_state": series["value_states"][i],
                            }
                        )
                        px = sorted([pc.x.position(xl) * scale, pc.x.position(xr) * scale])
                        py = sorted([pc.y.position(yl) * scale, pc.y.position(yh) * scale])
                        numeric_svg.append(
                            f'<rect x="{px[0]}" y="{py[0]}" width="{px[1]-px[0]}" height="{py[1]-py[0]}" fill="none" stroke="{mark_color}" stroke-width=".8"/>'
                        )
                    ax = axes[0, index]
                    collection = PolyCollection(
                        vertices,
                        array=np.asarray(series["z"]),
                        cmap=ListedColormap(palette),
                        norm=norm,
                        edgecolors="none",
                    )
                    ax.add_collection(collection)
                    fig.colorbar(collection, ax=ax, label="Z [" + axis_metadata["z"]["unit"] + "]")
                    ax.set(
                        xscale=pc.x.scale.replace("log10", "log"),
                        yscale=pc.y.scale.replace("log10", "log"),
                        xlabel="X [" + axis_metadata["x"]["unit"] + "]",
                        ylabel="Y [" + axis_metadata["y"]["unit"] + "]",
                        title="Reconstructed from extracted cell values",
                    )
                    _apply_source_axis_limits(ax, pc)
                else:
                    if "raster_color_rgb" in series:
                        color = series["raster_color_rgb"]
                    else:
                        spec = next(s for s in record["calibration"]["series"] if s["series_id"] == sid)
                        # A filled shape carries its colour in "fill"; a stroked one in "color".
                        path = record["raw_source"]["paths"][spec["path_indices"][0]]
                        color = path.get("color") or path.get("fill") or [0, 0, 0]
                    xs = []
                    ys = []
                    segments = []
                    for i, (x, y) in enumerate(zip(series["x"], series["y"])):
                        intervals = series.get("rendered_intervals", [])
                        left, right = intervals[i] if intervals else [x, x]
                        rows.append(
                            {
                                "series_id": sid,
                                "element_index": i,
                                "kind": "visible_interval" if intervals else "curve_sample",
                                "x": x,
                                "y": y,
                                "x_left": left,
                                "x_right": right,
                            }
                        )
                        if intervals:
                            if xs and abs(xs[-1] - left) > 1e-7:
                                xs.append(float("nan"))
                                ys.append(float("nan"))
                            xs.extend([left, right])
                            ys.extend([y, y])
                            numeric_svg.append(
                                f'<line x1="{cal.x.position(left)*scale}" y1="{cal.y.position(y)*scale}" x2="{cal.x.position(right)*scale}" y2="{cal.y.position(y)*scale}" stroke="{mark_color}" stroke-width="2"/>'
                            )
                        else:
                            xs.append(x)
                            ys.append(y)
                            segments.append(f"{cal.x.position(x)*scale},{cal.y.position(y)*scale}")
                    if segments:
                        numeric_svg.append(
                            f'<polyline points="{" ".join(segments)}" fill="none" stroke="{mark_color}" stroke-width="2"/>'
                        )
                    axes[0, 0].plot(xs, ys, color=color, label=sid)
                for row in rows[start:]:
                    for axis, metadata in axis_metadata.items():
                        row[axis + "_unit"] = metadata["unit"]
                        row[axis + "_unit_status"] = metadata["unit_status"]
                        row[axis + "_scale"] = metadata["scale"]
                series_info.append(
                    {
                        "series_id": sid,
                        "geometry": series["geometry"],
                        "elements": len(rows) - start,
                        "units": series.get("units", {}),
                        "axis_metadata": axis_metadata,
                        "original_bin_edges_verified": series.get("original_bin_edges_verified", False),
                    }
                )
            if not is2d:
                ax = axes[0, 0]
                ax.set(
                    xscale=cal.x.scale.replace("log10", "log"),
                    yscale=cal.y.scale.replace("log10", "log"),
                    xlabel="X [" + axis_metadata["x"]["unit"] + "]",
                    ylabel="Y [" + axis_metadata["y"]["unit"] + "]",
                    title="Reconstructed from extracted numbers",
                )
                ax.legend()
                _apply_source_axis_limits(ax, cal)
            fig.suptitle(asset, fontsize=10)
            fig.tight_layout()
            for ext in ["png", "svg", "pdf"]:
                fig.savefig(directory / ("reconstruction." + ext), dpi=180, bbox_inches="tight")
            plt.close(fig)
            numeric_svg.append("</svg>")
            (directory / "numeric_overlay.svg").write_text("".join(numeric_svg))
            with Image.open(directory / "source.png") as a, Image.open(directory / "reconstruction.png") as b:
                h = 600
                a = a.resize((round(a.width * h / a.height), h))
                b = b.resize((round(b.width * h / b.height), h))
                pair = Image.new("RGB", (a.width + b.width, h), "white")
                pair.paste(a, (0, 0))
                pair.paste(b, (a.width, 0))
                pair.save(directory / "side_by_side.png")
                ns = "http://www.w3.org/2000/svg"
                ET.register_namespace("", ns)
                root = ET.Element(
                    "{" + ns + "}svg",
                    {"width": str(pair.width), "height": str(h), "viewBox": f"0 0 {pair.width} {h}"},
                )
                ET.SubElement(
                    root,
                    "{" + ns + "}image",
                    {
                        "width": str(a.width),
                        "height": str(h),
                        "href": "data:image/png;base64," + base64.b64encode(source.read_bytes()).decode(),
                    },
                )
                vector = ET.parse(directory / "reconstruction.svg").getroot()
                vector.attrib.update(x=str(a.width), y="0", width=str(b.width), height=str(h))
                root.append(vector)
                ET.ElementTree(root).write(
                    directory / "side_by_side.svg", encoding="utf-8", xml_declaration=True
                )
            fields = list(dict.fromkeys(k for row in rows for k in row))
            with (directory / "extracted_values.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            info = {
                "asset_id": asset,
                "data_id": identifier,
                "series": series_info,
                "elements": len(rows),
                "assessment": "visual inspection only; no independent original-data benchmark",
            }
            write_json(directory / "extraction_summary.json", info)
            assets.append(info)
            esc = html.escape
            links = " · ".join(
                f'<a href="{identifier}/{name}">{label}</a>'
                for name, label in [
                    ("extracted_values.csv", "Extracted values CSV"),
                    ("side_by_side.svg", "Side-by-side SVG"),
                    ("reconstruction.svg", "Reconstruction SVG"),
                    ("reconstruction.pdf", "Reconstruction PDF"),
                ]
            )
            columns = ["series_id", "element_index", "kind", "x", "y"] + (["z"] if is2d else [])
            table = "<table><tr>" + "".join("<th>" + key + "</th>" for key in columns) + "</tr>"
            for row in rows[:20]:
                table += (
                    "<tr>"
                    + "".join(
                        "<td>"
                        + esc(f"{row[key]:.6g}" if isinstance(row.get(key), float) else str(row.get(key, "")))
                        + "</td>"
                        for key in columns
                    )
                    + "</tr>"
                )
            table += "</table>"
            units_html = "<h3>Axis units and scales</h3><table><tr><th>Series</th><th>Axis</th><th>Unit</th><th>Unit status</th><th>Scale</th><th>Applied display multiplier</th></tr>"
            for item in series_info:
                for axis, metadata in item["axis_metadata"].items():
                    units_html += (
                        "<tr>"
                        + "".join(
                            "<td>" + esc(str(value)) + "</td>"
                            for value in [
                                item["series_id"],
                                axis.upper(),
                                metadata["unit"],
                                metadata["unit_status"],
                                metadata["scale"],
                                metadata["display_multiplier"],
                            ]
                        )
                        + "</tr>"
                    )
            units_html += "</table><p>Values already include display multipliers. Log10 describes axis spacing; it does not change the physical unit. Unspecified does not mean dimensionless.</p>"
            cards.append(
                f'<article><h2>{esc(asset)}</h2><p>{len(series_info)} extracted series; {len(rows)} displayed elements. Visual review pending.</p>{units_html}<p>{links}</p><p>Left: source figure. Right: redrawn from extracted numbers.</p><img src="{identifier}/side_by_side.png" alt="Original and numerical reconstruction"><details><summary>Numeric-data overlay for visual inspection</summary><img src="{identifier}/numeric_overlay.svg" alt="Extracted coordinates projected onto source"></details><details><summary>First 20 extracted elements; all elements in CSV</summary>{table}</details></article>'
            )
        except (ValueError, OSError, KeyError, IndexError, RuntimeError) as exc:
            plt.close("all")
            errors.append({"asset_id": asset, "status": "review_error", "reason": str(exc)})
    failure_html = (
        "<details><summary>Assets not reconstructed ("
        + str(len(errors))
        + ")</summary><pre>"
        + html.escape(json.dumps(errors, indent=2))
        + "</pre></details>"
    )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Digitization visual review</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;background:#f4f6fa;color:#192438}article,header{background:white;border-radius:12px;padding:24px;margin:22px 0}img{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}a{color:#175bd5}</style><header><h1>Source and numerical reconstruction</h1><p>Visual inspection only. No independent original scientific data are available for numerical validation. Extracted values describe displayed geometry; original measurement points, bin edges and experimental uncertainties are not recovered automatically.</p><p>Calibration maps page coordinates to axis values, including units, scale and direction. Vector paths provide page geometry; they still need this mapping to become physical numbers. For 2D maps, the calibrated colorbar converts colors into estimated Z values.</p><p>Images and downloads are embedded. If an IDE blocks downloads, use a browser or the companion report.pdf and exports.zip.</p></header>"""
        + failure_html
        + "".join(cards)
        + "</html>"
    )
    (output_dir / "review.html").write_text(_embed_review_resources(page, output_dir))
    write_json(
        output_dir / "extraction_summary.json",
        {"assets": assets, "assessment": "visual inspection only; scientific accuracy not established"},
    )
    write_json(output_dir / "review_errors.json", errors)
    with pymupdf.open() as combined:
        for info in assets:
            picture = output_dir / info["data_id"] / "side_by_side.png"
            with Image.open(picture) as im:
                ph = 1000 * im.height / im.width
            sheet = combined.new_page(width=1000, height=ph + 48)
            sheet.insert_image(pymupdf.Rect(0, 0, 1000, ph), filename=str(picture))
            sheet.insert_text((16, ph + 18), info["asset_id"], fontsize=9)
            sheet.insert_text(
                (16, ph + 34),
                "Left: source figure. Right: numerical reconstruction. Visual inspection only; scientific accuracy not established.",
                fontsize=9,
            )
        if combined.page_count:
            combined.save(output_dir / "report.pdf")
    summary = {
        "mode": "digitization_review",
        "figures": len({r["figure_id"] for r in records}),
        "assets": len(records),
        "reported_assets": len(assets),
        "series": sum(len(a["series"]) for a in assets),
        "elements": sum(a["elements"] for a in assets),
        "review_errors": sum(e["status"] == "review_error" for e in errors),
        "extraction_failures": sum(e["status"] != "review_error" for e in errors),
        "assessment": "visual_inspection_only",
        "manual_review": "pending",
    }
    write_manifest(
        output_dir,
        step="extract_plots",
        params={
            "mode": "digitization_review",
            "method": "visual_reconstruction_v2",
            "digitized_dir": digitized_dir,
            "embedded_resources": True,
        },
        inputs=inputs,
        summary=summary,
    )
    with zipfile.ZipFile(output_dir / "exports.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for target in sorted(output_dir.rglob("*")):
            if target.is_file() and target.name not in {"exports.zip", "review.html"}:
                archive.write(target, target.relative_to(output_dir))
    return summary


def _readable_reason(reason: str) -> str:
    """Keep diagnostic codes, with an explanation for visual validation."""
    meanings = {
        "fewer_than_three_unambiguous_ticks": "fewer than three unambiguously matched numeric ticks",
        "unit_not_explicit_or_ambiguous": "unit missing or ambiguous",
        "ambiguous_or_inconsistent_scale": "linear/logarithmic scale ambiguous or inconsistent with the labels",
        "two_label_scale_requires_confirmation": "Two labels alone do not establish the scale; independent log tick-spacing evidence is unavailable",
        "nonmonotone_or_duplicate_labels": "duplicate or nonmonotone labels",
        "unique panel/colorbar pairing": "unique panel/colorbar association not found",
        "partially_transparent_cell_colors": "partially transparent cells: color does not identify a unique Z value",
        "no_unique_rectangular_cell_grid": "rectangular cell grid not uniquely identified",
        "axis_ticks_consistent": "consistent ticks and scale; unit status recorded separately",
        "non_quantitative_class": "diagram, image or mixed asset: numerical calibration skipped",
    }
    for code, meaning in meanings.items():
        if reason == code or reason.endswith(": " + code):
            prefix = reason[: -len(code)]
            return prefix + meaning + " [" + code + "]"
    return reason


def run_calibration_review(
    *, calibration_dir: Path, output_dir: Path, _inside_run: bool = False
) -> dict[str, Any]:
    """Write a standalone visual audit without changing the calibration run."""
    from collections import Counter
    from .assets import local_path
    from .digitization import prepare_pdf_orientation
    from kamiknows.common import (
        create_output_dir,
        read_jsonl,
        sha256_file,
        write_json,
        write_jsonl,
        write_manifest,
    )
    import base64
    import html
    import io
    import json
    import pymupdf
    from PIL import Image
    from kamiknows.common import read_manifest

    calibration_dir = Path(calibration_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(calibration_dir) and not _inside_run:
        raise ValueError("review output must be outside calibration run")
    manifest = read_manifest(calibration_dir, "extract_plots")
    if manifest["params"].get("mode") != "calibrate":
        raise ValueError("expected calibration run")
    diagnostics = calibration_dir / "calibration_diagnostics.jsonl"
    records = read_jsonl(diagnostics)
    inputs = {"calibration_manifest": calibration_dir / "manifest.json", "diagnostics": diagnostics}
    cards = []
    rendered = 0
    errors = []
    esc = lambda x: html.escape(str(x), quote=True)
    for record in records:
        asset_id = record["asset_id"]
        evidence = record.get("evidence", {})
        svg = ""
        try:
            original = manifest["inputs"][asset_id]
            path = Path(original["path"])
            if sha256_file(path) != original["sha256"]:
                raise ValueError("source hash mismatch")
            inputs[asset_id] = path
            if path.suffix.lower() == ".pdf":
                with pymupdf.open(path) as doc:
                    page = doc[0]
                    prepare_pdf_orientation(page, evidence.get("pdf_orientation", "unrotated"))
                    width, height = page.rect.width, page.rect.height
                    scale = min(1.8, 1200 / max(width, height))
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    width, height = image.size
                image.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=83)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            parts = [
                f'<svg role="img" aria-label="{esc(asset_id)}" viewBox="0 0 {width} {height}">',
                f'<image width="{width}" height="{height}" href="data:image/jpeg;base64,{data}"/>',
                '<g class="marks">',
            ]
            for index, observation in enumerate(record.get("ocr_evidence", {}).get("observations", []), 1):
                x0, y0, x1, y1 = observation["box"]
                parts.append(
                    f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#0077cc" stroke-width=".8"><title>OCR {index}: {esc(observation["text"])} (unverified)</title></rect>'
                )
            for index, box in enumerate(evidence.get("frame_candidates", [])[:20], 1):
                x0, y0, x1, y1 = box
                parts.append(
                    f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#db2777" stroke-width="1.5"/>'
                )
                parts.append(f'<text x="{x0+3}" y="{y0+10}" fill="#db2777" font-size="9">R{index}</text>')
            panels = evidence.get("panels") or (
                [{**evidence, "roi": evidence["frame_candidates"][0]}]
                if len(evidence.get("frame_candidates", [])) == 1
                else []
            )
            for panel_index, panel in enumerate(panels, 1):
                left, top, right, bottom = panel["roi"]
                if panel.get("colorbar_roi"):
                    cl, ct, cr, cb = panel["colorbar_roi"]
                    parts.append(
                        f'<rect x="{cl}" y="{ct}" width="{cr-cl}" height="{cb-ct}" fill="none" stroke="#6236cc" stroke-width="2"/>'
                    )
                for axis, color in [("x", "#008060"), ("y", "#d35400"), ("z", "#6236cc")]:
                    info = panel.get(axis, {})
                    for index, (anchor, label) in enumerate(
                        zip(info.get("anchors", []), info.get("labels", [])), 1
                    ):
                        if axis == "z":
                            x, y = panel["colorbar_roi"][2], anchor["position"]
                        else:
                            x, y = (anchor["position"], bottom) if axis == "x" else (left, anchor["position"])
                        bx0, by0, bx1, by1 = label["box"]
                        parts.extend(
                            [
                                f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>',
                                f'<line x1="{x}" y1="{y}" x2="{(bx0+bx1)/2}" y2="{(by0+by1)/2}" stroke="{color}" stroke-width=".7"/>',
                                f'<rect x="{bx0}" y="{by0}" width="{bx1-bx0}" height="{by1-by0}" fill="none" stroke="{color}" stroke-width=".6"/>',
                                f'<text x="{x+4}" y="{y-4}" fill="{color}" font-size="8">P{panel_index}{axis.upper()}{index}</text>',
                            ]
                        )
            parts.append("</g></svg>")
            svg = "".join(parts)
            rendered += 1
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            errors.append({"asset_id": asset_id, "error": str(exc)})
            svg = f'<p class="error">Image could not be verified: {esc(exc)}</p>'
        axis_tables = []
        panels = evidence.get("panels") or [evidence]
        for panel_index, panel in enumerate(panels, 1):
            for axis in ["x", "y", "z"]:
                if axis == "z" and "z" not in panel:
                    continue
                info = panel.get(axis, {})
                if not info:
                    continue
                rows = "".join(
                    f'<tr><td>P{panel_index}{axis.upper()}{i}</td><td>{esc(label["text"])}</td><td>{anchor["position"]:.4f}</td><td>{anchor["value"]:.7g}</td></tr>'
                    for i, (anchor, label) in enumerate(
                        zip(info.get("anchors", []), info.get("labels", [])), 1
                    )
                )
                multiplier_html = (
                    f'<p>Axis-wide multiplier: {info["display_multiplier"]:g}, already applied to the reported values.</p>'
                    if info.get("display_multiplier", 1) != 1
                    else ""
                )
                axis_tables.append(
                    multiplier_html
                    + f'<h4>Panel {panel_index}, axis {axis.upper()} — {esc(info.get("scale","unresolved scale"))}</h4><p>{esc(_readable_reason(info.get("reason","not analyzed")))}</p>'
                    f"<table><tr><th>Mark</th><th>Read label</th><th>PDF position</th><th>Value</th></tr>{rows}</table>"
                )
        candidate_note = (
            f'<p>{len(evidence.get("frame_candidates",[]))} unverified frame candidates; only the first 20 are marked. All candidates remain in the diagnostic details.</p>'
            if len(evidence.get("frame_candidates", [])) > 20
            else ""
        )
        missing = record.get("missing_requirements") or evidence.get("missing_requirements", [])
        counts = Counter(missing)
        missing_html = (
            '<p class="error">Missing requirements: '
            + esc(
                "; ".join(
                    _readable_reason(m) + (f" ({count} occurrences; see panel details)" if count > 1 else "")
                    for m, count in counts.items()
                )
            )
            + "</p>"
            if missing
            else ""
        )
        visual_html = ""
        if record.get("ocr_evidence"):
            ocr = record["ocr_evidence"]
            visual_html += "<h4>OCR observations — unverified</h4><p>" + esc(ocr["limitation"]) + "</p>"
            visual_html += "<details><summary>Recognized text and OCR scores (not verification confidence)</summary><table><tr><th>Mark</th><th>Text</th><th>OCR score</th></tr>"
            visual_html += (
                "".join(
                    f'<tr><td>{i}</td><td>{esc(o["text"])}</td><td>{o["score"]:.3f}</td></tr>'
                    for i, o in enumerate(ocr["observations"], 1)
                )
                + "</table></details>"
            )
            for i, panel in enumerate(ocr.get("panels", []), 1):
                for axis in ["x", "y"] + (["z"] if "z" in panel else []):
                    info = panel.get(axis, {})
                    visual_html += f'<p>OCR candidate P{i} {axis.upper()}: {esc(info.get("scale","unresolved"))}; {len(info.get("anchors",[]))} matched labels; {esc(info.get("reason","not analyzed"))}. Unverified.</p>'
        if record.get("visual_axis_observations"):
            visual_html += "<h4>Gemma visual observations (unverified)</h4>"
            for i, region in enumerate(record["visual_axis_observations"], 1):
                observed = "; ".join(
                    axis + ": " + ", ".join(k + "=" + v for k, v in observation.items())
                    for axis, observation in region["axes"].items()
                )
                visual_html += f'<p>Visual region {i}: {esc(region["analysis_type"])} — {esc(observed)}</p>'
        if record.get("visual_description"):
            description = record["visual_description"]["content"]
            visual_html += (
                "<h4>Model-generated visual description — human review pending</h4><p>"
                + esc(description["visible_content"])
                + "</p>"
            )
            visual_html += "<p>Visible labels: " + esc("; ".join(description["visible_labels"])) + "</p>"
            visual_html += "<p>Limitations: " + esc("; ".join(description["limitations"])) + "</p>"
        independent_html = ""
        if record.get("class_role_conflict"):
            independent_html += '<p class="error">Conflicting Gemma role and quantitative class. Numeric PDF evidence is checked independently; representation requires review. Neither model claim is treated as verified.</p>'
        if record.get("routing_reason"):
            independent_html += f'<p>PDF calibration representation: {esc(record["calibration_type"])}. Routing evidence: {esc(record["routing_reason"])}. Original Gemma class is retained above.</p>'
        if record.get("routing"):
            independent_html += (
                "<h4>Automatic source routing</h4><p>"
                + esc(json.dumps(record["source_inspection"]))
                + "</p><p>"
                + esc(json.dumps(record["routing"]))
                + "</p>"
            )
        if record.get("pdf_numeric_assessment"):
            assessment = record["pdf_numeric_assessment"]
            independent_html += (
                "<h4>Independent PDF verification</h4><p>"
                + (
                    "Numeric labels and scales verified"
                    if assessment["verified"]
                    else "Some numeric requirements could not be verified"
                )
                + "</p>"
            )
            for check in assessment["axes"]:
                if not check.get("attempted", True):
                    independent_html += f'<p>P{check["panel"]} {check["axis"].upper()}: not attempted — panel layout was not established.</p>'
                    continue
                independent_html += f'<p>P{check["panel"]} {check["axis"].upper()}: {check["matched_labels"]} matched labels — {esc(_readable_reason(check["reason"]))}</p>'
            if record.get("gemma_pdf_disagreement"):
                independent_html += '<p class="error">Disagreement: the PDF verifies numeric labels that Gemma reports as absent or uncertain. Gemma does not veto PDF verification.</p>'
        if record.get("calibration_status"):
            independent_html += f'<h4>Calibration and digitization preparation</h4><p>Calibration: {esc(record["calibration_status"])}. Digitization preparation: {esc(record["digitization_readiness"])}. No digitization is implied.</p>'
        status = record["status"]
        figure = record["figure_id"]
        cards.append(
            (
                status,
                f'<article data-status="{esc(status)}" data-search="{esc((figure+" "+asset_id+" "+record["reason"]).lower())}">'
                f'<h3>{esc(figure)}</h3><p>{esc(asset_id)}</p><p><b>{esc(status)}</b> · {esc(record["reason"])}</p><p>Suggested class: {esc(record.get("expected_type","unspecified"))}</p>{candidate_note}{missing_html}{visual_html}{independent_html}'
                f'<div class="layout"><div>{svg}</div><div>{"".join(axis_tables)}</div></div>'
                f"<details><summary>Details and Gemma context</summary><pre>{esc(json.dumps(record,indent=2,ensure_ascii=False))}</pre></details></article>",
            )
        )
    groups = [
        ("proposed", "Proposals requiring validation", "proposed"),
        (
            "proposed_class_conflict",
            "Numeric calibration candidates — model class requires review",
            "class-conflict-proposals",
        ),
        ("partial_proposed", "Partially calibrated assets — review remaining panels", "partial-proposals"),
        ("hint_mismatch", "Suggested class: requirements not established", "hint-mismatches"),
        (
            "excluded_no_numbers",
            "Excluded: Gemma reports absence and PDF labels could not be matched",
            "no-numbers",
        ),
        ("needs_review_axes", "Review required: numeric requirements not verified", "uncertain-numbers"),
        (
            "needs_raster_calibration",
            "Raster images — OCR or manual calibration required",
            "raster-calibration",
        ),
        (
            "needs_review_classification",
            "Classification suggestions requiring review",
            "classification-review",
        ),
        ("needs_review", "Other cases requiring review", "unresolved"),
        ("skipped_reviewed", "Human-reviewed numeric exclusions", "reviewed-exclusions"),
        ("skipped_class", "Classes excluded from numerical calibration", "skipped"),
        ("error", "Processing errors", "errors"),
    ]
    groups.insert(0, ("needs_review_ocr", "OCR candidates requiring numeric verification", "ocr-review"))
    groups.insert(
        0, ("proposed_raster", "Automatic pixel-extraction proposals — review pending", "raster-proposals")
    )
    sections = ""
    for status, title, identifier in groups:
        content = "".join(card for state, card in cards if state == status)
        count = sum(state == status for state, card in cards)
        if status in {"skipped_reviewed", "skipped_class"}:
            sections += f'<section id="{identifier}"><details><summary>{title} ({count}) — excluded from numeric work; not calibration failures</summary>{content}</details></section>'
        else:
            sections += f'<section id="{identifier}"><h2>{title} ({count})</h2>{content}</section>'

    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual calibration review</title><style>
body{font:15px system-ui;margin:0;background:#f3f5f7;color:#17212b}header{padding:20px;background:white;position:sticky;top:0;z-index:2;border-bottom:1px solid #ccc}
main{max-width:1500px;margin:auto;padding:16px}article{background:white;padding:18px;margin:16px 0;border:1px solid #ddd;border-radius:8px}h3{margin:0;overflow-wrap:anywhere}
.layout{display:grid;grid-template-columns: minmax(0,3fr) minmax(260px,1fr);gap:20px}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px;border:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}input,select{padding:8px;margin:4px}body.hide-marks .marks{display:none}.error{color:#a00}article[hidden]{display:none}section{scroll-margin-top:260px}@media(max-width:850px){.layout{grid-template-columns:1fr}}
</style><header><b>Calibration: visual review</b><p>Pink rectangles: candidates. Green: X ticks. Orange: Y ticks. Purple: colorbar and Z ticks. Lines connect ticks to extracted labels.</p>
<p>A proposal is not a verified calibration. Check positions, values, scales and units. Human review pending.</p>
<input id="search" placeholder="Search paper, asset or reason"><select id="status"><option value="">All statuses</option><option>proposed</option><option>skipped_reviewed</option><option>proposed_class_conflict</option><option>needs_raster_calibration</option><option>needs_review_ocr</option><option>needs_review_classification</option><option>partial_proposed</option><option>excluded_no_numbers</option><option>needs_review_axes</option><option>needs_review</option><option>hint_mismatch</option><option>skipped_class</option><option>error</option></select>
<label><input type="checkbox" id="marks" checked>Show detections</label><span id="count"></span> · <a href="#hint-mismatches">Jump to unmet requirements for the suggested class</a></header><main>"""
        + sections
        + """</main>
<script>const cards=[...document.querySelectorAll('article')];function filter(){const q=document.querySelector('#search').value.toLowerCase();const s=document.querySelector('#status').value;let n=0;for(const c of cards){c.hidden=!!((s&&c.dataset.status!==s)||!c.dataset.search.includes(q));if(!c.hidden)n++}document.querySelector('#count').textContent=n+' / '+cards.length+' asset';}document.querySelector('#search').addEventListener('input',filter);document.querySelector('#status').addEventListener('change',filter);document.querySelector('#marks').addEventListener('change',e=>document.body.classList.toggle('hide-marks',!e.target.checked));filter();</script></html>"""
    )
    if not _inside_run:
        output_dir = create_output_dir(output_dir)
    (output_dir / "review.html").write_text(page, encoding="utf-8")
    summary = {
        "mode": "calibration_review",
        "assets": len(records),
        "rendered": rendered,
        "render_errors": len(errors),
        "manual_review": "pending",
    }
    write_json(output_dir / "review_summary.json", summary)
    write_jsonl(output_dir / "review_errors.jsonl", errors)
    if not _inside_run:
        write_manifest(
            output_dir,
            step="extract_plots",
            params={
                "mode": "calibration_review",
                "calibration_dir": calibration_dir,
                "pdf_page": 1,
                "image_quality": 83,
            },
            inputs=inputs,
            summary=summary,
        )
    return summary
