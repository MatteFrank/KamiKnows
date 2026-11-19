"""Digitized plots -> catalog entries.

The only module that reads the digitization schema (``digitized_data.jsonl``
written by ``extract_plots``): when that schema changes, nothing else in
``compute`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from kamiknows.common import read_jsonl, read_manifest
from kamiknows.compute.catalog import Catalog, Entry
from kamiknows.compute.types import Axis, Series
from kamiknows.compute.units import UNKNOWN, Unit

DATA_FILE = "digitized_data.jsonl"
DISPLAYED_VALUES = "values estimate the displayed figure, not the original measurements"


def series_handle(asset_id: str, series_id: str | None) -> str:
    """Stable name of one plotted series; an asset without series is named alone."""
    return asset_id if series_id is None else f"{asset_id}/{series_id}"


def load_plots(digitized_dir: Path) -> Catalog:
    """Read one digitization run into a catalog: one entry per series, or per undigitized asset."""
    read_manifest(digitized_dir, "extract_plots")
    return Catalog(e for record in read_jsonl(Path(digitized_dir) / DATA_FILE) for e in record_entries(record))


def record_entries(record: dict[str, Any]) -> list[Entry]:
    """One entry per series of a digitized record, or one asset entry saying why there is none."""
    asset_id = record["provenance"]["asset_id"]
    source = {"paper_id": record["paper_id"], "figure_id": record["figure_id"], "asset_id": asset_id}
    status = record.get("extraction_status")
    if status != "success" or not record.get("series"):
        reason = f"plot not digitized (extraction status: {status})"
        return [Entry(series_handle(asset_id, None), None, reason, source)]
    entries = []
    for raw in record["series"]:
        handle = series_handle(asset_id, raw["series_id"])
        meta = {**source, "series_id": raw["series_id"]}
        builder = GEOMETRIES.get(raw.get("geometry"))
        if builder is None:
            entries.append(Entry(handle, None, f"geometry {raw.get('geometry')!r} is not supported", meta))
        else:
            entries.append(Entry(handle, builder(handle, record, raw), None, meta))
    return entries


def _cells(handle: str, record: dict[str, Any], raw: dict[str, Any]) -> Series:
    """Heatmap cells: 2D domain with cell bounds, value from the colour scale."""
    bounds = raw["cell_bounds"]
    return Series(
        handle,
        domain=(_axis("x", record, raw), _axis("y", record, raw)),
        value_axis=_axis("z", record, raw),
        centers=np.column_stack([raw["x"], raw["y"]]),
        values=np.asarray(raw["z"]),
        lower=np.asarray([[b["x"][0], b["y"][0]] for b in bounds]),
        upper=np.asarray([[b["x"][1], b["y"][1]] for b in bounds]),
        origin=handle,
        caveats=_caveats(raw),
    )


def _steps(handle: str, record: dict[str, Any], raw: dict[str, Any]) -> Series:
    """1D histogram steps: each drawn horizontal step is a bin."""
    intervals = np.asarray(raw["rendered_intervals"], dtype=float)
    return Series(
        handle,
        domain=(_axis("x", record, raw),),
        value_axis=_axis("y", record, raw),
        centers=np.asarray(raw["x"], dtype=float)[:, None],
        values=np.asarray(raw["y"]),
        lower=intervals[:, :1],
        upper=intervals[:, 1:],
        origin=handle,
        caveats=_caveats(raw),
    )


def _curve(handle: str, record: dict[str, Any], raw: dict[str, Any]) -> Series:
    """A drawn line: point samples, interpolated along the line as drawn."""
    return Series(
        handle,
        domain=(_axis("x", record, raw),),
        value_axis=_axis("y", record, raw),
        centers=np.asarray(raw["x"], dtype=float)[:, None],
        values=np.asarray(raw["y"]),
        origin=handle,
        caveats=_caveats(raw),
    )


GEOMETRIES: dict[str, Callable[[str, dict[str, Any], dict[str, Any]], Series]] = {
    "histogram_2d_cells": _cells,
    "rendered_step_intervals": _steps,
    "rendered_curve_samples": _curve,
}
BINNED = {"histogram_2d_cells", "rendered_step_intervals"}


def _axis(name: str, record: dict[str, Any], raw: dict[str, Any]) -> Axis:
    """Axis from the series' own calibration, falling back to the record axes."""
    calibration = (raw.get("axis_calibration") or {}).get(name) or _record_axis(record, name)
    unit = raw["units"].get(name)
    if unit in (None, "unknown") or calibration.get("unit_status") == "missing":
        return Axis(name, UNKNOWN, calibration["scale"])
    return Axis(name, Unit.parse(unit), calibration["scale"])


def _record_axis(record: dict[str, Any], name: str) -> dict[str, Any]:
    axes = record.get("axes") or {}
    if name in axes:
        return axes[name]
    for panel in axes.get("panels") or []:
        if name in panel:
            return panel[name]
    calibration = record.get("calibration") or {}
    if name in calibration:
        return calibration[name]
    raise ValueError(f"{record['provenance']['asset_id']}: no scale recorded for axis {name!r}")


def _caveats(raw: dict[str, Any]) -> tuple[str, ...]:
    caveats = [DISPLAYED_VALUES]
    if raw.get("geometry") in BINNED and not raw.get("original_bin_edges_verified", False):
        caveats.append("bin edges are the drawn ones; the original binning is unverified")
    if "color_inverted_estimate" in raw.get("value_states", ()):
        caveats.append("values are read from the colour scale")
    return tuple(caveats)
