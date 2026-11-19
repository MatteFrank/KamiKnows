"""Metadata of a series: what a question about axes, units or coverage needs, without computing."""

from __future__ import annotations

from typing import Any

from kamiknows.compute.types import Series


def describe(series: Series) -> dict[str, Any]:
    """Axes (unit, scale, range), number of elements, whether they are bins, and caveats of a series."""
    axes: dict[str, Any] = {}
    for k, axis in enumerate(series.domain):
        axes[axis.name] = {"role": "domain", "unit": str(axis.unit), "scale": axis.scale}
        if series.size:
            low = series.lower[:, k] if series.has_support else series.centers[:, k]
            high = series.upper[:, k] if series.has_support else series.centers[:, k]
            axes[axis.name]["range"] = [float(low.min()), float(high.max())]
    value = series.value_axis
    axes[value.name] = {"role": "value", "unit": str(value.unit), "scale": value.scale}
    if series.size:
        axes[value.name]["range"] = [float(series.values.min()), float(series.values.max())]
    return {
        "series": series.origin,
        "axes": axes,
        "elements": series.size,
        "elements_are": "bins or cells" if series.has_support else "point samples",
        "caveats": list(series.caveats),
    }
