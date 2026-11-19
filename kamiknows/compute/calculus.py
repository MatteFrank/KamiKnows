"""Calculus on 1D series: slopes and running integrals. 2D series are cut with section first."""

from __future__ import annotations

import numpy as np

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.geometry import order_1d
from kamiknows.compute.types import Axis, Series


def differentiate(series: Series) -> Series:
    """Slope between consecutive elements, placed at the midpoint of their centers.

    The result has one element fewer and is a set of point samples; its unit
    is the value unit divided by the coordinate unit.
    """
    order = order_1d(series, "differentiate")
    if series.size < 2:
        raise NotComputable(f"differentiate needs at least two elements ({series.origin})")
    x, v = series.centers[order, 0], series.values[order]
    steps = np.diff(x)
    if (steps <= 0).any():
        raise NotComputable(f"two elements of {series.origin} share a coordinate")
    axis = series.domain[0]
    return Series(
        series.series_id, (axis,), Axis(series.value_axis.name, series.value_axis.unit / axis.unit),
        centers=((x[:-1] + x[1:]) / 2)[:, None], values=np.diff(v) / steps,
        origin=f"differentiate({series.origin})", caveats=series.caveats,
    )


def cumulate(series: Series) -> Series:
    """Running integral from the lowest coordinate, as point samples.

    For bins the running integral is exact and piecewise linear: samples sit
    on the bin edges (flat across gaps between bins), so evaluate and solve
    on the result interpolate it without error, e.g. a median is
    ``solve(cumulate(s), integral / 2)``. For point samples it is the
    trapezoidal running integral at the same coordinates.
    """
    order = order_1d(series, "cumulate")
    axis = series.domain[0]
    if series.has_support:
        x, total = _bin_running_integral(series, order)
    else:
        x, v = series.centers[order, 0], series.values[order]
        total = np.concatenate([[0.0], np.cumsum(np.diff(x) * (v[1:] + v[:-1]) / 2)])
    return Series(
        series.series_id, (axis,), Axis(series.value_axis.name, series.value_axis.unit * axis.unit),
        centers=np.asarray(x)[:, None], values=np.asarray(total),
        origin=f"cumulate({series.origin})", caveats=series.caveats,
    )


def _bin_running_integral(series: Series, order: np.ndarray) -> tuple[list[float], list[float]]:
    x, total = [float(series.lower[order[0], 0])], [0.0]
    for i in order:
        low, high = float(series.lower[i, 0]), float(series.upper[i, 0])
        tolerance = 1e-9 * max(1.0, abs(x[-1]))
        if low < x[-1] - tolerance or high <= x[-1]:
            raise NotComputable(f"bins of {series.origin} overlap: no single running integral")
        if low > x[-1] + tolerance:
            x.append(low)
            total.append(total[-1])
        x.append(high)
        total.append(total[-1] + float(series.values[i]) * (high - max(low, x[-2])))
    return x, total
