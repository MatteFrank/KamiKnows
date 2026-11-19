"""Selection: take part of a series, or turn its coordinates into values."""

from __future__ import annotations

import numpy as np

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.geometry import containing
from kamiknows.compute.types import Axis, Quantity, Series, expr, to_unit

WHICH = ("center", "lower", "upper")


def restrict(series: Series, axis: str, lower: float | Quantity | None = None,
             upper: float | Quantity | None = None) -> Series:
    """Keep the elements whose coordinate (domain axis) or value (value axis) lies in [lower, upper].

    On a domain axis an element is kept when its center is in range, so a bin
    straddling a limit is kept or dropped whole. A missing limit is unbounded.
    """
    unit = series.axis(axis).unit
    low = -np.inf if lower is None else to_unit(lower, unit)
    high = np.inf if upper is None else to_unit(upper, unit)
    if low > high:
        raise NotComputable(f"empty range [{low:g}, {high:g}] on {axis}")
    if axis == series.value_axis.name:
        column = series.values
    else:
        column = series.centers[:, series.domain_index(axis)]
    keep = (column >= low) & (column <= high)
    return series.take(keep, f"restrict({series.origin}, {axis}, {expr(lower)}, {expr(upper)})")


def section(series: Series, axis: str, at: float | Quantity) -> Series:
    """Cut a 2D series at ``axis = at``.

    The cells whose support contains ``at`` become a 1D series over the other
    axis. Point samples have no support to cut through, so they are refused.
    """
    if series.dim != 2:
        raise NotComputable(f"section needs a 2D series; {series.origin} is 1D")
    if not series.has_support:
        raise NotComputable(f"section needs cells with bounds; {series.origin} has point samples")
    k = series.domain_index(axis)
    position = to_unit(at, series.domain[k].unit)
    keep = containing(series.lower[:, k], series.upper[:, k], position)
    if not keep.any():
        raise NotComputable(f"no element of {series.origin} contains {axis} = {position:g}")
    other = [1 - k]
    return series.derive(
        f"section({series.origin}, {axis}, {expr(at)})",
        domain=(series.domain[1 - k],),
        centers=series.centers[keep][:, other],
        values=series.values[keep],
        lower=series.lower[keep][:, other],
        upper=series.upper[keep][:, other],
    )


def coordinate(series: Series, axis: str, which: str = "center") -> Series:
    """A series on the same elements whose values are their coordinates on ``axis``.

    ``which`` picks the center or, for bins and cells, the lower or upper
    bound. Combined with combine and reduce it yields widths (upper - lower),
    moments, centroids and least-squares fits without dedicated operations.
    """
    if which not in WHICH:
        raise NotComputable(f"unknown coordinate {which!r}; use one of {WHICH}")
    k = series.domain_index(axis)
    if which != "center" and not series.has_support:
        raise NotComputable(f"{series.origin} has point samples: only centers exist")
    column = {"center": series.centers, "lower": series.lower, "upper": series.upper}[which][:, k]
    domain_axis = series.domain[k]
    return series.derive(
        f"coordinate({series.origin}, {axis}, {which})",
        values=column,
        value_axis=Axis(series.value_axis.name, domain_axis.unit, domain_axis.scale),
    )
