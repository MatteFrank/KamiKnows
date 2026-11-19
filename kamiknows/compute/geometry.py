"""Geometric helpers shared by the operations: ordering, containment, display space."""

from __future__ import annotations

import numpy as np

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.types import Scale, Series


def order_1d(series: Series, operation: str) -> np.ndarray:
    """Indices that sort a non-empty 1D series by coordinate."""
    if series.dim != 1:
        raise NotComputable(f"{operation} needs a 1D series; take a section of {series.origin} first")
    if series.size == 0:
        raise NotComputable(f"{operation} of an empty series ({series.origin})")
    return np.argsort(series.centers[:, 0], kind="stable")


def containing(lower: np.ndarray, upper: np.ndarray, position: float) -> np.ndarray:
    """Mask of the supports that contain ``position``.

    Supports are half-open [lower, upper), so a position on a shared edge
    belongs to one element only; the overall top edge is closed, so the last
    bin keeps its upper limit.
    """
    mask = (lower <= position) & (position < upper)
    return mask | ((position == upper) & (upper == upper.max()))


def to_display(values: np.ndarray | float, scale: Scale, what: str) -> np.ndarray:
    """Map physical numbers to the space in which the axis was drawn."""
    values = np.asarray(values, dtype=float)
    if scale == "linear":
        return values
    if (values <= 0).any():
        raise NotComputable(f"{what} has non-positive numbers on a log10 axis")
    return np.log10(values)


def from_display(values: np.ndarray | float, scale: Scale) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values if scale == "linear" else 10.0 ** values


def interpolate(x: np.ndarray, v: np.ndarray, position: float, x_scale: Scale, v_scale: Scale) -> float:
    """Value at ``position`` on the polyline through sorted samples, straight as drawn on the axes."""
    xs, vs = to_display(x, x_scale, "x"), to_display(v, v_scale, "value")
    p = to_display(position, x_scale, "position")
    return float(from_display(np.interp(p, xs, vs), v_scale))


def interpolate_inverse(x: tuple[float, float], v: tuple[float, float], target: float,
                        x_scale: Scale, v_scale: Scale) -> float:
    """Coordinate where the drawn segment from (x0, v0) to (x1, v1) reaches ``target``."""
    x0, x1 = to_display(np.array(x), x_scale, "x")
    v0, v1 = to_display(np.array(v), v_scale, "value")
    t_display = to_display(target, v_scale, "level")
    fraction = (t_display - v0) / (v1 - v0)
    return float(from_display(x0 + fraction * (x1 - x0), x_scale))
