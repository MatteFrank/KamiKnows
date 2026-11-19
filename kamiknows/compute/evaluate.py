"""Evaluation and inversion: the value at a point, and where a value is reached."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.geometry import containing, interpolate, interpolate_inverse, order_1d
from kamiknows.compute.types import Point, Quantity, Series, expr, to_unit

DIRECTIONS = ("up", "down", "any")


def evaluate(series: Series, at: Mapping[str, float | Quantity]) -> Point:
    """The series at a position given for every domain axis.

    Bins and cells return the element containing the position, with its
    bounds: a histogram is constant within a bin, so nothing is interpolated.
    1D point samples are interpolated along the polyline as drawn (straight on
    log axes too). A position outside the data, or in a gap between bins, is
    not computable.
    """
    names = [axis.name for axis in series.domain]
    if sorted(at) != sorted(names):
        raise NotComputable(f"{series.origin} needs coordinates for {names}, got {sorted(at)}")
    position = np.array([to_unit(at[axis.name], axis.unit) for axis in series.domain])
    origin = f"evaluate({series.origin}, {expr(at)})"
    outside = NotComputable(f"{expr(at)} is outside the data of {series.origin} (beyond the plotted range or in a gap)")
    if series.has_support:
        index = element_at(series, position)
        if index is None:
            raise outside
        return series.point(index, origin)
    value = sample_at(series, float(position[0]))
    if value is None:
        raise outside
    axis = series.domain[0]
    return Point({axis.name: Quantity(position[0], axis.unit, f"{origin}.{axis.name}")},
                 Quantity(value, series.value_axis.unit, f"{origin}.value"), None, origin)


def lookup(series: Series, position: np.ndarray) -> float | None:
    """The value at a position in the series' own units, or None outside its data."""
    if series.has_support:
        index = element_at(series, position)
        return None if index is None else float(series.values[index])
    return sample_at(series, float(position[0]))


def element_at(series: Series, position: np.ndarray) -> int | None:
    """Index of the bin or cell containing ``position`` (the nearest center if supports overlap)."""
    mask = np.ones(series.size, dtype=bool)
    for k in range(series.dim):
        mask &= containing(series.lower[:, k], series.upper[:, k], position[k])
    hits = np.flatnonzero(mask)
    if hits.size == 0:
        return None
    return int(hits[np.argmin(np.linalg.norm(series.centers[hits] - position, axis=1))])


def sample_at(series: Series, position: float) -> float | None:
    """Interpolated value of 1D point samples at ``position``, or None outside the sampled range."""
    if series.dim != 1:
        raise NotComputable(f"interpolation between 2D point samples is not supported ({series.origin})")
    order = order_1d(series, "evaluate")
    x, v = series.centers[order, 0], series.values[order]
    if not x[0] <= position <= x[-1]:
        return None
    return interpolate(x, v, position, series.domain[0].scale, series.value_axis.scale)


def solve(series: Series, level: float | Quantity, direction: str = "any") -> tuple[Point, ...]:
    """Where a 1D series crosses ``level``, in order of coordinate.

    ``up`` finds where it reaches the level from below, ``down`` where it
    drops below it, ``any`` both. Between point samples the crossing is
    interpolated as drawn. A histogram is a step function and crosses between
    two bins: the bin it enters is returned (center, bounds and value),
    because there is no finer position. No crossing gives an empty tuple.
    """
    if direction not in DIRECTIONS:
        raise NotComputable(f"unknown direction {direction!r}; use one of {DIRECTIONS}")
    order = order_1d(series, "solve")
    target = to_unit(level, series.value_axis.unit)
    origin = f"solve({series.origin}, {expr(level)}, {direction})"
    above = series.values[order] >= target
    up, down = ~above[:-1] & above[1:], above[:-1] & ~above[1:]
    crossings = np.flatnonzero({"up": up, "down": down, "any": up | down}[direction])
    if series.has_support:
        return tuple(series.point(int(order[i + 1]), f"{origin}[{k}]") for k, i in enumerate(crossings))
    axis, x, v = series.domain[0], series.centers[order, 0], series.values[order]
    points = []
    for k, i in enumerate(crossings):
        position = interpolate_inverse((x[i], x[i + 1]), (v[i], v[i + 1]), target,
                                       axis.scale, series.value_axis.scale)
        name = f"{origin}[{k}]"
        points.append(Point({axis.name: Quantity(position, axis.unit, f"{name}.{axis.name}")},
                            Quantity(target, series.value_axis.unit, f"{name}.value"), None, name))
    return tuple(points)
