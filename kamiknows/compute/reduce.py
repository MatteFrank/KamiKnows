"""Reductions: a series to one number (reduce), to one axis (project), or to its extremes."""

from __future__ import annotations

import numpy as np

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.geometry import order_1d
from kamiknows.compute.types import Axis, Point, Quantity, Series
from kamiknows.compute.units import DIMENSIONLESS

HOWS = ("sum", "integral", "mean", "min", "max", "count")
KINDS = ("max", "min")
SCOPES = ("global", "local")


def reduce(series: Series, how: str) -> Quantity:
    """Reduce a series to one number.

    ``sum``, ``mean``, ``min``, ``max`` and ``count`` act on the values.
    ``integral`` is the area (volume in 2D) under the series: sum of value
    times support size for bins and cells, trapezoids for 1D point samples;
    its unit is the value unit times the domain units.
    """
    if how not in HOWS:
        raise NotComputable(f"unknown reduction {how!r}; use one of {HOWS}")
    if series.size == 0:
        raise NotComputable(f"{how} of an empty series ({series.origin})")
    origin = f"{how}({series.origin})"
    values, unit = series.values, series.value_axis.unit
    if how == "count":
        return Quantity(series.size, DIMENSIONLESS, origin)
    if how == "integral":
        for axis in series.domain:
            unit = unit * axis.unit
        return Quantity(_integral(series), unit, origin)
    return Quantity({"sum": np.sum, "mean": np.mean, "min": np.min, "max": np.max}[how](values), unit, origin)


def project(series: Series, onto: str, how: str = "sum") -> Series:
    """Reduce a 2D series along its other axis, giving a 1D series over ``onto``.

    Elements are grouped by position on ``onto``: a cell joins the current
    group when its center lies within the support of the group's first cell
    (samples join when they share the coordinate). Each group is reduced
    exactly as ``reduce`` would reduce it; ``integral`` integrates over the
    removed axis.
    """
    if series.dim != 2:
        raise NotComputable(f"project needs a 2D series; {series.origin} is 1D")
    if series.size == 0:
        raise NotComputable(f"project of an empty series ({series.origin})")
    k = series.domain_index(onto)
    removed = [1 - k]
    centers, lower, upper, values, unit = [], [], [], [], None
    for group in _groups(series, k):
        part = Series(
            series.series_id, (series.domain[1 - k],), series.value_axis,
            centers=series.centers[group][:, removed], values=series.values[group],
            lower=None if series.lower is None else series.lower[group][:, removed],
            upper=None if series.upper is None else series.upper[group][:, removed],
            origin=series.origin,
        )
        quantity = reduce(part, how)
        unit = quantity.unit
        values.append(quantity.value)
        centers.append(series.centers[group, k].mean())
        if series.has_support:
            lower.append(series.lower[group, k].min())
            upper.append(series.upper[group, k].max())
    return Series(
        series.series_id, (series.domain[k],), Axis(series.value_axis.name, unit),
        centers=np.array(centers)[:, None], values=np.array(values),
        lower=np.array(lower)[:, None] if series.has_support else None,
        upper=np.array(upper)[:, None] if series.has_support else None,
        origin=f"project({series.origin}, {onto}, {how})", caveats=series.caveats,
    )


def argextremum(series: Series, kind: str = "max", scope: str = "global") -> tuple[Point, ...]:
    """Where a series reaches its maximum or minimum: the global one, or with scope local every local one.

    ``global`` returns every element attaining the extreme value (ties
    included). ``local`` (1D) returns the interior local extremes in order of
    coordinate: a run of equal values counts once, at its first element, and
    the two ends are not local extremes (``global`` covers them).
    """
    if kind not in KINDS or scope not in SCOPES:
        raise NotComputable(f"argextremum takes kind in {KINDS} and scope in {SCOPES}")
    if series.size == 0:
        raise NotComputable(f"arg{kind} of an empty series ({series.origin})")
    origin = f"arg{kind}({series.origin}, {scope})"
    sign = 1.0 if kind == "max" else -1.0
    if scope == "global":
        signed = sign * series.values
        indices = np.flatnonzero(signed == signed.max())
    else:
        order = order_1d(series, "local extrema")
        signed = sign * series.values[order]
        starts = np.flatnonzero(np.r_[True, signed[1:] != signed[:-1]])
        runs = signed[starts]
        indices = [order[starts[r]] for r in range(1, len(runs) - 1) if runs[r] > runs[r - 1] and runs[r] > runs[r + 1]]
    return tuple(series.point(int(i), f"{origin}[{k}]") for k, i in enumerate(indices))


def _integral(series: Series) -> float:
    if series.has_support:
        sizes = np.prod(series.upper - series.lower, axis=1)
        return float(np.sum(series.values * sizes))
    order = order_1d(series, "integral of point samples")
    return float(np.trapezoid(series.values[order], series.centers[order, 0]))


def _groups(series: Series, k: int) -> list[np.ndarray]:
    groups: list[list[int]] = []
    for i in np.argsort(series.centers[:, k], kind="stable"):
        center = series.centers[i, k]
        if groups:
            first = groups[-1][0]
            if series.has_support:
                same = series.lower[first, k] <= center <= series.upper[first, k]
            else:
                same = bool(np.isclose(center, series.centers[first, k]))
            if same:
                groups[-1].append(int(i))
                continue
        groups.append([int(i)])
    return [np.array(group) for group in groups]
