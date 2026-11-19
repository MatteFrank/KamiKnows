"""Common quantities written only as compositions of the primitives.

Nothing here touches arrays: every function chains primitives. This shows
that the primitives form a basis, and gives the planner short plans for
frequent questions.
"""

from __future__ import annotations

from kamiknows.compute.algebra import combine, component, transform
from kamiknows.compute.calculus import cumulate
from kamiknows.compute.errors import NotComputable
from kamiknows.compute.evaluate import evaluate, solve
from kamiknows.compute.reduce import argextremum, reduce
from kamiknows.compute.select import coordinate
from kamiknows.compute.types import Point, Quantity, Series


def crossing(first: Series, second: Series, direction: str = "any") -> tuple[Point, ...]:
    """Points of ``first`` where it crosses ``second`` (``up``: rises to or above it, ``down``: drops below it).

    Each point is ``first`` itself at the crossing, at the resolution of the
    data: for histograms the bin of ``first`` where the new order starts,
    with that bin's value; between point samples the interpolated position.
    """
    origin = f"crossing({first.origin}, {second.origin}, {direction})"
    points = []
    for k, point in enumerate(solve(combine(first, second, "sub"), 0.0, direction)):
        at = {name: coordinate for name, coordinate in point.coords.items()}
        located = evaluate(first, at)
        points.append(Point(
            {name: Quantity(q.value, q.unit, f"{origin}[{k}].{name}") for name, q in located.coords.items()},
            Quantity(located.value.value, located.value.unit, f"{origin}[{k}].value"),
            located.bounds, f"{origin}[{k}]"))
    return tuple(points)


def normalize(series: Series) -> Series:
    """The series divided by its integral, so that it integrates to one."""
    return combine(series, reduce(series, "integral"), "div")


def mean(series: Series, axis: str) -> Quantity:
    """Mean coordinate on ``axis``, weighting each element by its value (the series read as a density)."""
    weighted = combine(coordinate(series, axis), series, "mul")
    return combine(reduce(weighted, "integral"), reduce(series, "integral"), "div")


def std(series: Series, axis: str) -> Quantity:
    """Standard deviation of the coordinate on ``axis``, weighting each element by its value."""
    deviation = combine(coordinate(series, axis), mean(series, axis), "sub")
    weighted = combine(transform(deviation, "power", 2), series, "mul")
    variance = combine(reduce(weighted, "integral"), reduce(series, "integral"), "div")
    return transform(variance, "power", 0.5)


def quantile(series: Series, q: float) -> Quantity:
    """Coordinate below which a fraction ``q`` (0 < q < 1) of the integral lies; q = 0.5 is the median.

    With negative values the running integral is not monotonic and the first
    coordinate where it reaches the fraction is returned.
    """
    if not 0 < q < 1:
        raise NotComputable(f"quantile needs 0 < q < 1, got {q}")
    target = combine(reduce(series, "integral"), q, "mul")
    crossings = solve(cumulate(series), target, "up")
    if not crossings:
        raise NotComputable(f"the running integral of {series.origin} never reaches {q:g} of the total")
    return component(crossings[0], series.domain[0].name)


def fwhm(series: Series) -> Quantity:
    """Full width at half maximum of the global peak of a 1D series.

    The width runs from where the series rises to half the peak value before
    the peak to where it drops below it after: bin edges for histograms (the
    lower edge of each bin entered), interpolated positions for point
    samples. With tied peaks the first one is used.
    """
    peak = argextremum(series, "max")[0]
    axis = series.domain[0].name
    half = combine(component(peak, "value"), 2, "div")
    peak_x = component(peak, axis).value
    rises = [p for p in solve(series, half, "up") if component(p, axis).value <= peak_x]
    drops = [p for p in solve(series, half, "down") if component(p, axis).value > peak_x]
    if not rises or not drops:
        raise NotComputable(f"{series.origin} does not fall to half its maximum on both sides of the peak")
    edge = "lower" if series.has_support else "center"
    return combine(component(drops[0], axis, edge), component(rises[-1], axis, edge), "sub")


def linear_fit(series: Series) -> dict[str, Quantity]:
    """Least-squares line value = intercept + slope * x through the elements of a 1D series (unweighted)."""
    if series.dim != 1:
        raise NotComputable(f"linear_fit needs a 1D series; take a section of {series.origin} first")
    x = coordinate(series, series.domain[0].name)
    n = reduce(series, "count")
    sum_x, sum_y = reduce(x, "sum"), reduce(series, "sum")
    sum_xx = reduce(transform(x, "power", 2), "sum")
    sum_xy = reduce(combine(x, series, "mul"), "sum")
    numerator = combine(combine(n, sum_xy, "mul"), combine(sum_x, sum_y, "mul"), "sub")
    denominator = combine(combine(n, sum_xx, "mul"), transform(sum_x, "power", 2), "sub")
    slope = combine(numerator, denominator, "div")
    intercept = combine(combine(sum_y, combine(slope, sum_x, "mul"), "sub"), n, "div")
    return {"slope": slope, "intercept": intercept}


def distance(first: Point, second: Point, axis: str | None = None) -> Quantity:
    """Distance between two points along ``axis``, or Euclidean over all their coordinates."""
    if axis is not None:
        return transform(combine(component(first, axis), component(second, axis), "sub"), "abs")
    if sorted(first.coords) != sorted(second.coords):
        raise NotComputable(f"{first.origin} and {second.origin} have different coordinates")
    total = None
    for name in first.coords:
        square = transform(combine(component(first, name), component(second, name), "sub"), "power", 2)
        total = square if total is None else combine(total, square, "add")
    return transform(total, "power", 0.5)
