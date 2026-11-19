"""Value types shared by every compute operation.

Every numeric source (digitized plots today; tables and equations later)
becomes a Series: a discrete function from a 1D or 2D domain to one value
axis. Operations take and return only Series, Point and Quantity. Each object
carries an ``origin``: the expression that produced it, e.g.
``div(max(A), max(B))``, so every number can be traced to its data.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.units import Unit, conversion_factor

Scale = Literal["linear", "log10"]


@dataclass(frozen=True)
class Axis:
    """A coordinate or value axis.

    ``scale`` records how the axis was drawn; values are always stored as
    physical numbers. It matters only where the drawing itself is
    reconstructed, e.g. interpolating between samples of a log-scale curve.
    """

    name: str
    unit: Unit
    scale: Scale = "linear"

    def __post_init__(self) -> None:
        if self.scale not in ("linear", "log10"):
            raise ValueError(f"axis {self.name!r}: unknown scale {self.scale!r}")


@dataclass(frozen=True)
class Quantity:
    """A finite scalar with its unit and the expression that produced it."""

    value: float
    unit: Unit
    origin: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))
        if not math.isfinite(self.value):
            raise ValueError(f"non-finite quantity {self.value!r}")


@dataclass(frozen=True)
class Point:
    """One location of a series: coordinates, support (if any) and value."""

    coords: Mapping[str, Quantity]
    value: Quantity
    bounds: Mapping[str, tuple[float, float]] | None = None
    origin: str = ""


@dataclass(frozen=True, eq=False)
class Series:
    """A discrete function from a 1D or 2D domain to one value axis.

    Element i sits at ``centers[i]`` (shape n x d) with value ``values[i]``.
    Bins and cells also carry their support ``[lower[i], upper[i]]`` on every
    domain axis; point samples (curve points, table rows) have none. Arrays
    are copied, checked finite and made read-only.
    """

    series_id: str
    domain: tuple[Axis, ...]
    value_axis: Axis
    centers: np.ndarray
    values: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    origin: str = ""
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= len(self.domain) <= 2:
            raise ValueError(f"{self.series_id}: domain must have 1 or 2 axes, got {len(self.domain)}")
        names = [axis.name for axis in (*self.domain, self.value_axis)]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.series_id}: repeated axis names {names}")
        values = self._freeze("values", None)
        shape = (len(values), len(self.domain))
        centers = self._freeze("centers", shape)
        if (self.lower is None) != (self.upper is None):
            raise ValueError(f"{self.series_id}: give both lower and upper bounds, or neither")
        if self.lower is not None:
            lower, upper = self._freeze("lower", shape), self._freeze("upper", shape)
            if not ((lower <= centers) & (centers <= upper)).all():
                raise ValueError(f"{self.series_id}: every center must lie within its bounds")
        if not self.origin:
            object.__setattr__(self, "origin", self.series_id)

    def _freeze(self, name: str, shape: tuple[int, int] | None) -> np.ndarray:
        array = np.array(getattr(self, name), dtype=float)
        if (array.ndim != 1) if shape is None else (array.shape != shape):
            raise ValueError(f"{self.series_id}: {name} has shape {array.shape}, expected {shape or '(n,)'}")
        if not np.isfinite(array).all():
            raise ValueError(f"{self.series_id}: {name} contains non-finite numbers")
        array.setflags(write=False)
        object.__setattr__(self, name, array)
        return array

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def dim(self) -> int:
        return len(self.domain)

    @property
    def has_support(self) -> bool:
        return self.lower is not None

    def axis(self, name: str) -> Axis:
        for axis in (*self.domain, self.value_axis):
            if axis.name == name:
                return axis
        raise NotComputable(f"{self.origin} has no axis {name!r}")

    def domain_index(self, name: str) -> int:
        for index, axis in enumerate(self.domain):
            if axis.name == name:
                return index
        raise NotComputable(f"{name!r} is not a domain axis of {self.origin}")

    def derive(self, origin: str, **changes: Any) -> Series:
        """A new series with some fields replaced and the expression that produced it."""
        return dataclasses.replace(self, origin=origin, **changes)

    def take(self, selection: np.ndarray, origin: str) -> Series:
        """The elements picked by a boolean mask or an index array.

        No operation can use an empty series, so a selection that keeps nothing is refused here,
        with the expression that emptied it, instead of failing later inside another operation.
        """
        if self.values[selection].size == 0:
            raise NotComputable(f"{origin} keeps no element of {self.origin}")
        return self.derive(
            origin,
            centers=self.centers[selection],
            values=self.values[selection],
            lower=None if self.lower is None else self.lower[selection],
            upper=None if self.upper is None else self.upper[selection],
        )

    def point(self, index: int, origin: str | None = None) -> Point:
        origin = origin or self.origin
        coords = {axis.name: Quantity(self.centers[index, k], axis.unit, f"{origin}.{axis.name}")
                  for k, axis in enumerate(self.domain)}
        bounds = None
        if self.has_support:
            bounds = {axis.name: (float(self.lower[index, k]), float(self.upper[index, k]))
                      for k, axis in enumerate(self.domain)}
        value = Quantity(self.values[index], self.value_axis.unit, f"{origin}.value")
        return Point(coords, value, bounds, origin)


def to_unit(value: float | Quantity, unit: Unit) -> float:
    """A plain number is taken to be in ``unit``; a Quantity is converted to it.

    Equal units pass unchanged, including two unknown ones: a coordinate read
    from a series can always be used again on that same series.
    """
    if not isinstance(value, Quantity):
        return float(value)
    if value.unit == unit:
        return value.value
    return value.value * conversion_factor(value.unit, unit)


def expr(value: Any) -> str:
    """How an operand is written inside an origin expression."""
    if isinstance(value, (Series, Point)):
        return value.origin
    if isinstance(value, Quantity):
        return value.origin or f"{value.value:.6g} {value.unit}"
    if value is None:
        return "none"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={expr(item)}" for key, item in value.items())
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)
