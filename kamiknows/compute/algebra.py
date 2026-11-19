"""Algebra: pointwise arithmetic on series and quantities, and access to point components."""

from __future__ import annotations

import numpy as np

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.evaluate import lookup
from kamiknows.compute.types import Axis, Point, Quantity, Series, expr
from kamiknows.compute.units import DIMENSIONLESS, Unit, conversion_factor

Operand = Series | Quantity | float
OPERATORS = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}
FUNCTIONS = ("neg", "abs", "reciprocal", "power", "log10", "exp10")
COMPONENTS = ("center", "lower", "upper")


def combine(left: Operand, right: Operand, op: str) -> Series | Quantity:
    """Apply ``add``, ``sub``, ``mul`` or ``div`` to two operands.

    Each operand is a series, a quantity or a plain (dimensionless) number.
    Two series are combined on the elements of ``left``: ``right`` is
    evaluated at each of their centers (containing bin, or interpolation
    between samples) and elements where it is undefined are dropped. For add
    and sub, ``right`` is converted to the unit of ``left``. A series result
    drops elements where the operation is undefined (division by zero) and
    says so in its caveats; a scalar result is then not computable.
    """
    if op not in OPERATORS:
        raise NotComputable(f"unknown operator {op!r}; use one of {sorted(OPERATORS)}")
    origin = f"{op}({expr(left)}, {expr(right)})"
    left_q, right_q = _as_operand(left), _as_operand(right)
    base = left_q if isinstance(left_q, Series) else right_q if isinstance(right_q, Series) else None
    if base is None:
        unit, factor = _result_unit(op, left_q.unit, right_q.unit)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = OPERATORS[op](left_q.value, right_q.value * factor)
        if not np.isfinite(value):
            raise NotComputable(f"{origin} is undefined")
        return Quantity(value, unit, origin)
    a, a_unit, a_defined = _values_on(left_q, base)
    b, b_unit, b_defined = _values_on(right_q, base)
    unit, factor = _result_unit(op, a_unit, b_unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = OPERATORS[op](a, b * factor)
    caveats = _merge_caveats(left_q, right_q)
    return _finish(base, result, a_defined & b_defined, unit, origin, caveats)


def transform(operand: Operand, function: str, exponent: float | None = None) -> Series | Quantity:
    """Apply a pointwise function: neg, abs, reciprocal, power (with ``exponent``), log10, exp10.

    log10 and exp10 need a dimensionless operand: divide by a reference
    quantity first (log10 of E / 1 MeV). A series drops the elements where
    the function is undefined; a quantity is then not computable.
    """
    operand = _as_operand(operand)
    if function not in FUNCTIONS:
        raise NotComputable(f"unknown function {function!r}; use one of {FUNCTIONS}")
    if (function == "power") != (exponent is not None):
        raise NotComputable("an exponent is given with power, and only with power")
    unit = operand.unit if isinstance(operand, Quantity) else operand.value_axis.unit
    unit = _function_unit(function, unit, exponent)
    origin = f"{function}({expr(operand)}{'' if exponent is None else f', {exponent:g}'})"
    numbers = operand.value if isinstance(operand, Quantity) else operand.values
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = {
            "neg": np.negative, "abs": np.abs, "reciprocal": np.reciprocal,
            "power": lambda x: np.power(x, exponent),
            "log10": lambda x: np.log10(np.where(np.asarray(x) > 0, x, np.nan)),
            "exp10": lambda x: np.power(10.0, x),
        }[function](np.asarray(numbers, dtype=float))
    if isinstance(operand, Quantity):
        if not np.isfinite(result):
            raise NotComputable(f"{origin} is undefined")
        return Quantity(float(result), unit, origin)
    return _finish(operand, result, np.ones(operand.size, dtype=bool), unit, origin, operand.caveats)


def component(point: Point, axis: str, which: str = "center") -> Quantity:
    """One number of a point: ``value``, or a coordinate (center, lower or upper bound) on ``axis``."""
    if axis == "value":
        return point.value
    if axis not in point.coords:
        raise NotComputable(f"{point.origin} has no coordinate {axis!r}: its coordinates are "
                            f"{sorted(point.coords)}, and its value is axis 'value'")
    if which not in COMPONENTS:
        raise NotComputable(f"unknown component {which!r}; use one of {COMPONENTS}")
    center = point.coords[axis]
    if which == "center":
        return center
    if point.bounds is None:
        raise NotComputable(f"{point.origin} is a point sample: it has no bounds")
    bound = point.bounds[axis][0 if which == "lower" else 1]
    return Quantity(bound, center.unit, f"{point.origin}.{axis}.{which}")


def _as_operand(operand: Operand) -> Series | Quantity:
    if isinstance(operand, (Series, Quantity)):
        return operand
    if isinstance(operand, (int, float)) and not isinstance(operand, bool):
        return Quantity(operand, DIMENSIONLESS, f"{operand:g}")
    raise NotComputable(f"cannot do arithmetic on {type(operand).__name__}")


def _values_on(operand: Series | Quantity, base: Series) -> tuple[np.ndarray, Unit, np.ndarray]:
    """The operand's numbers on the elements of ``base``, its unit, and where they are defined."""
    if isinstance(operand, Quantity):
        return np.full(base.size, operand.value), operand.unit, np.ones(base.size, dtype=bool)
    if operand is base:
        return operand.values, operand.value_axis.unit, np.ones(base.size, dtype=bool)
    if [a.name for a in operand.domain] != [a.name for a in base.domain]:
        raise NotComputable(f"cannot combine {base.origin} and {operand.origin}: different domains")
    factors = np.array([1.0 if b.unit == o.unit else conversion_factor(b.unit, o.unit)
                        for b, o in zip(base.domain, operand.domain)])
    values = np.zeros(base.size)
    defined = np.zeros(base.size, dtype=bool)
    for i, center in enumerate(base.centers):
        value = lookup(operand, center * factors)
        if value is not None:
            values[i], defined[i] = value, True
    return values, operand.value_axis.unit, defined


def _result_unit(op: str, left: Unit, right: Unit) -> tuple[Unit, float]:
    """Unit of the result, and the factor that converts the right operand for add and sub."""
    if op in ("add", "sub"):
        return left, 1.0 if left == right else conversion_factor(right, left)
    return (left * right if op == "mul" else left / right), 1.0


def _function_unit(function: str, unit: Unit, exponent: float | None) -> Unit:
    if function in ("neg", "abs"):
        return unit
    if function == "reciprocal":
        return unit ** -1
    if function == "power":
        return unit ** exponent
    if not unit.dimensionless:
        raise NotComputable(f"{function} needs a dimensionless operand, got {unit}; divide by a reference first")
    return unit


def _finish(base: Series, result: np.ndarray, defined: np.ndarray, unit: Unit,
            origin: str, caveats: tuple[str, ...]) -> Series:
    keep = defined & np.isfinite(result)
    dropped = int((~keep).sum())
    if dropped:
        caveats = (*caveats, f"{dropped} of {base.size} elements dropped where the operation is undefined")
    return base.take(keep, origin).derive(
        origin, values=result[keep], value_axis=Axis(base.value_axis.name, unit), caveats=caveats,
    )


def _merge_caveats(*operands: Series | Quantity) -> tuple[str, ...]:
    caveats = [c for operand in operands if isinstance(operand, Series) for c in operand.caveats]
    return tuple(dict.fromkeys(caveats))
