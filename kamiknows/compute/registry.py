"""The operations a plan may call: name, parameters, result and one-line description.

This is the only list a planner sees. Each description is the first line of
the function's docstring, so documentation and catalogue cannot drift apart;
choices come from the constants the functions themselves check.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from kamiknows.compute.algebra import COMPONENTS, FUNCTIONS, OPERATORS, combine, component, transform
from kamiknows.compute.calculus import cumulate, differentiate
from kamiknows.compute.derived import crossing, distance, fwhm, linear_fit, mean, normalize, quantile, std
from kamiknows.compute.evaluate import DIRECTIONS, evaluate, solve
from kamiknows.compute.metadata import describe
from kamiknows.compute.reduce import HOWS, KINDS as EXTREMA, SCOPES, argextremum, project, reduce
from kamiknows.compute.select import WHICH, coordinate, restrict, section

KINDS = {
    "series": "a series handle, or a reference to a series",
    "point": "a reference to a point",
    "operand": "a series, a quantity or a number",
    "quantity": 'a number in the axis unit, {"value": v, "unit": "u"}, or a reference',
    "coords": "an object giving a quantity for every domain axis",
    "axis": "an axis name",
    "number": "a plain number",
    "choice": "one of the listed words",
}
RESULTS = ("series", "point", "points", "quantity", "series|quantity", "record")


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str
    required: bool = True
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"parameter {self.name!r}: unknown kind {self.kind!r}")
        if (self.kind == "choice") != bool(self.choices):
            raise ValueError(f"parameter {self.name!r}: choices go with kind 'choice' only")


@dataclass(frozen=True)
class Operation:
    name: str
    function: Callable[..., Any]
    parameters: tuple[Parameter, ...]
    returns: str

    @property
    def summary(self) -> str:
        return inspect.getdoc(self.function).splitlines()[0]

    def signature(self) -> str:
        parts = []
        for parameter in self.parameters:
            kind = "|".join(parameter.choices) if parameter.choices else parameter.kind
            parts.append(f"{parameter.name}{'' if parameter.required else '?'}: {kind}")
        return f"{self.name}({', '.join(parts)}) -> {self.returns}"


def _operation(function: Callable[..., Any], returns: str, *parameters: Parameter) -> Operation:
    if returns not in RESULTS:
        raise ValueError(f"{function.__name__}: unknown result {returns!r}")
    return Operation(function.__name__, function, parameters, returns)


P = Parameter
OPERATIONS: dict[str, Operation] = {operation.name: operation for operation in (
    # selection
    _operation(restrict, "series", P("series", "series"), P("axis", "axis"),
               P("lower", "quantity", False), P("upper", "quantity", False)),
    _operation(section, "series", P("series", "series"), P("axis", "axis"), P("at", "quantity")),
    _operation(coordinate, "series", P("series", "series"), P("axis", "axis"), P("which", "choice", False, WHICH)),
    # evaluation
    _operation(evaluate, "point", P("series", "series"), P("at", "coords")),
    _operation(solve, "points", P("series", "series"), P("level", "quantity"),
               P("direction", "choice", False, DIRECTIONS)),
    # algebra
    _operation(combine, "series|quantity", P("left", "operand"), P("right", "operand"),
               P("op", "choice", True, tuple(OPERATORS))),
    _operation(transform, "series|quantity", P("operand", "operand"), P("function", "choice", True, FUNCTIONS),
               P("exponent", "number", False)),
    _operation(component, "quantity", P("point", "point"), P("axis", "axis"),
               P("which", "choice", False, COMPONENTS)),
    # reduction
    _operation(reduce, "quantity", P("series", "series"), P("how", "choice", True, HOWS)),
    _operation(project, "series", P("series", "series"), P("onto", "axis"), P("how", "choice", False, HOWS)),
    _operation(argextremum, "points", P("series", "series"), P("kind", "choice", False, EXTREMA),
               P("scope", "choice", False, SCOPES)),
    # calculus
    _operation(differentiate, "series", P("series", "series")),
    _operation(cumulate, "series", P("series", "series")),
    # metadata
    _operation(describe, "record", P("series", "series")),
    # compositions
    _operation(crossing, "points", P("first", "series"), P("second", "series"),
               P("direction", "choice", False, DIRECTIONS)),
    _operation(normalize, "series", P("series", "series")),
    _operation(mean, "quantity", P("series", "series"), P("axis", "axis")),
    _operation(std, "quantity", P("series", "series"), P("axis", "axis")),
    _operation(quantile, "quantity", P("series", "series"), P("q", "number")),
    _operation(fwhm, "quantity", P("series", "series")),
    _operation(linear_fit, "record", P("series", "series")),
    _operation(distance, "quantity", P("first", "point"), P("second", "point"), P("axis", "axis", False)),
)}


def catalogue() -> str:
    """The operations as text for a planner prompt: one signature and description per line."""
    return "\n".join(f"{op.signature()}: {op.summary}" for op in OPERATIONS.values())
