"""Execute a plan: a JSON list of operation steps over a catalog of series.

A plan is ``{"steps": [{"id": "a", "op": "<operation>", "args": {...}}, ...],
"result": ["$a", ...]}``, or ``{"not_computable": "<reason>"}`` when the
planner decides that the data cannot answer the question. An argument is a
literal, a series handle, or a reference to an earlier step: ``$a``,
``$a[0]`` (one of several points), ``$a.value`` or ``$a.x`` (the value or a
coordinate of a point), ``$a.slope`` (a field of a record).

The outcome keeps apart a malformed plan (``invalid_plan``: the planner's
fault), data that cannot give the answer (``not_computable``), a plan that
declines (``declined``) and a result (``computed``).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from kamiknows.compute.algebra import component
from kamiknows.compute.catalog import Catalog
from kamiknows.compute.errors import NotComputable
from kamiknows.compute.registry import KINDS, OPERATIONS, Parameter
from kamiknows.compute.types import Point, Quantity, Series
from kamiknows.compute.units import DIMENSIONLESS, Unit

STATUSES = ("computed", "declined", "not_computable", "invalid_plan")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_REFERENCE = re.compile(r"\$(?P<step>[A-Za-z_]\w*)(?P<path>(?:\[\d+\]|\.[A-Za-z_]\w*)*)")
_PATH_PART = re.compile(r"\[(\d+)\]|\.([A-Za-z_]\w*)")


class PlanError(Exception):
    """The plan itself is malformed: unknown operation, bad argument or bad reference."""


def execute(plan: Any, catalog: Catalog) -> dict[str, Any]:
    """Run a plan and return its outcome: status, rendered results or reason, and the steps run."""
    log: list[dict[str, str]] = []
    try:
        if isinstance(plan, Mapping) and "not_computable" in plan:
            if set(plan) != {"not_computable"} or not str(plan["not_computable"]).strip():
                raise PlanError("a declining plan has only a non-empty 'not_computable' reason")
            return {"status": "declined", "reason": str(plan["not_computable"]), "steps": log}
        steps, result = _validate(plan)
        results: dict[str, Any] = {}
        caveats: dict[str, None] = {}
        for step in steps:
            try:
                arguments = _arguments(OPERATIONS[step["op"]], step["args"], catalog, results)
                value = OPERATIONS[step["op"]].function(**arguments)
            except NotComputable as error:
                return {"status": "not_computable", "reason": str(error), "failed_step": step["id"], "steps": log}
            for item in (*arguments.values(), value):
                if isinstance(item, Series):
                    caveats.update(dict.fromkeys(item.caveats))
            results[step["id"]] = value
            log.append({"id": step["id"], "op": step["op"], "origin": _origin(value),
                        **({"points": len(value)} if isinstance(value, tuple) else {})})
        try:
            outputs = [_resolve(reference, results) for reference in result]
        except NotComputable as error:
            return {"status": "not_computable", "reason": str(error), "failed_step": "result", "steps": log}
        return {"status": "computed", "results": [render(value) for value in outputs],
                "caveats": list(caveats), "steps": log}
    except PlanError as error:
        return {"status": "invalid_plan", "reason": str(error), "steps": log}


def render(value: Any) -> Any:
    """A JSON-ready description of a result, with a short text form for prompts and reviews."""
    if isinstance(value, Quantity):
        return {"type": "quantity", "value": value.value, "unit": str(value.unit),
                "origin": value.origin, "text": _quantity_text(value)}
    if isinstance(value, Point):
        coords = {}
        for name, coordinate in value.coords.items():
            coords[name] = {"value": coordinate.value, "unit": str(coordinate.unit)}
            if value.bounds is not None:
                coords[name]["lower"], coords[name]["upper"] = value.bounds[name]
        position = ", ".join(
            f"{name}={_quantity_text(q)}" + ("" if value.bounds is None else
                                            " [{:.6g}, {:.6g}]".format(*value.bounds[name]))
            for name, q in value.coords.items())
        return {"type": "point", "coords": coords, "value": {"value": value.value.value, "unit": str(value.value.unit)},
                "origin": value.origin, "text": f"{_quantity_text(value.value)} at {position}"}
    if isinstance(value, tuple):
        return {"type": "points", "items": [render(item) for item in value],
                "text": f"{len(value)} point(s)"}
    if isinstance(value, Series):
        return {"type": "series", "origin": value.origin, "elements": value.size,
                "text": f"series of {value.size} elements"}
    if isinstance(value, Mapping):
        return {"type": "record", "fields": _plain(value)}
    return value


def _validate(plan: Any) -> tuple[list[Mapping[str, Any]], list[str]]:
    if not isinstance(plan, Mapping) or set(plan) != {"steps", "result"}:
        raise PlanError("a plan is an object with exactly 'steps' and 'result' (or only 'not_computable')")
    steps, result = plan["steps"], plan["result"]
    if not isinstance(steps, list) or not steps:
        raise PlanError("'steps' must be a non-empty list")
    seen: set[str] = set()
    for number, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or set(step) != {"id", "op", "args"}:
            raise PlanError(f"step {number} must have exactly 'id', 'op' and 'args'")
        if not isinstance(step["id"], str) or not _IDENTIFIER.fullmatch(step["id"]) or step["id"] in seen:
            raise PlanError(f"step {number}: id {step['id']!r} must be a new identifier")
        if step["op"] not in OPERATIONS:
            raise PlanError(f"step {step['id']}: unknown operation {step['op']!r}")
        if not isinstance(step["args"], Mapping):
            raise PlanError(f"step {step['id']}: 'args' must be an object")
        seen.add(step["id"])
    if not isinstance(result, list) or not result or not all(isinstance(r, str) and r.startswith("$") for r in result):
        raise PlanError("'result' must be a non-empty list of references such as \"$a\"")
    return steps, result


def _arguments(operation: Any, raw: Mapping[str, Any], catalog: Catalog, results: dict[str, Any]) -> dict[str, Any]:
    names = {parameter.name for parameter in operation.parameters}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise PlanError(f"{operation.name} has no parameter {unknown}; signature: {operation.signature()}")
    arguments = {}
    for parameter in operation.parameters:
        if parameter.name not in raw:
            if parameter.required:
                raise PlanError(f"{operation.name} needs {parameter.name!r}; signature: {operation.signature()}")
            continue
        try:
            arguments[parameter.name] = _coerce(parameter, raw[parameter.name], catalog, results)
        except PlanError as error:
            raise PlanError(f"{operation.name}: {error}; signature: {operation.signature()}") from error
    return arguments


def _coerce(parameter: Parameter, raw: Any, catalog: Catalog, results: dict[str, Any]) -> Any:
    kind = parameter.kind
    if isinstance(raw, str) and raw.startswith("$"):
        value = _resolve(raw, results)
    elif kind in ("series", "operand") and isinstance(raw, str):
        value = catalog.get(raw)
    elif kind in ("quantity", "operand") and isinstance(raw, Mapping):
        value = _literal_quantity(raw, parameter)
    elif kind == "coords":
        if not isinstance(raw, Mapping):
            raise PlanError(f"{parameter.name} must be {KINDS[kind]}")
        return {axis: _coerce(Parameter(axis, "quantity"), item, catalog, results) for axis, item in raw.items()}
    else:
        value = raw
    return _checked(parameter, value)


def _checked(parameter: Parameter, value: Any) -> Any:
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if parameter.kind == "number" and isinstance(value, Quantity) and value.unit == DIMENSIONLESS:
        return value.value
    valid = {
        "series": isinstance(value, Series),
        "point": isinstance(value, Point),
        "operand": isinstance(value, (Series, Quantity)) or number,
        "quantity": isinstance(value, Quantity) or number,
        "axis": isinstance(value, str),
        "number": number,
        "choice": value in parameter.choices,
    }[parameter.kind]
    if not valid:
        expected = KINDS[parameter.kind] + (f" ({', '.join(parameter.choices)})" if parameter.choices else "")
        raise PlanError(f"{parameter.name} must be {expected}, got {value!r}")
    return value


def _literal_quantity(raw: Mapping[str, Any], parameter: Parameter) -> Quantity:
    if set(raw) != {"value", "unit"}:
        raise PlanError(f"{parameter.name}: a literal quantity is {{\"value\": v, \"unit\": \"u\"}}")
    try:
        return Quantity(raw["value"], Unit.parse(str(raw["unit"])), f"{raw['value']} {raw['unit']}")
    except (TypeError, ValueError) as error:
        raise PlanError(f"{parameter.name}: bad quantity {dict(raw)!r} ({error})") from error


def _resolve(reference: str, results: dict[str, Any]) -> Any:
    match = _REFERENCE.fullmatch(reference)
    if match is None:
        raise PlanError(f"malformed reference {reference!r}")
    if match["step"] not in results:
        raise PlanError(f"reference {reference!r} names a step that has not run")
    value = results[match["step"]]
    for index, name in _PATH_PART.findall(match["path"]):
        if index:
            if not isinstance(value, tuple):
                raise PlanError(f"{reference}: [{index}] applies only to a list of points")
            if int(index) >= len(value):
                raise NotComputable(f"{reference}: the step found {len(value)} point(s)")
            value = value[int(index)]
        elif isinstance(value, Point):
            if name != "value" and name not in value.coords:
                raise PlanError(f"{reference}: a point has the coordinates {sorted(value.coords)}; "
                                "its value is '.value'")
            value = component(value, name)
        elif isinstance(value, Mapping):
            if name not in value:
                raise PlanError(f"{reference}: the record has {sorted(value)}")
            value = value[name]
        else:
            raise PlanError(f"{reference}: cannot take .{name} of a {type(value).__name__}")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, (Quantity, Point, Series, tuple)):
        return render(value)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _origin(value: Any) -> str:
    if isinstance(value, (Quantity, Point, Series)):
        return value.origin
    if isinstance(value, tuple):
        return "; ".join(point.origin for point in value) or "no points"
    return "record"


def _quantity_text(quantity: Quantity) -> str:
    return f"{quantity.value:.6g}" + ("" if quantity.unit == DIMENSIONLESS else f" {quantity.unit}")
