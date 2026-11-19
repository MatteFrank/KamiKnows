"""Unit algebra: products, ratios, powers and conversions.

A unit is a product of symbols with integer exponents, written with ``*``,
``/`` and ``^`` (``GeV``, ``GeV/c^2``, ``1/GeV``). Symbols missing from the
conversion table (``cell_index``, ``A.U.``) are opaque base units: they
multiply and cancel, but convert only to themselves. ``UNKNOWN`` marks a unit
that was never established: it absorbs every product and never converts, so
an unverified unit cannot silently turn into a verified one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kamiknows.compute.errors import NotComputable

# Accepted symbol -> (base symbol, value of one symbol in the base).
CONVERSIONS: dict[str, tuple[str, float]] = {
    "eV": ("eV", 1.0), "keV": ("eV", 1e3), "MeV": ("eV", 1e6), "GeV": ("eV", 1e9), "TeV": ("eV", 1e12),
    "um": ("m", 1e-6), "mm": ("m", 1e-3), "cm": ("m", 1e-2), "m": ("m", 1.0),
}

_OPERATOR = re.compile(r"\s*([*/])\s*")
_POWER = re.compile(r"^(?P<symbol>.+?)\s*\^\s*(?P<exponent>-?\d+)$")


@dataclass(frozen=True)
class Unit:
    """An immutable product of symbols with non-zero integer exponents."""

    factors: tuple[tuple[str, int], ...] = ()
    known: bool = True

    @classmethod
    def parse(cls, text: str) -> Unit:
        text = text.strip()
        if text == "unknown":
            return UNKNOWN
        if text in ("1", "dimensionless"):
            return DIMENSIONLESS
        if not text:
            raise ValueError("empty unit: write 'unknown' or 'dimensionless'")
        parts = _OPERATOR.split(text)
        signs = [1] + [-1 if operator == "/" else 1 for operator in parts[1::2]]
        exponents: dict[str, int] = {}
        for sign, term in zip(signs, parts[0::2]):
            symbol, exponent = _parse_term(term, text)
            if symbol != "1":
                exponents[symbol] = exponents.get(symbol, 0) + sign * exponent
        return _from_exponents(exponents)

    @property
    def dimensionless(self) -> bool:
        return self.known and not self.factors

    def __mul__(self, other: Unit) -> Unit:
        if not (self.known and other.known):
            return UNKNOWN
        exponents = dict(self.factors)
        for symbol, exponent in other.factors:
            exponents[symbol] = exponents.get(symbol, 0) + exponent
        return _from_exponents(exponents)

    def __truediv__(self, other: Unit) -> Unit:
        return self * other ** -1

    def __pow__(self, power: float) -> Unit:
        """Fractional powers must leave integer exponents: sqrt(GeV^2) is GeV, sqrt(GeV) is refused."""
        if not self.known:
            return UNKNOWN
        exponents = {}
        for symbol, exponent in self.factors:
            scaled = exponent * power
            if scaled != int(scaled):
                raise NotComputable(f"unit {self} cannot be raised to the power {power}")
            exponents[symbol] = int(scaled)
        return _from_exponents(exponents)

    def __str__(self) -> str:
        if not self.known:
            return "unknown"
        if not self.factors:
            return "1"
        numerator = [_format_term(s, e) for s, e in self.factors if e > 0]
        denominator = [_format_term(s, -e) for s, e in self.factors if e < 0]
        return "*".join(numerator or ["1"]) + "".join("/" + term for term in denominator)


DIMENSIONLESS = Unit()
UNKNOWN = Unit(known=False)


def conversion_factor(source: Unit, target: Unit) -> float:
    """Return f such that a value expressed in ``source`` is f times that number in ``target``."""
    if not (source.known and target.known):
        raise NotComputable(f"cannot convert {source} to {target}: unit not established")
    source_base, source_scale = _to_base(source)
    target_base, target_scale = _to_base(target)
    if source_base != target_base:
        raise NotComputable(f"cannot convert {source} to {target}: different dimensions")
    return source_scale / target_scale


def _to_base(unit: Unit) -> tuple[Unit, float]:
    exponents: dict[str, int] = {}
    scale = 1.0
    for symbol, exponent in unit.factors:
        base, factor = CONVERSIONS.get(symbol, (symbol, 1.0))
        exponents[base] = exponents.get(base, 0) + exponent
        scale *= factor ** exponent
    return _from_exponents(exponents), scale


def _parse_term(term: str, text: str) -> tuple[str, int]:
    if not term:
        raise ValueError(f"malformed unit {text!r}")
    match = _POWER.match(term)
    return (match["symbol"], int(match["exponent"])) if match else (term, 1)


def _format_term(symbol: str, exponent: int) -> str:
    return symbol if exponent == 1 else f"{symbol}^{exponent}"


def _from_exponents(exponents: dict[str, int]) -> Unit:
    return Unit(tuple(sorted((s, e) for s, e in exponents.items() if e)))
