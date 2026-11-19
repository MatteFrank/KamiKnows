"""Deterministic numeric operations over extracted data.

Sources (``compute.sources``) turn extracted data, such as digitized plots,
into Series. The primitives take and return Series, Points and Quantities,
so any composition of them is itself an operation:

- selection:  restrict, section, coordinate
- evaluation: evaluate, solve
- algebra:    combine, transform, component
- reduction:  reduce, project, argextremum
- calculus:   differentiate, cumulate

``derived`` writes common quantities as compositions of the primitives,
``registry`` lists every operation a plan may call, and ``plan.execute``
runs a JSON plan over a catalog of series.
"""

from kamiknows.compute.algebra import combine, component, transform
from kamiknows.compute.calculus import cumulate, differentiate
from kamiknows.compute.catalog import Catalog, Entry
from kamiknows.compute.derived import crossing, distance, fwhm, linear_fit, mean, normalize, quantile, std
from kamiknows.compute.errors import NotComputable
from kamiknows.compute.evaluate import evaluate, solve
from kamiknows.compute.metadata import describe
from kamiknows.compute.plan import execute
from kamiknows.compute.reduce import argextremum, project, reduce
from kamiknows.compute.select import coordinate, restrict, section
from kamiknows.compute.types import Axis, Point, Quantity, Series
from kamiknows.compute.units import DIMENSIONLESS, UNKNOWN, Unit, conversion_factor

__all__ = [
    "Axis", "Catalog", "DIMENSIONLESS", "Entry", "NotComputable", "Point", "Quantity", "Series",
    "UNKNOWN", "Unit", "argextremum", "combine", "component", "conversion_factor", "coordinate",
    "crossing", "cumulate", "describe", "differentiate", "distance", "evaluate", "execute", "fwhm",
    "linear_fit", "mean", "normalize", "project", "quantile", "reduce", "restrict", "section", "solve",
    "std", "transform",
]
