"""Units, value types and the catalog of the compute package."""

import dataclasses

import numpy as np
import pytest

from kamiknows.compute import (
    DIMENSIONLESS, UNKNOWN, Axis, Catalog, Entry, NotComputable, Series, Unit, conversion_factor,
)


@pytest.mark.parametrize("text", ["GeV", "GeV/c^2", "1/GeV", "A.U.", "cell_index", "MeV*mm", "unknown", "1"])
def test_unit_text_round_trips(text):
    assert str(Unit.parse(text)) == text


def test_unit_algebra_cancels_and_combines():
    gev = Unit.parse("GeV")
    assert gev / gev == DIMENSIONLESS
    assert gev * gev == Unit.parse("GeV^2")
    assert (gev * gev) ** 0.5 == gev
    assert Unit.parse("GeV/c^2") * Unit.parse("c^2") == gev


def test_unknown_unit_absorbs_products():
    assert UNKNOWN * Unit.parse("MeV") == UNKNOWN
    assert (UNKNOWN / UNKNOWN).known is False


def test_fractional_power_must_leave_integer_exponents():
    with pytest.raises(NotComputable):
        Unit.parse("GeV") ** 0.5


def test_conversion_factor():
    assert conversion_factor(Unit.parse("GeV"), Unit.parse("MeV")) == pytest.approx(1e3)
    assert conversion_factor(Unit.parse("1/MeV"), Unit.parse("1/GeV")) == pytest.approx(1e3)
    assert conversion_factor(Unit.parse("A.U."), Unit.parse("A.U.")) == 1.0
    for source, target in [("GeV", "mm"), ("A.U.", "GeV"), ("unknown", "unknown")]:
        with pytest.raises(NotComputable):
            conversion_factor(Unit.parse(source), Unit.parse(target))


def _bins(values=(1.0, 3.0, 2.0)) -> Series:
    edges = np.arange(len(values) + 1, dtype=float)
    return Series("h", (Axis("x", Unit.parse("GeV")),), Axis("y", Unit.parse("A.U.")),
                  centers=((edges[:-1] + edges[1:]) / 2)[:, None], values=np.asarray(values),
                  lower=edges[:-1, None], upper=edges[1:, None])


def test_series_is_read_only_and_exposes_points():
    series = _bins()
    with pytest.raises(ValueError):
        series.values[0] = 5.0
    point = series.point(1)
    assert point.coords["x"].value == 1.5
    assert point.value.value == 3.0
    assert point.bounds == {"x": (1.0, 2.0)}


@pytest.mark.parametrize("change", [
    {"values": np.array([1.0, np.nan, 2.0])},
    {"lower": np.zeros((2, 1))},
    {"upper": None},
    {"centers": np.array([[0.5], [1.5], [9.0]])},
    {"value_axis": Axis("x", UNKNOWN)},
])
def test_series_rejects_invalid_construction(change):
    with pytest.raises(ValueError):
        dataclasses.replace(_bins(), **change)


def test_unknown_axis_is_not_computable():
    with pytest.raises(NotComputable):
        _bins().axis("z")


def test_catalog_explains_missing_series():
    catalog = Catalog([Entry("a/s", _bins()), Entry("b", None, "plot not digitized (extraction status: needs_calibration)")])
    assert catalog.get("a/s").size == 3
    with pytest.raises(NotComputable, match="needs_calibration"):
        catalog.get("b")
    with pytest.raises(NotComputable, match="no extracted data"):
        catalog.get("c")
    with pytest.raises(ValueError):
        Catalog([Entry("a", _bins()), Entry("a", _bins())])
