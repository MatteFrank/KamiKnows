"""The primitive operations of the compute package, on series with known answers."""

import numpy as np
import pytest

from kamiknows.compute import (
    DIMENSIONLESS, Axis, NotComputable, Quantity, Series, Unit, argextremum, combine, component,
    coordinate, cumulate, differentiate, evaluate, project, reduce, restrict, section, solve, transform,
)

GEV, MEV, AU = Unit.parse("GeV"), Unit.parse("MeV"), Unit.parse("A.U.")


def histogram(values, edges, value_scale="linear", name="h"):
    edges = np.asarray(edges, dtype=float)
    return Series(name, (Axis("x", GEV),), Axis("y", AU, value_scale),
                  centers=((edges[:-1] + edges[1:]) / 2)[:, None], values=np.asarray(values, dtype=float),
                  lower=edges[:-1, None], upper=edges[1:, None])


def samples(x, values, value_scale="linear", name="c"):
    return Series(name, (Axis("x", GEV),), Axis("y", AU, value_scale),
                  centers=np.asarray(x, dtype=float)[:, None], values=np.asarray(values, dtype=float))


def grid(values):
    """2D cells of unit size; values[row][column] sits at x = column, y = row."""
    values = np.asarray(values, dtype=float)
    rows, columns = np.indices(values.shape)
    centers = np.column_stack([columns.ravel(), rows.ravel()]).astype(float)
    cell = Unit.parse("cell_index")
    return Series("g", (Axis("x", cell), Axis("y", cell)), Axis("z", MEV),
                  centers=centers, values=values.ravel(), lower=centers - 0.5, upper=centers + 0.5)


# --- selection --------------------------------------------------------------

def test_restrict_keeps_bins_by_center_and_converts_limits():
    h = histogram([1, 2, 3, 4], [0, 10, 20, 30, 40])
    assert restrict(h, "x", 10, 30).values.tolist() == [2, 3]
    assert restrict(h, "x", Quantity(10_000, MEV), None).values.tolist() == [2, 3, 4]
    assert restrict(h, "y", 2.5).values.tolist() == [3, 4]
    with pytest.raises(NotComputable):
        restrict(h, "x", 30, 10)


def test_section_cuts_a_row_of_cells():
    g = grid([[1, 2, 3], [4, 5, 6]])
    row = section(g, "y", 1)
    assert row.dim == 1 and row.domain[0].name == "x" and row.values.tolist() == [4, 5, 6]
    assert section(g, "x", 2.2).values.tolist() == [3, 6]
    with pytest.raises(NotComputable):
        section(g, "y", 7)
    with pytest.raises(NotComputable):
        section(row, "x", 0)


def test_coordinate_gives_centers_and_bounds():
    h = histogram([1, 2], [0, 10, 30])
    assert coordinate(h, "x").values.tolist() == [5, 20]
    widths = combine(coordinate(h, "x", "upper"), coordinate(h, "x", "lower"), "sub")
    assert widths.values.tolist() == [10, 20] and widths.value_axis.unit == GEV
    with pytest.raises(NotComputable):
        coordinate(samples([0, 1], [1, 1]), "x", "lower")


# --- evaluation ---------------------------------------------------------------

def test_evaluate_finds_the_containing_bin_without_interpolating():
    h = histogram([1, 2, 3], [0, 10, 20, 30])
    point = evaluate(h, {"x": 12})
    assert point.value.value == 2 and point.bounds == {"x": (10, 20)}
    assert evaluate(h, {"x": 10}).value.value == 2          # shared edge: upper bin
    assert evaluate(h, {"x": 30}).value.value == 3          # overall top edge is closed
    assert evaluate(h, {"x": Quantity(25_000, MEV)}).value.value == 3
    with pytest.raises(NotComputable, match="outside"):
        evaluate(h, {"x": 31})
    with pytest.raises(NotComputable):
        evaluate(h, {"y": 1})


def test_evaluate_skips_gaps_between_bins():
    h = Series("h", (Axis("x", GEV),), Axis("y", AU), centers=[[5.0], [25.0]], values=[1.0, 2.0],
               lower=[[0.0], [20.0]], upper=[[10.0], [30.0]])
    with pytest.raises(NotComputable, match="gap"):
        evaluate(h, {"x": 15})


def test_evaluate_reads_a_cell():
    assert evaluate(grid([[1, 2, 3], [4, 5, 6]]), {"x": 2, "y": 1}).value.value == 6


def test_evaluate_interpolates_samples_as_drawn():
    assert evaluate(samples([0, 2], [0, 4]), {"x": 0.5}).value.value == pytest.approx(1)
    # On a log10 value axis the drawn segment is straight in log space: geometric midpoint.
    assert evaluate(samples([0, 1], [1, 100], "log10"), {"x": 0.5}).value.value == pytest.approx(10)
    with pytest.raises(NotComputable):
        evaluate(samples([0, 1], [1, 2]), {"x": 2})


def test_solve_interpolates_sample_crossings_by_direction():
    curve = samples([0, 1, 2, 3, 4], [0, 2, 0, 2, 0])
    assert [p.coords["x"].value for p in solve(curve, 1)] == pytest.approx([0.5, 1.5, 2.5, 3.5])
    assert [p.coords["x"].value for p in solve(curve, 1, "up")] == pytest.approx([0.5, 2.5])
    assert [p.coords["x"].value for p in solve(curve, 1, "down")] == pytest.approx([1.5, 3.5])
    assert solve(curve, 5) == ()


def test_solve_on_a_histogram_returns_the_bin_entered():
    h = histogram([1, 3, 2, 5], [0, 10, 20, 30, 40])
    entered = solve(h, 2.5, "up")
    assert [p.bounds["x"] for p in entered] == [(10, 20), (30, 40)]


# --- algebra ------------------------------------------------------------------

def test_combine_quantities_with_units():
    difference = combine(Quantity(2, GEV), Quantity(500, MEV), "sub")
    assert difference.value == pytest.approx(1.5) and difference.unit == GEV
    ratio = combine(Quantity(6, AU), Quantity(3, AU), "div")
    assert ratio.value == 2 and ratio.unit == DIMENSIONLESS
    assert combine(Quantity(3, GEV), 2, "mul").unit == GEV
    with pytest.raises(NotComputable):
        combine(Quantity(1, GEV), Quantity(1, AU), "add")
    with pytest.raises(NotComputable):
        combine(Quantity(1, AU), 0, "div")


def test_combine_series_aligns_on_the_left_elements():
    a = histogram([1, 2, 3], [0, 10, 20, 30])
    b = histogram([5, 5], [0, 20, 40])                     # coarser binning
    assert combine(a, b, "sub").values.tolist() == [-4, -3, -2]
    shifted = histogram([1, 1], [20, 30, 40])
    partial = combine(a, shifted, "add")
    assert partial.values.tolist() == [4]                  # only the overlapping bin
    assert any("dropped" in caveat for caveat in partial.caveats)
    assert combine(a, a, "sub").values.tolist() == [0, 0, 0]


def test_combine_drops_division_by_zero_in_series():
    ratio = combine(histogram([2, 4], [0, 1, 2]), histogram([0, 2], [0, 1, 2]), "div")
    assert ratio.values.tolist() == [2] and ratio.value_axis.unit == DIMENSIONLESS


def test_transform_functions_and_units():
    assert transform(Quantity(3, GEV), "power", 2).unit == GEV ** 2
    assert transform(Quantity(4, GEV ** 2), "power", 0.5).value == 2
    assert transform(Quantity(-2, AU), "abs").value == 2
    assert transform(Quantity(100, DIMENSIONLESS), "log10").value == 2
    with pytest.raises(NotComputable):
        transform(Quantity(100, MEV), "log10")
    with pytest.raises(NotComputable):
        transform(Quantity(1, GEV), "power")
    logged = transform(histogram([0, 10], [0, 1, 2]).derive("h", value_axis=Axis("y", DIMENSIONLESS)), "log10")
    assert logged.values.tolist() == [1]


def test_component_reads_coordinates_bounds_and_value():
    point = evaluate(histogram([1, 2], [0, 10, 20]), {"x": 15})
    assert component(point, "value").value == 2
    assert component(point, "x").value == 15
    assert component(point, "x", "lower").value == 10 and component(point, "x", "upper").value == 20
    with pytest.raises(NotComputable):
        component(point, "z")


# --- reduction ----------------------------------------------------------------

def test_reduce_values_and_integral():
    h = histogram([1, 2, 3], [0, 10, 20, 40])
    assert reduce(h, "sum").value == 6
    assert reduce(h, "mean").value == 2
    assert (reduce(h, "min").value, reduce(h, "max").value) == (1, 3)
    assert reduce(h, "count").value == 3
    integral = reduce(h, "integral")
    assert integral.value == 1 * 10 + 2 * 10 + 3 * 20 and integral.unit == AU * GEV
    assert reduce(samples([0, 1, 2], [0, 1, 2]), "integral").value == pytest.approx(2)
    with pytest.raises(NotComputable):
        reduce(restrict(h, "x", 100), "sum")


def test_project_reduces_along_the_other_axis():
    g = grid([[1, 2, 3], [4, 5, 6]])
    assert project(g, "x", "sum").values.tolist() == [5, 7, 9]
    assert project(g, "y", "max").values.tolist() == [3, 6]
    integral = project(g, "x", "integral")
    assert integral.values.tolist() == [5, 7, 9] and integral.value_axis.unit == MEV * Unit.parse("cell_index")
    with pytest.raises(NotComputable):
        project(section(g, "y", 0), "x")


def test_argextremum_global_with_ties_and_local_with_plateaus():
    h = histogram([1, 5, 2, 5, 1], [0, 1, 2, 3, 4, 5])
    assert [p.coords["x"].value for p in argextremum(h)] == [1.5, 3.5]
    curve = samples(range(8), [0, 3, 3, 1, 4, 2, 2, 5])
    assert [p.coords["x"].value for p in argextremum(curve, "max", "local")] == [1, 4]
    assert [p.coords["x"].value for p in argextremum(curve, "min", "local")] == [3, 5]
    with pytest.raises(NotComputable):
        argextremum(grid([[1, 2], [3, 4]]), "max", "local")


def test_argextremum_of_a_heatmap_gives_the_cell():
    peak, = argextremum(grid([[1, 9, 3], [4, 5, 6]]))
    assert (peak.coords["x"].value, peak.coords["y"].value, peak.value.value) == (1, 0, 9)


# --- calculus -----------------------------------------------------------------

def test_differentiate_a_line():
    slope = differentiate(samples([0, 1, 3], [1, 4, 10]))
    assert slope.values.tolist() == pytest.approx([3, 3])
    assert slope.centers[:, 0].tolist() == [0.5, 2]
    assert slope.value_axis.unit == AU / GEV


def test_cumulate_histogram_is_exact_and_flat_across_gaps():
    h = Series("h", (Axis("x", GEV),), Axis("y", AU), centers=[[5.0], [25.0]], values=[1.0, 2.0],
               lower=[[0.0], [20.0]], upper=[[10.0], [30.0]])
    running = cumulate(h)
    assert running.centers[:, 0].tolist() == [0, 10, 20, 30]
    assert running.values.tolist() == [0, 10, 10, 30]
    assert running.values[-1] == reduce(h, "integral").value
    assert running.value_axis.unit == AU * GEV


def test_cumulate_samples_matches_the_integral():
    curve = samples([0, 1, 2, 4], [0, 1, 4, 16])
    assert cumulate(curve).values[-1] == pytest.approx(reduce(curve, "integral").value)


# --- compositions: the primitives are a basis ---------------------------------

def test_gaussian_statistics_come_from_compositions():
    edges = np.linspace(-10, 10, 2001)
    centers = (edges[:-1] + edges[1:]) / 2
    mu, sigma = 1.505, 2.0                                  # on a bin center: a single peak bin
    h = histogram(np.exp(-0.5 * ((centers - mu) / sigma) ** 2), edges)
    x = coordinate(h, "x")
    total = reduce(h, "sum")
    mean = combine(reduce(combine(x, h, "mul"), "sum"), total, "div")
    assert mean.value == pytest.approx(mu, abs=1e-3)
    deviation = combine(x, mean, "sub")
    variance = combine(reduce(combine(transform(deviation, "power", 2), h, "mul"), "sum"), total, "div")
    assert transform(variance, "power", 0.5).value == pytest.approx(sigma, rel=1e-3)
    median, = solve(cumulate(h), combine(reduce(h, "integral"), 2, "div"))
    assert component(median, "x").value == pytest.approx(mu, abs=1e-3)
    peak, = argextremum(h)
    left, right = solve(h, combine(component(peak, "value"), 2, "div"))
    fwhm = combine(component(right, "x"), component(left, "x"), "sub")
    assert fwhm.value == pytest.approx(2 * np.sqrt(2 * np.log(2)) * sigma, abs=0.02)
    assert fwhm.unit == GEV


def test_origin_records_the_expression():
    a, b = histogram([1, 4], [0, 1, 2], name="A"), histogram([2, 2], [0, 1, 2], name="B")
    ratio = combine(reduce(a, "max"), reduce(b, "max"), "div")
    assert ratio.origin == "div(max(A), max(B))"
    tied = argextremum(b)
    assert [p.coords["x"].origin for p in tied] == ["argmax(B, global)[0].x", "argmax(B, global)[1].x"]
