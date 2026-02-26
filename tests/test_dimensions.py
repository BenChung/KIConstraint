import math

import pytest

from kiconstraint.dimensions import (
    Coincident,
    Distance,
    DimensionMapping,
    Equal,
    Horizontal,
    MappedEdgeDimension,
    MappedPointDimension,
    Midpoint,
    Parallel,
    Perpendicular,
    Vertical,
    parse_suffix,
)
from kiconstraint.solver import Sketch


# ---------------------------------------------------------------------------
# parse_suffix — individual tokens
# ---------------------------------------------------------------------------


class TestParseDistance:
    def test_integer(self):
        [spec] = parse_suffix("=3mm")
        assert spec == Distance(3.0)

    def test_float(self):
        [spec] = parse_suffix("=25.4mm")
        assert spec == Distance(25.4)

    def test_zero(self):
        [spec] = parse_suffix("=0mm")
        assert spec == Distance(0.0)


class TestParseParallel:
    def test_long(self):
        [spec] = parse_suffix("par(top)")
        assert spec == Parallel("top")

    def test_short(self):
        [spec] = parse_suffix("p(top)")
        assert spec == Parallel("top")


class TestParsePerpendicular:
    def test_long(self):
        [spec] = parse_suffix("perp(left)")
        assert spec == Perpendicular("left")

    def test_short(self):
        [spec] = parse_suffix("x(left)")
        assert spec == Perpendicular("left")


class TestParseCoincident:
    def test_long(self):
        [spec] = parse_suffix("coin(pt)")
        assert spec == Coincident("pt")

    def test_short(self):
        [spec] = parse_suffix("c(pt)")
        assert spec == Coincident("pt")


class TestParseVertical:
    def test_long(self):
        [spec] = parse_suffix("vert")
        assert isinstance(spec, Vertical)

    def test_short(self):
        [spec] = parse_suffix("v")
        assert isinstance(spec, Vertical)


class TestParseHorizontal:
    def test_long(self):
        [spec] = parse_suffix("horiz")
        assert isinstance(spec, Horizontal)

    def test_short(self):
        [spec] = parse_suffix("h")
        assert isinstance(spec, Horizontal)


class TestParseEqual:
    def test_long(self):
        [spec] = parse_suffix("eq(other)")
        assert spec == Equal("other")

    def test_short(self):
        [spec] = parse_suffix("e(other)")
        assert spec == Equal("other")


class TestParseMidpoint:
    def test_long(self):
        [spec] = parse_suffix("mid(edge)")
        assert spec == Midpoint("edge")

    def test_short(self):
        [spec] = parse_suffix("m(edge)")
        assert spec == Midpoint("edge")


# ---------------------------------------------------------------------------
# parse_suffix — multi-token and edge cases
# ---------------------------------------------------------------------------


class TestParseMultiple:
    def test_two_constraints(self):
        specs = parse_suffix("v, =3mm")
        assert len(specs) == 2
        assert isinstance(specs[0], Vertical)
        assert specs[1] == Distance(3.0)

    def test_three_constraints(self):
        specs = parse_suffix("h, eq(other), =10mm")
        assert len(specs) == 3
        assert isinstance(specs[0], Horizontal)
        assert specs[1] == Equal("other")
        assert specs[2] == Distance(10.0)

    def test_mixed_short_and_long(self):
        specs = parse_suffix("v, par(a), x(b)")
        assert len(specs) == 3
        assert isinstance(specs[0], Vertical)
        assert specs[1] == Parallel("a")
        assert specs[2] == Perpendicular("b")


class TestParseEdgeCases:
    def test_empty_string(self):
        assert parse_suffix("") == []

    def test_whitespace_only(self):
        assert parse_suffix("   ") == []

    def test_whitespace_around_tokens(self):
        specs = parse_suffix("  v ,  =5mm  ")
        assert len(specs) == 2
        assert isinstance(specs[0], Vertical)
        assert specs[1] == Distance(5.0)

    def test_unknown_bare_token(self):
        with pytest.raises(ValueError, match="unrecognized"):
            parse_suffix("bogus")

    def test_unknown_function(self):
        with pytest.raises(ValueError, match="unknown constraint"):
            parse_suffix("foo(bar)")

    def test_empty_token_in_list(self):
        with pytest.raises(ValueError, match="empty"):
            parse_suffix("v,,h")


# ---------------------------------------------------------------------------
# Helpers for constraint-spec tests
# ---------------------------------------------------------------------------


class _FakeDim:
    """Minimal stand-in for a Dimension source (used only in MappedEdge/Point)."""
    pass


def _make_line(sketch, x1, y1, x2, y2):
    """Create a line in the sketch and return it."""
    p1 = sketch.point(x1, y1)
    p2 = sketch.point(x2, y2)
    return sketch.line(p1, p2)


def _edge_entry(sketch, name, x1, y1, x2, y2):
    """Create a MappedEdgeDimension and return (entry, line)."""
    line = _make_line(sketch, x1, y1, x2, y2)
    entry = MappedEdgeDimension(source=_FakeDim(), name=name, line=line)
    return entry, line


def _point_entry(sketch, name, x, y):
    """Create a MappedPointDimension and return (entry, point)."""
    pt = sketch.point(x, y)
    entry = MappedPointDimension(source=_FakeDim(), name=name, point=pt)
    return entry, pt


# ---------------------------------------------------------------------------
# apply_to_line tests
# ---------------------------------------------------------------------------


class TestApplyDistanceLine:
    def test_constrains_length(self):
        sketch = Sketch()
        line = _make_line(sketch, 0, 0, 10, 0)
        Distance(5.0).apply_to_line(sketch, line, "top", DimensionMapping())
        assert sketch.solve().ok
        dist = math.hypot(line.p1.u - line.p2.u, line.p1.v - line.p2.v)
        assert dist == pytest.approx(5.0, abs=1e-6)


class TestApplyVerticalHorizontal:
    def test_vertical(self):
        sketch = Sketch()
        line = _make_line(sketch, 1, 0, 3, 5)
        Vertical().apply_to_line(sketch, line, "e", DimensionMapping())
        assert sketch.solve().ok
        assert line.p1.u == pytest.approx(line.p2.u, abs=1e-6)

    def test_horizontal(self):
        sketch = Sketch()
        line = _make_line(sketch, 0, 1, 5, 3)
        Horizontal().apply_to_line(sketch, line, "e", DimensionMapping())
        assert sketch.solve().ok
        assert line.p1.v == pytest.approx(line.p2.v, abs=1e-6)


class TestApplyParallel:
    def test_parallel(self):
        sketch = Sketch()
        entry_a, line_a = _edge_entry(sketch, "a", 0, 0, 10, 0)
        line_b = _make_line(sketch, 0, 5, 8, 7)
        dim_map = DimensionMapping(edges={"a": entry_a})
        Horizontal().apply_to_line(sketch, line_a, "a", dim_map)
        Parallel("a").apply_to_line(sketch, line_b, "b", dim_map)
        assert sketch.solve().ok
        dy_b = line_b.p2.v - line_b.p1.v
        assert dy_b == pytest.approx(0.0, abs=1e-6)


class TestApplyPerpendicular:
    def test_perpendicular(self):
        sketch = Sketch()
        entry_a, line_a = _edge_entry(sketch, "a", 0, 0, 10, 0)
        line_b = _make_line(sketch, 5, 0, 7, 8)
        dim_map = DimensionMapping(edges={"a": entry_a})
        Horizontal().apply_to_line(sketch, line_a, "a", dim_map)
        Perpendicular("a").apply_to_line(sketch, line_b, "b", dim_map)
        assert sketch.solve().ok
        dx_b = line_b.p2.u - line_b.p1.u
        assert dx_b == pytest.approx(0.0, abs=1e-6)


class TestApplyEqual:
    def test_equal_length(self):
        sketch = Sketch()
        entry_a, line_a = _edge_entry(sketch, "a", 0, 0, 10, 0)
        line_b = _make_line(sketch, 0, 5, 7, 5)
        dim_map = DimensionMapping(edges={"a": entry_a})
        Horizontal().apply_to_line(sketch, line_a, "a", dim_map)
        Horizontal().apply_to_line(sketch, line_b, "b", dim_map)
        Equal("a").apply_to_line(sketch, line_b, "b", dim_map)
        assert sketch.solve().ok
        len_a = abs(line_a.p2.u - line_a.p1.u)
        len_b = abs(line_b.p2.u - line_b.p1.u)
        assert len_a == pytest.approx(len_b, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_to_two_points tests
# ---------------------------------------------------------------------------


class TestApplyDistanceTwoPoints:
    def test_constrains_distance_between_unconnected_points(self):
        sketch = Sketch()
        p1 = sketch.point(0, 0)
        p2 = sketch.point(10, 5)
        Distance(7.0).apply_to_two_points(
            sketch, p1, p2, "gap", DimensionMapping(),
        )
        assert sketch.solve().ok
        dist = math.hypot(p1.u - p2.u, p1.v - p2.v)
        assert dist == pytest.approx(7.0, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_to_point tests
# ---------------------------------------------------------------------------


class TestApplyCoincident:
    def test_coincident_points(self):
        sketch = Sketch()
        pt_a = sketch.point(0, 0)
        entry_b, pt_b = _point_entry(sketch, "b", 5, 5)
        dim_map = DimensionMapping(points={"b": entry_b})
        Coincident("b").apply_to_point(sketch, pt_a, "a", dim_map)
        assert sketch.solve().ok
        assert pt_a.u == pytest.approx(pt_b.u, abs=1e-6)
        assert pt_a.v == pytest.approx(pt_b.v, abs=1e-6)


class TestApplyMidpoint:
    def test_point_midpoint_of_edge(self):
        sketch = Sketch()
        entry_seg, line = _edge_entry(sketch, "seg", 0, 0, 10, 0)
        pt = sketch.point(3, 2)
        dim_map = DimensionMapping(edges={"seg": entry_seg})
        Horizontal().apply_to_line(sketch, line, "seg", dim_map)
        Midpoint("seg").apply_to_point(sketch, pt, "mid_pt", dim_map)
        assert sketch.solve().ok
        mid_u = (line.p1.u + line.p2.u) / 2
        mid_v = (line.p1.v + line.p2.v) / 2
        assert pt.u == pytest.approx(mid_u, abs=1e-6)
        assert pt.v == pytest.approx(mid_v, abs=1e-6)


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


class TestApplyErrors:
    def test_coin_on_line_raises(self):
        sketch = Sketch()
        line = _make_line(sketch, 0, 0, 5, 0)
        with pytest.raises(ValueError, match="requires a point"):
            Coincident("other").apply_to_line(
                sketch, line, "e", DimensionMapping(),
            )

    def test_vert_on_point_raises(self):
        sketch = Sketch()
        pt = sketch.point(0, 0)
        with pytest.raises(ValueError, match="requires a line"):
            Vertical().apply_to_point(sketch, pt, "p", DimensionMapping())

    def test_missing_edge_reference(self):
        sketch = Sketch()
        line = _make_line(sketch, 0, 0, 5, 0)
        with pytest.raises(ValueError, match="not found"):
            Parallel("missing").apply_to_line(
                sketch, line, "e", DimensionMapping(),
            )

    def test_missing_point_reference(self):
        sketch = Sketch()
        pt = sketch.point(0, 0)
        with pytest.raises(ValueError, match="not found"):
            Coincident("missing").apply_to_point(
                sketch, pt, "p", DimensionMapping(),
            )

    def test_distance_on_point_raises(self):
        sketch = Sketch()
        pt = sketch.point(0, 0)
        with pytest.raises(ValueError, match="requires a line"):
            Distance(5.0).apply_to_point(sketch, pt, "p", DimensionMapping())

    def test_horiz_on_two_points_raises(self):
        sketch = Sketch()
        p1 = sketch.point(0, 0)
        p2 = sketch.point(5, 5)
        with pytest.raises(ValueError, match="cannot apply between"):
            Horizontal().apply_to_two_points(
                sketch, p1, p2, "gap", DimensionMapping(),
            )
