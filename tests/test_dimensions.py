import math

import pytest

from kiconstraint.dimensions import (
    Coincident,
    Distance,
    Equal,
    Horizontal,
    Midpoint,
    Parallel,
    Perpendicular,
    Vertical,
    parse_suffix,
)


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
# apply_dimension_constraints — integration with solver
# ---------------------------------------------------------------------------

from kiconstraint.dimensions import (
    DimensionMapping,
    MappedEdgeDimension,
    MappedPointDimension,
    apply_dimension_constraints,
)
from kiconstraint.solver import Sketch


class _FakeDimension:
    """Minimal stand-in for a kipy Dimension with prefix/suffix."""

    def __init__(self, prefix: str = "", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix


def _make_edge_entry(sketch, name, x1, y1, x2, y2, suffix=""):
    p1 = sketch.point(x1, y1)
    p2 = sketch.point(x2, y2)
    line = sketch.line(p1, p2)
    dim = _FakeDimension(prefix=f"{name}:", suffix=suffix)
    return MappedEdgeDimension(source=dim, name=name, line=line)


def _make_point_entry(sketch, name, x, y, suffix=""):
    pt = sketch.point(x, y)
    dim = _FakeDimension(prefix=f"{name}:", suffix=suffix)
    return MappedPointDimension(source=dim, name=name, point=pt)


class TestApplyDistance:
    def test_constrains_length(self):
        sketch = Sketch()
        entry = _make_edge_entry(sketch, "top", 0, 0, 10, 0, suffix="=5mm")
        dim_map = DimensionMapping(edges={"top": entry})
        constraints = apply_dimension_constraints(sketch, dim_map)
        assert len(constraints) == 1
        result = sketch.solve()
        assert result.ok
        dist = math.hypot(entry.line.p1.u - entry.line.p2.u,
                          entry.line.p1.v - entry.line.p2.v)
        assert dist == pytest.approx(5.0, abs=1e-6)


class TestApplyVerticalHorizontal:
    def test_vertical(self):
        sketch = Sketch()
        entry = _make_edge_entry(sketch, "e", 1, 0, 3, 5, suffix="v")
        dim_map = DimensionMapping(edges={"e": entry})
        apply_dimension_constraints(sketch, dim_map)
        assert sketch.solve().ok
        assert entry.line.p1.u == pytest.approx(entry.line.p2.u, abs=1e-6)

    def test_horizontal(self):
        sketch = Sketch()
        entry = _make_edge_entry(sketch, "e", 0, 1, 5, 3, suffix="h")
        dim_map = DimensionMapping(edges={"e": entry})
        apply_dimension_constraints(sketch, dim_map)
        assert sketch.solve().ok
        assert entry.line.p1.v == pytest.approx(entry.line.p2.v, abs=1e-6)


class TestApplyParallel:
    def test_parallel(self):
        sketch = Sketch()
        a = _make_edge_entry(sketch, "a", 0, 0, 10, 0, suffix="h")
        b = _make_edge_entry(sketch, "b", 0, 5, 8, 7, suffix="p(a)")
        dim_map = DimensionMapping(edges={"a": a, "b": b})
        apply_dimension_constraints(sketch, dim_map)
        assert sketch.solve().ok
        dy_b = b.line.p2.v - b.line.p1.v
        assert dy_b == pytest.approx(0.0, abs=1e-6)


class TestApplyPerpendicular:
    def test_perpendicular(self):
        sketch = Sketch()
        a = _make_edge_entry(sketch, "a", 0, 0, 10, 0, suffix="h")
        b = _make_edge_entry(sketch, "b", 5, 0, 7, 8, suffix="x(a)")
        dim_map = DimensionMapping(edges={"a": a, "b": b})
        apply_dimension_constraints(sketch, dim_map)
        assert sketch.solve().ok
        dx_b = b.line.p2.u - b.line.p1.u
        assert dx_b == pytest.approx(0.0, abs=1e-6)


class TestApplyEqual:
    def test_equal_length(self):
        sketch = Sketch()
        a = _make_edge_entry(sketch, "a", 0, 0, 10, 0, suffix="h")
        b = _make_edge_entry(sketch, "b", 0, 5, 7, 5, suffix="h, e(a)")
        dim_map = DimensionMapping(edges={"a": a, "b": b})
        apply_dimension_constraints(sketch, dim_map)
        assert sketch.solve().ok
        len_a = abs(a.line.p2.u - a.line.p1.u)
        len_b = abs(b.line.p2.u - b.line.p1.u)
        assert len_a == pytest.approx(len_b, abs=1e-6)


class TestApplyCoincident:
    def test_coincident_points(self):
        sketch = Sketch()
        a = _make_point_entry(sketch, "a", 0, 0, suffix="c(b)")
        b = _make_point_entry(sketch, "b", 5, 5)
        dim_map = DimensionMapping(points={"a": a, "b": b})
        apply_dimension_constraints(sketch, dim_map)
        assert sketch.solve().ok
        assert a.point.u == pytest.approx(b.point.u, abs=1e-6)
        assert a.point.v == pytest.approx(b.point.v, abs=1e-6)


class TestApplyMidpoint:
    def test_point_midpoint_of_edge(self):
        sketch = Sketch()
        edge = _make_edge_entry(sketch, "seg", 0, 0, 10, 0, suffix="h")
        pt = _make_point_entry(sketch, "mid_pt", 3, 2, suffix="m(seg)")
        dim_map = DimensionMapping(edges={"seg": edge}, points={"mid_pt": pt})
        apply_dimension_constraints(sketch, dim_map)
        assert sketch.solve().ok
        mid_u = (edge.line.p1.u + edge.line.p2.u) / 2
        mid_v = (edge.line.p1.v + edge.line.p2.v) / 2
        assert pt.point.u == pytest.approx(mid_u, abs=1e-6)
        assert pt.point.v == pytest.approx(mid_v, abs=1e-6)


class TestApplyErrors:
    def test_coin_on_edge_raises(self):
        sketch = Sketch()
        entry = _make_edge_entry(sketch, "e", 0, 0, 5, 0, suffix="c(other)")
        dim_map = DimensionMapping(edges={"e": entry})
        with pytest.raises(ValueError, match="coin.*not applicable to edges"):
            apply_dimension_constraints(sketch, dim_map)

    def test_vert_on_point_raises(self):
        sketch = Sketch()
        entry = _make_point_entry(sketch, "p", 0, 0, suffix="v")
        dim_map = DimensionMapping(points={"p": entry})
        with pytest.raises(ValueError, match="requires an edge"):
            apply_dimension_constraints(sketch, dim_map)

    def test_missing_edge_reference(self):
        sketch = Sketch()
        entry = _make_edge_entry(sketch, "e", 0, 0, 5, 0, suffix="p(missing)")
        dim_map = DimensionMapping(edges={"e": entry})
        with pytest.raises(ValueError, match="not found"):
            apply_dimension_constraints(sketch, dim_map)

    def test_missing_point_reference(self):
        sketch = Sketch()
        entry = _make_point_entry(sketch, "p", 0, 0, suffix="c(missing)")
        dim_map = DimensionMapping(points={"p": entry})
        with pytest.raises(ValueError, match="not found"):
            apply_dimension_constraints(sketch, dim_map)
