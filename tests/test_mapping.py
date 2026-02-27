import math

import pytest
from kipy.board_types import Pad, PadStackShape
from kipy.common_types import Polygon
from kipy.geometry import Vector2
from kipy.proto.common.types import base_types_pb2

from kiconstraint.mapping import (
    MappedArc,
    MappedBezier,
    MappedCircle,
    MappedPad,
    MappedPadChamferedRect,
    MappedPadCircle,
    MappedPadRectangle,
    MappedPadTrapezoid,
    MappedRectangle,
    MappedSegment,
    map_pad,
    map_shape,
    write_back_shapes,
)
from kiconstraint.solver import Sketch
from kiconstraint.solver.entities import Arc, Circle, Cubic, Line


# -- helpers ----------------------------------------------------------------

def _make_segment(x1, y1, x2, y2):
    """Create a kipy Segment from nanometer coordinates."""
    from kipy.common_types import Segment

    proto = base_types_pb2.GraphicShape()
    proto.segment.start.x_nm = x1
    proto.segment.start.y_nm = y1
    proto.segment.end.x_nm = x2
    proto.segment.end.y_nm = y2
    return Segment(proto)


def _make_arc(sx, sy, mx, my, ex, ey):
    """Create a kipy Arc from nanometer coordinates (start, mid, end)."""
    from kipy.common_types import Arc as KiArc

    proto = base_types_pb2.GraphicShape()
    proto.arc.start.x_nm = sx
    proto.arc.start.y_nm = sy
    proto.arc.mid.x_nm = mx
    proto.arc.mid.y_nm = my
    proto.arc.end.x_nm = ex
    proto.arc.end.y_nm = ey
    return KiArc(proto)


def _make_circle(cx, cy, rx, ry):
    """Create a kipy Circle from nanometer center and radius point."""
    from kipy.common_types import Circle as KiCircle

    proto = base_types_pb2.GraphicShape()
    proto.circle.center.x_nm = cx
    proto.circle.center.y_nm = cy
    proto.circle.radius_point.x_nm = rx
    proto.circle.radius_point.y_nm = ry
    return KiCircle(proto)


def _make_rectangle(tlx, tly, brx, bry):
    """Create a kipy Rectangle from nanometer corners."""
    from kipy.common_types import Rectangle

    proto = base_types_pb2.GraphicShape()
    proto.rectangle.top_left.x_nm = tlx
    proto.rectangle.top_left.y_nm = tly
    proto.rectangle.bottom_right.x_nm = brx
    proto.rectangle.bottom_right.y_nm = bry
    return Rectangle(proto)


def _make_bezier(sx, sy, c1x, c1y, c2x, c2y, ex, ey):
    """Create a kipy Bezier from nanometer coordinates."""
    from kipy.common_types import Bezier

    proto = base_types_pb2.GraphicShape()
    proto.bezier.start.x_nm = sx
    proto.bezier.start.y_nm = sy
    proto.bezier.control1.x_nm = c1x
    proto.bezier.control1.y_nm = c1y
    proto.bezier.control2.x_nm = c2x
    proto.bezier.control2.y_nm = c2y
    proto.bezier.end.x_nm = ex
    proto.bezier.end.y_nm = ey
    return Bezier(proto)


NM = 1_000_000  # 1 mm in nm


# -- tests ------------------------------------------------------------------


def test_map_segment():
    sketch = Sketch()
    seg = _make_segment(0, 0, 10 * NM, 5 * NM)
    m = map_shape(sketch, seg)

    assert isinstance(m, MappedSegment)
    assert isinstance(m.line, Line)

    result = sketch.solve()
    assert result.ok

    assert m.start.u == pytest.approx(0.0)
    assert m.start.v == pytest.approx(0.0)
    assert m.end.u == pytest.approx(10.0)
    assert m.end.v == pytest.approx(5.0)
    assert len(m.points) == 2


def test_map_arc():
    # Quarter-circle arc: start=(10mm,0), mid≈(7.07mm,7.07mm), end=(0,10mm)
    # Center should be at origin
    r = 10 * NM
    mid_v = int(r * math.cos(math.pi / 4))  # ~7.07mm in nm
    sketch = Sketch()
    arc = _make_arc(r, 0, mid_v, mid_v, 0, r)
    m = map_shape(sketch, arc)

    assert isinstance(m, MappedArc)
    assert isinstance(m.arc, Arc)

    result = sketch.solve()
    assert result.ok

    assert m.center.u == pytest.approx(0.0, abs=0.01)
    assert m.center.v == pytest.approx(0.0, abs=0.01)
    assert m.start.u == pytest.approx(10.0, abs=0.01)
    assert m.start.v == pytest.approx(0.0, abs=0.01)
    assert m.end.u == pytest.approx(0.0, abs=0.01)
    assert m.end.v == pytest.approx(10.0, abs=0.01)
    assert len(m.points) == 3


def test_map_circle():
    sketch = Sketch()
    circ = _make_circle(5 * NM, 5 * NM, 10 * NM, 5 * NM)
    m = map_shape(sketch, circ)

    assert isinstance(m, MappedCircle)
    assert isinstance(m.circle, Circle)

    result = sketch.solve()
    assert result.ok

    assert m.center.u == pytest.approx(5.0)
    assert m.center.v == pytest.approx(5.0)
    assert m.circle.radius.value == pytest.approx(5.0)
    assert len(m.points) == 1


def test_map_rectangle():
    sketch = Sketch()
    rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
    m = map_shape(sketch, rect)

    assert isinstance(m, MappedRectangle)

    result = sketch.solve()
    assert result.ok

    # Corners
    assert m.top_left.u == pytest.approx(0.0)
    assert m.top_left.v == pytest.approx(0.0)
    assert m.top_right.u == pytest.approx(10.0)
    assert m.top_right.v == pytest.approx(0.0)
    assert m.bottom_right.u == pytest.approx(10.0)
    assert m.bottom_right.v == pytest.approx(5.0)
    assert m.bottom_left.u == pytest.approx(0.0)
    assert m.bottom_left.v == pytest.approx(5.0)

    # Lines exist
    assert isinstance(m.top, Line)
    assert isinstance(m.right, Line)
    assert isinstance(m.bottom, Line)
    assert isinstance(m.left, Line)
    assert len(m.points) == 4


def test_map_bezier():
    sketch = Sketch()
    bez = _make_bezier(
        0, 0,
        3 * NM, 10 * NM,
        7 * NM, 10 * NM,
        10 * NM, 0,
    )
    m = map_shape(sketch, bez)

    assert isinstance(m, MappedBezier)
    assert isinstance(m.cubic, Cubic)

    result = sketch.solve()
    assert result.ok

    assert m.start.u == pytest.approx(0.0)
    assert m.start.v == pytest.approx(0.0)
    assert m.control1.u == pytest.approx(3.0)
    assert m.control1.v == pytest.approx(10.0)
    assert m.control2.u == pytest.approx(7.0)
    assert m.control2.v == pytest.approx(10.0)
    assert m.end.u == pytest.approx(10.0)
    assert m.end.v == pytest.approx(0.0)
    assert len(m.points) == 4


def test_map_unsupported():
    sketch = Sketch()
    proto = base_types_pb2.GraphicShape()
    # Set polygon geometry to satisfy the Polygon constructor assertion
    proto.polygon.polygons.add()
    polygon = Polygon(proto)
    with pytest.raises(TypeError, match="Unsupported shape type"):
        map_shape(sketch, polygon)


# -- helpers for geometric checks -------------------------------------------

def _dot(ux, uy, vx, vy):
    """Dot product of two 2D vectors."""
    return ux * vx + uy * vy


def _edge_vec(p1, p2):
    """Return the (dx, dy) vector from p1 to p2."""
    return (p2.u - p1.u, p2.v - p1.v)


def _edge_len(p1, p2):
    dx, dy = _edge_vec(p1, p2)
    return math.hypot(dx, dy)


def _assert_perpendicular(p1, p2, p3, p4, tol=1e-6):
    """Assert the edge p1→p2 is perpendicular to p3→p4."""
    ax, ay = _edge_vec(p1, p2)
    bx, by = _edge_vec(p3, p4)
    assert _dot(ax, ay, bx, by) == pytest.approx(0.0, abs=tol)


def _assert_parallel(p1, p2, p3, p4, tol=1e-6):
    """Assert the edge p1→p2 is parallel to p3→p4 (cross product ≈ 0)."""
    ax, ay = _edge_vec(p1, p2)
    bx, by = _edge_vec(p3, p4)
    cross = ax * by - ay * bx
    assert cross == pytest.approx(0.0, abs=tol)


# -- constraint enforcement tests -------------------------------------------


class TestRectangleConstraints:
    """Verify rectangle constraints enforce perpendicularity."""

    def test_rectangle_has_constraints(self):
        sketch = Sketch()
        rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, rect)
        assert len(m.constraints) == 3

    def test_rectangle_perpendicular_after_solve(self):
        """All four corners should be right angles after solving."""
        sketch = Sketch()
        rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, rect)
        sketch.solve()

        tl, tr, br, bl = m.top_left, m.top_right, m.bottom_right, m.bottom_left
        _assert_perpendicular(tl, tr, tl, bl)  # top ⊥ left
        _assert_perpendicular(tr, tl, tr, br)  # top ⊥ right
        _assert_perpendicular(br, tr, br, bl)  # right ⊥ bottom
        _assert_perpendicular(bl, tl, bl, br)  # left ⊥ bottom

    def test_rectangle_parallel_opposite_sides(self):
        sketch = Sketch()
        rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, rect)
        sketch.solve()

        tl, tr, br, bl = m.top_left, m.top_right, m.bottom_right, m.bottom_left
        _assert_parallel(tl, tr, bl, br)   # top ∥ bottom
        _assert_parallel(tl, bl, tr, br)   # left ∥ right

    def test_rectangle_stays_rectangular_after_perturbation(self):
        """Move one corner and re-solve; shape must remain rectangular."""
        sketch = Sketch()
        rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, rect)

        # Perturb top-left by shifting it and pinning it
        m.top_left.move(1.0, 1.0)
        sketch.dragged(m.top_left)
        result = sketch.solve()
        assert result.ok

        tl, tr, br, bl = m.top_left, m.top_right, m.bottom_right, m.bottom_left
        _assert_perpendicular(tl, tr, tl, bl)
        _assert_perpendicular(tr, tl, tr, br)
        _assert_perpendicular(br, tr, br, bl)
        _assert_perpendicular(bl, tl, bl, br)

    def test_rectangle_stays_rectangular_large_perturbation(self):
        """Large perturbation: move a corner far from original position."""
        sketch = Sketch()
        rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, rect)

        m.bottom_right.move(20.0, 15.0)
        sketch.dragged(m.bottom_right)
        result = sketch.solve()
        assert result.ok

        tl, tr, br, bl = m.top_left, m.top_right, m.bottom_right, m.bottom_left
        _assert_perpendicular(tl, tr, tl, bl)
        _assert_perpendicular(tr, tl, tr, br)
        _assert_perpendicular(br, tr, br, bl)
        _assert_perpendicular(bl, tl, bl, br)

    def test_rectangle_equal_opposite_sides_after_perturbation(self):
        """After perturbation, opposite sides should still be equal length."""
        sketch = Sketch()
        rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, rect)

        m.top_left.move(2.0, -1.0)
        sketch.dragged(m.top_left)
        result = sketch.solve()
        assert result.ok

        tl, tr, br, bl = m.top_left, m.top_right, m.bottom_right, m.bottom_left
        top_len = _edge_len(tl, tr)
        bottom_len = _edge_len(bl, br)
        left_len = _edge_len(tl, bl)
        right_len = _edge_len(tr, br)
        assert top_len == pytest.approx(bottom_len, rel=1e-4)
        assert left_len == pytest.approx(right_len, rel=1e-4)


class TestNoConstraintShapes:
    """Shapes without constraints should have empty constraint lists."""

    def test_segment_no_constraints(self):
        sketch = Sketch()
        seg = _make_segment(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, seg)
        assert m.constraints == []

    def test_arc_no_constraints(self):
        r = 10 * NM
        mid_v = int(r * math.cos(math.pi / 4))
        sketch = Sketch()
        arc = _make_arc(r, 0, mid_v, mid_v, 0, r)
        m = map_shape(sketch, arc)
        assert m.constraints == []

    def test_circle_no_constraints(self):
        sketch = Sketch()
        circ = _make_circle(5 * NM, 5 * NM, 10 * NM, 5 * NM)
        m = map_shape(sketch, circ)
        assert m.constraints == []

    def test_bezier_no_constraints(self):
        sketch = Sketch()
        bez = _make_bezier(0, 0, 3 * NM, 10 * NM, 7 * NM, 10 * NM, 10 * NM, 0)
        m = map_shape(sketch, bez)
        assert m.constraints == []


# -- write-back tests -------------------------------------------------------


def _to_mm(nm):
    return nm / NM


class TestWriteBackSegment:
    def test_positions_updated(self):
        sketch = Sketch()
        seg = _make_segment(0, 0, 10 * NM, 0)
        m = map_shape(sketch, seg)
        # Pin start and move end
        sketch.horizontal(m.line)
        m.end.move(5.0, 0.0)
        sketch.dragged(m.end)
        result = sketch.solve()
        assert result.ok
        modified = write_back_shapes([m], result)
        assert len(modified) == 1
        assert _to_mm(seg.start.x) == pytest.approx(m.start.u, abs=1e-4)
        assert _to_mm(seg.start.y) == pytest.approx(m.start.v, abs=1e-4)
        assert _to_mm(seg.end.x) == pytest.approx(m.end.u, abs=1e-4)
        assert _to_mm(seg.end.y) == pytest.approx(m.end.v, abs=1e-4)


class TestWriteBackRectangle:
    def test_positions_updated(self):
        sketch = Sketch()
        rect = _make_rectangle(0, 0, 10 * NM, 5 * NM)
        m = map_shape(sketch, rect)
        m.top_left.move(1.0, 1.0)
        sketch.dragged(m.top_left)
        result = sketch.solve()
        assert result.ok
        write_back_shapes([m], result)
        assert _to_mm(rect.top_left.x) == pytest.approx(m.top_left.u, abs=1e-4)
        assert _to_mm(rect.top_left.y) == pytest.approx(m.top_left.v, abs=1e-4)
        assert _to_mm(rect.bottom_right.x) == pytest.approx(m.bottom_right.u, abs=1e-4)
        assert _to_mm(rect.bottom_right.y) == pytest.approx(m.bottom_right.v, abs=1e-4)


class TestWriteBackCircle:
    def test_positions_updated(self):
        sketch = Sketch()
        circ = _make_circle(5 * NM, 5 * NM, 10 * NM, 5 * NM)
        m = map_shape(sketch, circ)
        result = sketch.solve()
        assert result.ok
        write_back_shapes([m], result)
        assert _to_mm(circ.center.x) == pytest.approx(m.center.u, abs=1e-4)
        assert _to_mm(circ.center.y) == pytest.approx(m.center.v, abs=1e-4)
        # radius_point should be at center + (radius, 0)
        assert _to_mm(circ.radius_point.x) == pytest.approx(
            m.center.u + m.circle.radius.value, abs=1e-4)
        assert _to_mm(circ.radius_point.y) == pytest.approx(m.center.v, abs=1e-4)


class TestWriteBackBezier:
    def test_positions_updated(self):
        sketch = Sketch()
        bez = _make_bezier(0, 0, 3 * NM, 10 * NM, 7 * NM, 10 * NM, 10 * NM, 0)
        m = map_shape(sketch, bez)
        result = sketch.solve()
        assert result.ok
        write_back_shapes([m], result)
        assert _to_mm(bez.start.x) == pytest.approx(m.start.u, abs=1e-4)
        assert _to_mm(bez.start.y) == pytest.approx(m.start.v, abs=1e-4)
        assert _to_mm(bez.control1.x) == pytest.approx(m.control1.u, abs=1e-4)
        assert _to_mm(bez.control1.y) == pytest.approx(m.control1.v, abs=1e-4)
        assert _to_mm(bez.control2.x) == pytest.approx(m.control2.u, abs=1e-4)
        assert _to_mm(bez.control2.y) == pytest.approx(m.control2.v, abs=1e-4)
        assert _to_mm(bez.end.x) == pytest.approx(m.end.u, abs=1e-4)
        assert _to_mm(bez.end.y) == pytest.approx(m.end.v, abs=1e-4)


class TestWriteBackArc:
    def test_positions_updated(self):
        r = 10 * NM
        mid_v = int(r * math.cos(math.pi / 4))
        sketch = Sketch()
        arc = _make_arc(r, 0, mid_v, mid_v, 0, r)
        m = map_shape(sketch, arc)
        result = sketch.solve()
        assert result.ok
        write_back_shapes([m], result)
        assert _to_mm(arc.start.x) == pytest.approx(m.start.u, abs=0.01)
        assert _to_mm(arc.start.y) == pytest.approx(m.start.v, abs=0.01)
        assert _to_mm(arc.end.x) == pytest.approx(m.end.u, abs=0.01)
        assert _to_mm(arc.end.y) == pytest.approx(m.end.v, abs=0.01)
        # mid should lie on the arc at the correct distance from center
        mid_x = _to_mm(arc.mid.x)
        mid_y = _to_mm(arc.mid.y)
        radius_mm = math.hypot(m.start.u - m.center.u, m.start.v - m.center.v)
        mid_r = math.hypot(mid_x - m.center.u, mid_y - m.center.v)
        assert mid_r == pytest.approx(radius_mm, abs=0.01)


class TestWriteBackErrors:
    def test_failed_solve_raises(self):
        sketch = Sketch()
        seg = _make_segment(0, 0, 10 * NM, 0)
        m = map_shape(sketch, seg)
        # Create an inconsistent system
        sketch.distance(m.start, m.end, 5.0)
        sketch.distance(m.start, m.end, 10.0)
        result = sketch.solve()
        assert not result.ok
        with pytest.raises(ValueError, match="solve failed"):
            write_back_shapes([m], result)


# -- pad helpers ----------------------------------------------------------------


def _make_pad(pos_x_mm, pos_y_mm, shape, size_x_mm, size_y_mm, **kwargs):
    """Create a kipy Pad with one copper layer of the given shape and size.

    Extra kwargs:
      trapezoid_delta_x_mm, trapezoid_delta_y_mm  – for trapezoid pads
      chamfer_ratio                                – for chamfered rect pads
      chamfered_corners                            – dict of corner bools
    """
    pad = Pad()
    pad.position = Vector2.from_xy_mm(pos_x_mm, pos_y_mm)

    layer = pad.padstack.copper_layers[0]
    layer.shape = shape
    layer.size = Vector2.from_xy_mm(size_x_mm, size_y_mm)

    if "trapezoid_delta_x_mm" in kwargs or "trapezoid_delta_y_mm" in kwargs:
        dx = kwargs.get("trapezoid_delta_x_mm", 0.0)
        dy = kwargs.get("trapezoid_delta_y_mm", 0.0)
        layer.trapezoid_delta = Vector2.from_xy_mm(dx, dy)

    if "chamfer_ratio" in kwargs:
        layer.chamfer_ratio = kwargs["chamfer_ratio"]

    if "chamfered_corners" in kwargs:
        corners = layer.chamfered_corners
        cc = kwargs["chamfered_corners"]
        corners.top_left = cc.get("top_left", False)
        corners.top_right = cc.get("top_right", False)
        corners.bottom_left = cc.get("bottom_left", False)
        corners.bottom_right = cc.get("bottom_right", False)

    return pad


# -- pad mapping tests ----------------------------------------------------------


class TestMapPadCircle:
    def test_creates_mapped_pad_circle(self):
        sketch = Sketch()
        pad = _make_pad(5, 10, PadStackShape.PSS_CIRCLE, 4, 4)
        m = map_pad(sketch, pad)

        assert isinstance(m, MappedPad)
        assert len(m.mapped_geometry) == 1
        assert isinstance(m.mapped_geometry[0], MappedPadCircle)

    def test_center_position(self):
        sketch = Sketch()
        pad = _make_pad(5, 10, PadStackShape.PSS_CIRCLE, 4, 4)
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        assert m.position.u == pytest.approx(5.0, abs=0.01)
        assert m.position.v == pytest.approx(10.0, abs=0.01)

    def test_radius(self):
        sketch = Sketch()
        pad = _make_pad(0, 0, PadStackShape.PSS_CIRCLE, 6, 6)
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        assert layer.circle.radius.value == pytest.approx(3.0, abs=0.01)


class TestMapPadRectangle:
    def test_creates_mapped_pad_rectangle(self):
        sketch = Sketch()
        pad = _make_pad(0, 0, PadStackShape.PSS_RECTANGLE, 10, 6)
        m = map_pad(sketch, pad)

        assert len(m.mapped_geometry) == 1
        assert isinstance(m.mapped_geometry[0], MappedPadRectangle)

    def test_corner_positions(self):
        sketch = Sketch()
        pad = _make_pad(5, 5, PadStackShape.PSS_RECTANGLE, 10, 6)
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        assert layer.tl.u == pytest.approx(0.0, abs=0.01)
        assert layer.tl.v == pytest.approx(2.0, abs=0.01)
        assert layer.tr.u == pytest.approx(10.0, abs=0.01)
        assert layer.tr.v == pytest.approx(2.0, abs=0.01)
        assert layer.br.u == pytest.approx(10.0, abs=0.01)
        assert layer.br.v == pytest.approx(8.0, abs=0.01)
        assert layer.bl.u == pytest.approx(0.0, abs=0.01)
        assert layer.bl.v == pytest.approx(8.0, abs=0.01)

    def test_perpendicular_constraints(self):
        sketch = Sketch()
        pad = _make_pad(5, 5, PadStackShape.PSS_RECTANGLE, 10, 6)
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        _assert_perpendicular(layer.tl, layer.tr, layer.tl, layer.bl)
        _assert_perpendicular(layer.tr, layer.tl, layer.tr, layer.br)

    def test_roundrect_maps_as_rectangle(self):
        sketch = Sketch()
        pad = _make_pad(0, 0, PadStackShape.PSS_ROUNDRECT, 8, 4)
        m = map_pad(sketch, pad)

        assert len(m.mapped_geometry) == 1
        assert isinstance(m.mapped_geometry[0], MappedPadRectangle)


class TestMapPadTrapezoid:
    def test_creates_mapped_pad_trapezoid_vertical_skew(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_TRAPEZOID, 10, 6,
            trapezoid_delta_x_mm=2.0, trapezoid_delta_y_mm=0.0,
        )
        m = map_pad(sketch, pad)

        assert len(m.mapped_geometry) == 1
        assert isinstance(m.mapped_geometry[0], MappedPadTrapezoid)

    def test_vertical_skew_constraints(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_TRAPEZOID, 10, 6,
            trapezoid_delta_x_mm=2.0, trapezoid_delta_y_mm=0.0,
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        # With vertical skew, left and right should be parallel
        _assert_parallel(layer.tl, layer.bl, layer.tr, layer.br, tol=0.01)
        # Top and bottom should be equal length
        top_len = _edge_len(layer.tl, layer.tr)
        bottom_len = _edge_len(layer.bl, layer.br)
        assert top_len == pytest.approx(bottom_len, abs=0.01)

    def test_creates_mapped_pad_trapezoid_horizontal_skew(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_TRAPEZOID, 10, 6,
            trapezoid_delta_x_mm=0.0, trapezoid_delta_y_mm=1.5,
        )
        m = map_pad(sketch, pad)

        assert len(m.mapped_geometry) == 1
        assert isinstance(m.mapped_geometry[0], MappedPadTrapezoid)

    def test_horizontal_skew_constraints(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_TRAPEZOID, 10, 6,
            trapezoid_delta_x_mm=0.0, trapezoid_delta_y_mm=1.5,
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        # With horizontal skew, top and bottom should be parallel
        _assert_parallel(layer.tl, layer.tr, layer.bl, layer.br, tol=0.01)
        # Left and right should be equal length
        left_len = _edge_len(layer.tl, layer.bl)
        right_len = _edge_len(layer.tr, layer.br)
        assert left_len == pytest.approx(right_len, abs=0.01)


class TestMapPadChamferedRect:
    def test_creates_mapped_pad_chamfered_rect(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_CHAMFEREDRECT, 10, 6,
            chamfer_ratio=0.25,
            chamfered_corners={
                "top_left": True, "top_right": True,
                "bottom_left": True, "bottom_right": True,
            },
        )
        m = map_pad(sketch, pad)

        assert len(m.mapped_geometry) == 1
        assert isinstance(m.mapped_geometry[0], MappedPadChamferedRect)

    def test_all_corners_chamfered(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_CHAMFEREDRECT, 10, 6,
            chamfer_ratio=0.25,
            chamfered_corners={
                "top_left": True, "top_right": True,
                "bottom_left": True, "bottom_right": True,
            },
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        assert layer.chamfer_tl is not None
        assert layer.chamfer_tr is not None
        assert layer.chamfer_bl is not None
        assert layer.chamfer_br is not None

    def test_partial_chamfer(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_CHAMFEREDRECT, 10, 6,
            chamfer_ratio=0.25,
            chamfered_corners={
                "top_left": True, "top_right": False,
                "bottom_left": False, "bottom_right": True,
            },
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        assert layer.chamfer_tl is not None
        assert layer.chamfer_tr is None
        assert layer.chamfer_bl is None
        assert layer.chamfer_br is not None

    def test_perpendicular_construction_lines(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_CHAMFEREDRECT, 10, 6,
            chamfer_ratio=0.25,
            chamfered_corners={
                "top_left": True, "top_right": True,
                "bottom_left": True, "bottom_right": True,
            },
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        layer = m.mapped_geometry[0]
        _assert_perpendicular(
            layer.construction_h.p1, layer.construction_h.p2,
            layer.construction_v.p1, layer.construction_v.p2,
        )


# -- pad write-back tests -------------------------------------------------------


class TestWriteBackPadCircle:
    def test_size_updated(self):
        sketch = Sketch()
        pad = _make_pad(0, 0, PadStackShape.PSS_CIRCLE, 4, 4)
        m = map_pad(sketch, pad)

        # Change the radius via a constraint
        layer = m.mapped_geometry[0]
        sketch.diameter(layer.circle, 10.0)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        new_size = pad.padstack.copper_layers[0].size
        assert _to_mm(new_size.x) == pytest.approx(10.0, abs=0.01)
        assert _to_mm(new_size.y) == pytest.approx(10.0, abs=0.01)

    def test_position_updated(self):
        sketch = Sketch()
        pad = _make_pad(5, 10, PadStackShape.PSS_CIRCLE, 4, 4)
        m = map_pad(sketch, pad)

        m.position.move(8.0, 12.0)
        sketch.dragged(m.position)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        new_pos = pad.position
        assert _to_mm(new_pos.x) == pytest.approx(8.0, abs=0.01)
        assert _to_mm(new_pos.y) == pytest.approx(12.0, abs=0.01)


class TestWriteBackPadRectangle:
    def test_size_updated(self):
        sketch = Sketch()
        pad = _make_pad(5, 5, PadStackShape.PSS_RECTANGLE, 10, 6)
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        new_size = pad.padstack.copper_layers[0].size
        assert _to_mm(new_size.x) == pytest.approx(10.0, abs=0.01)
        assert _to_mm(new_size.y) == pytest.approx(6.0, abs=0.01)

    def test_size_after_perturbation(self):
        sketch = Sketch()
        pad = _make_pad(5, 5, PadStackShape.PSS_RECTANGLE, 10, 6)
        m = map_pad(sketch, pad)

        layer = m.mapped_geometry[0]
        # Move tr corner to make the rectangle wider
        layer.tr.move(12.0, 2.0)
        sketch.dragged(layer.tr)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        new_size = pad.padstack.copper_layers[0].size
        # Width and height should reflect the new rectangle dimensions
        width = _to_mm(new_size.x)
        height = _to_mm(new_size.y)
        # The rectangle should still be valid (positive dimensions)
        assert width > 0
        assert height > 0


class TestWriteBackPadTrapezoid:
    def test_vertical_skew_size_and_delta(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_TRAPEZOID, 10, 6,
            trapezoid_delta_x_mm=2.0, trapezoid_delta_y_mm=0.0,
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        layer = pad.padstack.copper_layers[0]
        size = layer.size
        delta = layer.trapezoid_delta
        # Original construction width was 10mm, height average was 6mm
        assert _to_mm(size.x) == pytest.approx(10.0, abs=0.1)
        assert _to_mm(size.y) == pytest.approx(6.0, abs=0.1)
        # Delta x should be half the difference of left and right edge lengths
        assert _to_mm(delta.x) == pytest.approx(2.0, abs=0.1)
        assert _to_mm(delta.y) == pytest.approx(0.0, abs=0.1)

    def test_horizontal_skew_size_and_delta(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_TRAPEZOID, 10, 6,
            trapezoid_delta_x_mm=0.0, trapezoid_delta_y_mm=1.5,
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        layer = pad.padstack.copper_layers[0]
        size = layer.size
        delta = layer.trapezoid_delta
        assert _to_mm(size.y) == pytest.approx(6.0, abs=0.1)
        assert _to_mm(delta.x) == pytest.approx(0.0, abs=0.1)
        assert _to_mm(delta.y) == pytest.approx(1.5, abs=0.1)


class TestWriteBackPadChamferedRect:
    def test_size_updated(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_CHAMFEREDRECT, 10, 6,
            chamfer_ratio=0.25,
            chamfered_corners={
                "top_left": True, "top_right": True,
                "bottom_left": True, "bottom_right": True,
            },
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        layer = pad.padstack.copper_layers[0]
        size = layer.size
        assert _to_mm(size.x) == pytest.approx(10.0, abs=0.1)
        assert _to_mm(size.y) == pytest.approx(6.0, abs=0.1)

    def test_chamfer_ratio_updated(self):
        sketch = Sketch()
        pad = _make_pad(
            5, 5, PadStackShape.PSS_CHAMFEREDRECT, 10, 6,
            chamfer_ratio=0.25,
            chamfered_corners={
                "top_left": True, "top_right": True,
                "bottom_left": True, "bottom_right": True,
            },
        )
        m = map_pad(sketch, pad)
        result = sketch.solve()
        assert result.ok

        m.write_back()
        layer = pad.padstack.copper_layers[0]
        assert layer.chamfer_ratio == pytest.approx(0.25, abs=0.02)
