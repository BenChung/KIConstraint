import math

import pytest
from kipy.common_types import Polygon
from kipy.proto.common.types import base_types_pb2

from kiconstraint.mapping import (
    MappedArc,
    MappedBezier,
    MappedCircle,
    MappedRectangle,
    MappedSegment,
    map_shape,
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
