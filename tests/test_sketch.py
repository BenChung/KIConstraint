import math

import pytest

from kiconstraint.solver import Sketch


def test_empty_sketch_solves():
    sketch = Sketch()
    result = sketch.solve()
    assert result.ok


def test_fixed_point():
    sketch = Sketch()
    p = sketch.point(3.0, 7.0, fixed=True)
    result = sketch.solve()
    assert result.ok
    assert p.u == pytest.approx(3.0)
    assert p.v == pytest.approx(7.0)


def test_horizontal_line():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(5, 3)
    line = sketch.line(p1, p2)
    sketch.horizontal(line)
    result = sketch.solve()
    assert result.ok
    assert p1.v == pytest.approx(p2.v)


def test_vertical_line():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(3, 5)
    line = sketch.line(p1, p2)
    sketch.vertical(line)
    result = sketch.solve()
    assert result.ok
    assert p1.u == pytest.approx(p2.u)


def test_distance_constraint():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(5, 0)
    sketch.distance(p1, p2, 10.0)
    result = sketch.solve()
    assert result.ok
    dx = p2.u - p1.u
    dy = p2.v - p1.v
    assert math.hypot(dx, dy) == pytest.approx(10.0)


def test_perpendicular():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(5, 0)
    p3 = sketch.point(0, 5)
    l1 = sketch.line(p1, p2)
    l2 = sketch.line(p1, p3)
    sketch.perpendicular(l1, l2)
    result = sketch.solve()
    assert result.ok
    # dot product of direction vectors should be ~0
    d1x = p2.u - p1.u
    d1y = p2.v - p1.v
    d2x = p3.u - p1.u
    d2y = p3.v - p1.v
    dot = d1x * d2x + d1y * d2y
    assert dot == pytest.approx(0.0, abs=1e-6)


def test_parallel():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(5, 3)
    p3 = sketch.point(0, 2, fixed=True)
    p4 = sketch.point(5, 5)
    l1 = sketch.line(p1, p2)
    l2 = sketch.line(p3, p4)
    sketch.parallel(l1, l2)
    result = sketch.solve()
    assert result.ok
    # cross product of direction vectors should be ~0
    d1x = p2.u - p1.u
    d1y = p2.v - p1.v
    d2x = p4.u - p3.u
    d2y = p4.v - p3.v
    cross = d1x * d2y - d1y * d2x
    assert cross == pytest.approx(0.0, abs=1e-6)


def test_equal_length():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(5, 0)
    p3 = sketch.point(0, 3, fixed=True)
    p4 = sketch.point(0, 8)
    l1 = sketch.line(p1, p2)
    l2 = sketch.line(p3, p4)
    sketch.equal(l1, l2)
    result = sketch.solve()
    assert result.ok
    len1 = math.hypot(p2.u - p1.u, p2.v - p1.v)
    len2 = math.hypot(p4.u - p3.u, p4.v - p3.v)
    assert len1 == pytest.approx(len2)


def test_coincident():
    sketch = Sketch()
    p1 = sketch.point(1, 2)
    p2 = sketch.point(3, 4)
    sketch.coincident(p1, p2)
    result = sketch.solve()
    assert result.ok
    assert p1.u == pytest.approx(p2.u)
    assert p1.v == pytest.approx(p2.v)


def test_midpoint():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(10, 0, fixed=True)
    line = sketch.line(p1, p2)
    mid = sketch.point(3, 1)
    sketch.midpoint(mid, line)
    result = sketch.solve()
    assert result.ok
    assert mid.u == pytest.approx((p1.u + p2.u) / 2)
    assert mid.v == pytest.approx((p1.v + p2.v) / 2)


def test_circle_diameter():
    sketch = Sketch()
    center = sketch.point(0, 0, fixed=True)
    circ = sketch.circle(center, 5.0)
    sketch.diameter(circ, 20.0)
    result = sketch.solve()
    assert result.ok
    assert circ.radius.value == pytest.approx(10.0)


def test_angle():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(10, 0)
    p3 = sketch.point(10, 10)
    l1 = sketch.line(p1, p2)
    l2 = sketch.line(p1, p3)
    sketch.horizontal(l1)
    sketch.angle(l1, l2, 45.0)
    result = sketch.solve()
    assert result.ok
    d1x = p2.u - p1.u
    d1y = p2.v - p1.v
    d2x = p3.u - p1.u
    d2y = p3.v - p1.v
    cos_a = (d1x * d2x + d1y * d2y) / (math.hypot(d1x, d1y) * math.hypot(d2x, d2y))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
    assert angle_deg == pytest.approx(45.0, abs=0.1)


def test_distance_proj_horizontal():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(10, 5)
    # Axis along X
    a1 = sketch.point(0, 0, fixed=True)
    a2 = sketch.point(1, 0, fixed=True)
    axis = sketch.line(a1, a2)
    sketch.distance_proj(p1, p2, axis, 7.0)
    result = sketch.solve()
    assert result.ok
    assert abs(p2.u - p1.u) == pytest.approx(7.0, abs=1e-6)


def test_distance_proj_vertical():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(10, 5)
    # Axis along Y
    a1 = sketch.point(0, 0, fixed=True)
    a2 = sketch.point(0, 1, fixed=True)
    axis = sketch.line(a1, a2)
    sketch.distance_proj(p1, p2, axis, 3.0)
    result = sketch.solve()
    assert result.ok
    assert abs(p2.v - p1.v) == pytest.approx(3.0, abs=1e-6)


def test_on_line():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(10, 10, fixed=True)
    line = sketch.line(p1, p2)
    p3 = sketch.point(3, 7)
    sketch.on_line(p3, line)
    result = sketch.solve()
    assert result.ok
    # p3 should satisfy line equation: y = x (for this specific line)
    assert p3.u == pytest.approx(p3.v, abs=1e-6)


def test_solve_underconstrained():
    sketch = Sketch()
    sketch.point(1, 2)
    result = sketch.solve()
    assert result.ok
    assert result.dof > 0


def test_solve_inconsistent():
    sketch = Sketch()
    p1 = sketch.point(0, 0, fixed=True)
    p2 = sketch.point(10, 0, fixed=True)
    # distance of 5 between points fixed 10 apart -> inconsistent
    sketch.distance(p1, p2, 5.0)
    result = sketch.solve()
    assert not result.ok
