from __future__ import annotations

import slvs

from .constraints import Constraint, SolveResult
from .entities import (
    Arc,
    Circle,
    Cubic,
    Distance,
    Entity,
    Line,
    Normal,
    Point,
    Workplane,
)

_BASE_GROUP = 1
_USER_GROUP = 2


def _wrap_constraint(raw: slvs.Slvs_Constraint) -> Constraint:
    return Constraint(handle=raw["h"], type_code=raw["type"])


class Sketch:
    """High-level interface to the SolveSpace constraint solver.

    On creation, clears global solver state, sets up the default XY workplane
    (group 1), and uses group 2 for all user-created entities and constraints.
    """

    def __init__(self) -> None:
        slvs.clear_sketch()
        self._wp = slvs.add_base_2d(_BASE_GROUP)
        # 3D normal for XY plane (identity quaternion) — needed by circle/arc
        self._normal = Normal(slvs.add_normal_3d(_USER_GROUP, 1.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------

    def point(self, u: float, v: float, *, fixed: bool = False) -> Point:
        raw = slvs.add_point_2d(_USER_GROUP, u, v, self._wp)
        p = Point(raw)
        if fixed:
            slvs.dragged(_USER_GROUP, raw, self._wp)
        return p

    def point_3d(self, x: float, y: float, z: float, *, fixed: bool = False) -> Point:
        raw = slvs.add_point_3d(_USER_GROUP, x, y, z)
        p = Point(raw)
        if fixed:
            slvs.dragged(_USER_GROUP, raw)
        return p

    def line(self, p1: Point, p2: Point) -> Line:
        raw = slvs.add_line_2d(_USER_GROUP, p1._raw, p2._raw, self._wp)
        return Line(raw, p1, p2)

    def line_3d(self, p1: Point, p2: Point) -> Line:
        raw = slvs.add_line_3d(_USER_GROUP, p1._raw, p2._raw)
        return Line(raw, p1, p2)

    def circle(self, center: Point, radius_value: float) -> Circle:
        dist_raw = slvs.add_distance(_USER_GROUP, radius_value, self._wp)
        dist = Distance(dist_raw)
        raw = slvs.add_circle(
            _USER_GROUP, self._normal._raw, center._raw, dist_raw, self._wp
        )
        return Circle(raw, center, dist)

    def arc(self, center: Point, start: Point, end: Point) -> Arc:
        raw = slvs.add_arc(
            _USER_GROUP, self._normal._raw, center._raw, start._raw, end._raw, self._wp
        )
        return Arc(raw, center, start, end)

    def cubic(self, p1: Point, p2: Point, p3: Point, p4: Point) -> Cubic:
        raw = slvs.add_cubic(
            _USER_GROUP, p1._raw, p2._raw, p3._raw, p4._raw, self._wp
        )
        return Cubic(raw, p1, p2, p3, p4)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def coincident(self, a: Point, b: Point) -> Constraint:
        return _wrap_constraint(slvs.coincident(_USER_GROUP, a._raw, b._raw, self._wp))

    def distance(self, a: Entity, b: Entity, value: float) -> Constraint:
        return _wrap_constraint(slvs.distance(_USER_GROUP, a._raw, b._raw, value, self._wp))

    def distance_proj(self, ptA: Entity, ptB: Entity, axis: Line, value: float) -> Constraint:
        return _wrap_constraint(slvs.add_constraint(
            _USER_GROUP, int(slvs.ConstraintType.PROJ_PT_DISTANCE), self._wp,
            value, ptA._raw, ptB._raw, axis._raw
        ))

    def horizontal(self, entity: Line) -> Constraint:
        return _wrap_constraint(slvs.horizontal(_USER_GROUP, entity._raw, self._wp))

    def vertical(self, entity: Line) -> Constraint:
        return _wrap_constraint(slvs.vertical(_USER_GROUP, entity._raw, self._wp))

    def parallel(self, a: Line, b: Line) -> Constraint:
        return _wrap_constraint(slvs.parallel(_USER_GROUP, a._raw, b._raw, self._wp))

    def perpendicular(self, a: Line, b: Line, *, inverse: bool = False) -> Constraint:
        return _wrap_constraint(
            slvs.perpendicular(_USER_GROUP, a._raw, b._raw, self._wp, inverse)
        )

    def equal(self, a: Entity, b: Entity) -> Constraint:
        return _wrap_constraint(slvs.equal(_USER_GROUP, a._raw, b._raw, self._wp))

    def tangent(self, a: Entity, b: Entity) -> Constraint:
        return _wrap_constraint(slvs.tangent(_USER_GROUP, a._raw, b._raw, self._wp))

    def midpoint(self, point: Point, line: Line) -> Constraint:
        return _wrap_constraint(slvs.midpoint(_USER_GROUP, point._raw, line._raw, self._wp))

    def symmetric(self, a: Point, b: Point, line_or_plane: Entity | None = None) -> Constraint:
        if line_or_plane is not None:
            return _wrap_constraint(
                slvs.symmetric(_USER_GROUP, a._raw, b._raw, line_or_plane._raw, self._wp)
            )
        return _wrap_constraint(slvs.symmetric(_USER_GROUP, a._raw, b._raw, wp=self._wp))

    def symmetric_h(self, a: Point, b: Point) -> Constraint:
        return _wrap_constraint(slvs.symmetric_h(_USER_GROUP, a._raw, b._raw, self._wp))

    def symmetric_v(self, a: Point, b: Point) -> Constraint:
        return _wrap_constraint(slvs.symmetric_v(_USER_GROUP, a._raw, b._raw, self._wp))

    def angle(self, a: Line, b: Line, degrees: float, *, inverse: bool = False) -> Constraint:
        return _wrap_constraint(
            slvs.angle(_USER_GROUP, a._raw, b._raw, degrees, self._wp, inverse)
        )

    def diameter(self, circle: Circle, value: float) -> Constraint:
        return _wrap_constraint(slvs.diameter(_USER_GROUP, circle._raw, value))

    def ratio(self, a: Line, b: Line, value: float) -> Constraint:
        return _wrap_constraint(slvs.ratio(_USER_GROUP, a._raw, b._raw, value, self._wp))

    def length_diff(self, a: Line, b: Line, value: float) -> Constraint:
        return _wrap_constraint(slvs.length_diff(_USER_GROUP, a._raw, b._raw, value, self._wp))

    def on_line(self, point: Point, line: Line) -> Constraint:
        raw = slvs.add_constraint(
            _USER_GROUP,
            int(slvs.ConstraintType.PT_ON_LINE),
            self._wp,
            0.0,
            point._raw,
            slvs.E_NONE,
            line._raw,
            slvs.E_NONE,
        )
        return _wrap_constraint(raw)

    def on_circle(self, point: Point, circle: Circle) -> Constraint:
        raw = slvs.add_constraint(
            _USER_GROUP,
            int(slvs.ConstraintType.PT_ON_CIRCLE),
            self._wp,
            0.0,
            point._raw,
            slvs.E_NONE,
            circle._raw,
            slvs.E_NONE,
        )
        return _wrap_constraint(raw)

    def equal_angle(self, l1: Line, l2: Line, l3: Line, l4: Line) -> Constraint:
        return _wrap_constraint(
            slvs.equal_angle(_USER_GROUP, l1._raw, l2._raw, l3._raw, l4._raw, self._wp)
        )

    def equal_radius(self, c1: Circle | Arc, c2: Circle | Arc) -> Constraint:
        raw = slvs.add_constraint(
            _USER_GROUP,
            int(slvs.ConstraintType.EQUAL_RADIUS),
            self._wp,
            0.0,
            slvs.E_NONE,
            slvs.E_NONE,
            c1._raw,
            c2._raw,
        )
        return _wrap_constraint(raw)

    def dragged(self, point: Point) -> Constraint:
        return _wrap_constraint(slvs.dragged(_USER_GROUP, point._raw, self._wp))

    def same_orientation(self, n1: Normal, n2: Normal) -> Constraint:
        return _wrap_constraint(slvs.same_orientation(_USER_GROUP, n1._raw, n2._raw))

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def solve(self) -> SolveResult:
        raw = slvs.solve_sketch(_USER_GROUP, True)
        return SolveResult(raw)
