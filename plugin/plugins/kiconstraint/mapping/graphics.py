from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Union

from kipy.common_types import (
    Arc as KiArc,
    Bezier as KiBezier,
    Circle as KiCircle,
    GraphicShape as KiShape,
    Rectangle as KiRectangle,
    Segment as KiSegment,
)
from kipy.geometry import Vector2

from ..solver.constraints import Constraint, SolveResult
from ..solver.entities import Arc, Circle, Cubic, Line, Point
from ..solver.sketch import Sketch
from ._common import MappedGeometry, _to_mm, _v2


@dataclass(frozen=True)
class MappedSegment(MappedGeometry):
    source: KiSegment
    start: Point
    end: Point
    line: Line
    constraints: list[Constraint] = field(default_factory=list)

    @classmethod
    def create(cls, sketch: Sketch, seg: KiSegment) -> MappedSegment:
        p1 = sketch.point(_to_mm(seg.start.x), _to_mm(seg.start.y))
        p2 = sketch.point(_to_mm(seg.end.x), _to_mm(seg.end.y))
        line = sketch.line(p1, p2)
        return cls(source=seg, start=p1, end=p2, line=line)

    def write_back(self) -> None:
        self.source.start = _v2(self.start)
        self.source.end = _v2(self.end)

    @property
    def points(self) -> list[Point]:
        return [self.start, self.end]

    @property
    def lines(self) -> list[Line]:
        return [self.line]


@dataclass(frozen=True)
class MappedArc(MappedGeometry):
    source: KiArc
    center: Point
    start: Point
    end: Point
    arc: Arc
    constraints: list[Constraint] = field(default_factory=list)

    @classmethod
    def create(cls, sketch: Sketch, arc: KiArc) -> MappedArc:
        center_v = arc.center()
        if center_v is None:
            raise ValueError("Degenerate arc: cannot compute center")
        c = sketch.point(_to_mm(center_v.x), _to_mm(center_v.y))
        s = sketch.point(_to_mm(arc.start.x), _to_mm(arc.start.y))
        e = sketch.point(_to_mm(arc.end.x), _to_mm(arc.end.y))
        a = sketch.arc(c, s, e)
        return cls(source=arc, center=c, start=s, end=e, arc=a)

    def write_back(self) -> None:
        self.source.start = _v2(self.start)
        self.source.end = _v2(self.end)
        cx, cy = self.center.u, self.center.v
        sa = math.atan2(self.start.v - cy, self.start.u - cx)
        ea = math.atan2(self.end.v - cy, self.end.u - cx)
        sweep = (ea - sa) % (2 * math.pi)
        mid_angle = sa + sweep / 2
        radius = math.hypot(self.start.u - cx, self.start.v - cy)
        self.source.mid = Vector2.from_xy_mm(
            cx + radius * math.cos(mid_angle),
            cy + radius * math.sin(mid_angle),
        )

    @property
    def points(self) -> list[Point]:
        return [self.center, self.start, self.end]

    @property
    def lines(self) -> list[Line]:
        return []


@dataclass(frozen=True)
class MappedCircle(MappedGeometry):
    source: KiCircle
    center: Point
    circle: Circle
    constraints: list[Constraint] = field(default_factory=list)

    @classmethod
    def create(cls, sketch: Sketch, circ: KiCircle) -> MappedCircle:
        c = sketch.point(_to_mm(circ.center.x), _to_mm(circ.center.y))
        radius_mm = _to_mm(circ.radius())
        circle = sketch.circle(c, radius_mm)
        return cls(source=circ, center=c, circle=circle)

    def write_back(self) -> None:
        self.source.center = _v2(self.center)
        self.source.radius_point = Vector2.from_xy_mm(
            self.center.u + self.circle.radius.value, self.center.v,
        )

    @property
    def points(self) -> list[Point]:
        return [self.center]

    @property
    def lines(self) -> list[Line]:
        return []


@dataclass(frozen=True)
class MappedRectangle(MappedGeometry):
    source: KiRectangle
    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point
    top: Line
    right: Line
    bottom: Line
    left: Line
    constraints: list[Constraint]

    @classmethod
    def create(cls, sketch: Sketch, rect: KiRectangle) -> MappedRectangle:
        tl_x = _to_mm(rect.top_left.x)
        tl_y = _to_mm(rect.top_left.y)
        br_x = _to_mm(rect.bottom_right.x)
        br_y = _to_mm(rect.bottom_right.y)
        tl = sketch.point(tl_x, tl_y)
        tr = sketch.point(br_x, tl_y)
        br = sketch.point(br_x, br_y)
        bl = sketch.point(tl_x, br_y)
        top = sketch.line(tl, tr)
        right = sketch.line(tr, br)
        bottom = sketch.line(br, bl)
        left = sketch.line(bl, tl)
        constraints = [
            sketch.perpendicular(top, right),
            sketch.perpendicular(right, bottom),
            sketch.perpendicular(bottom, left),
        ]
        return cls(
            source=rect,
            top_left=tl, top_right=tr,
            bottom_right=br, bottom_left=bl,
            top=top, right=right,
            bottom=bottom, left=left,
            constraints=constraints,
        )

    def write_back(self) -> None:
        self.source.top_left = _v2(self.top_left)
        self.source.bottom_right = _v2(self.bottom_right)

    @property
    def points(self) -> list[Point]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    @property
    def lines(self) -> list[Line]:
        return [self.top, self.right, self.bottom, self.left]


@dataclass(frozen=True)
class MappedBezier(MappedGeometry):
    source: KiBezier
    start: Point
    control1: Point
    control2: Point
    end: Point
    cubic: Cubic
    constraints: list[Constraint] = field(default_factory=list)

    @classmethod
    def create(cls, sketch: Sketch, bez: KiBezier) -> MappedBezier:
        p1 = sketch.point(_to_mm(bez.start.x), _to_mm(bez.start.y))
        p2 = sketch.point(_to_mm(bez.control1.x), _to_mm(bez.control1.y))
        p3 = sketch.point(_to_mm(bez.control2.x), _to_mm(bez.control2.y))
        p4 = sketch.point(_to_mm(bez.end.x), _to_mm(bez.end.y))
        cubic = sketch.cubic(p1, p2, p3, p4)
        return cls(
            source=bez, start=p1, control1=p2,
            control2=p3, end=p4, cubic=cubic,
        )

    def write_back(self) -> None:
        self.source.start = _v2(self.start)
        self.source.control1 = _v2(self.control1)
        self.source.control2 = _v2(self.control2)
        self.source.end = _v2(self.end)

    @property
    def points(self) -> list[Point]:
        return [self.start, self.control1, self.control2, self.end]

    @property
    def lines(self) -> list[Line]:
        return []


MappedShape = Union[MappedSegment, MappedArc, MappedCircle, MappedRectangle, MappedBezier]

_SHAPE_MAP: dict[type, type] = {
    KiSegment: MappedSegment,
    KiArc: MappedArc,
    KiCircle: MappedCircle,
    KiRectangle: MappedRectangle,
    KiBezier: MappedBezier,
}


def map_shape(sketch: Sketch, shape: KiShape) -> MappedShape:
    """Map a KiCad GraphicShape into solver entities within *sketch*.

    Coordinates are converted from nanometers to millimeters.
    """
    for ki_type, mapped_cls in _SHAPE_MAP.items():
        if isinstance(shape, ki_type):
            return mapped_cls.create(sketch, shape)
    raise TypeError(f"Unsupported shape type: {type(shape).__name__}")


def write_back_shapes(
    mapped: Sequence[MappedShape],
    result: SolveResult,
) -> list[KiShape]:
    """Write solved positions back to the original KiCAD graphic objects.

    Returns the list of modified source objects (for passing to
    ``board.update_items()``).

    Raises :class:`ValueError` if the solve did not succeed.
    """
    if not result.ok:
        raise ValueError(
            f"Cannot write back: solve failed (result_code={result.result_code})"
        )

    sources: list[KiShape] = []
    for m in mapped:
        m.write_back()
        sources.append(m.source)
    return sources
