from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from kipy.common_types import (
    Arc as KiArc,
    Bezier as KiBezier,
    Circle as KiCircle,
    GraphicShape as KiShape,
    Rectangle as KiRectangle,
    Segment as KiSegment,
)

from .solver.entities import Arc, Circle, Cubic, Line, Point
from .solver.sketch import Sketch

_NM_PER_MM = 1_000_000


def _to_mm(nm: int) -> float:
    return nm / _NM_PER_MM


# ---------------------------------------------------------------------------
# Mapped shape dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappedSegment:
    source: KiSegment
    start: Point
    end: Point
    line: Line

    @property
    def points(self) -> list[Point]:
        return [self.start, self.end]


@dataclass(frozen=True)
class MappedArc:
    source: KiArc
    center: Point
    start: Point
    end: Point
    arc: Arc

    @property
    def points(self) -> list[Point]:
        return [self.center, self.start, self.end]


@dataclass(frozen=True)
class MappedCircle:
    source: KiCircle
    center: Point
    circle: Circle

    @property
    def points(self) -> list[Point]:
        return [self.center]


@dataclass(frozen=True)
class MappedRectangle:
    source: KiRectangle
    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point
    top: Line
    right: Line
    bottom: Line
    left: Line

    @property
    def points(self) -> list[Point]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]


@dataclass(frozen=True)
class MappedBezier:
    source: KiBezier
    start: Point
    control1: Point
    control2: Point
    end: Point
    cubic: Cubic

    @property
    def points(self) -> list[Point]:
        return [self.start, self.control1, self.control2, self.end]


MappedShape = Union[MappedSegment, MappedArc, MappedCircle, MappedRectangle, MappedBezier]


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------


def map_shape(sketch: Sketch, shape: KiShape) -> MappedShape:
    """Map a KiCad GraphicShape into solver entities within *sketch*.

    Coordinates are converted from nanometers to millimeters.
    """
    if isinstance(shape, KiSegment):
        return _map_segment(sketch, shape)
    if isinstance(shape, KiArc):
        return _map_arc(sketch, shape)
    if isinstance(shape, KiCircle):
        return _map_circle(sketch, shape)
    if isinstance(shape, KiRectangle):
        return _map_rectangle(sketch, shape)
    if isinstance(shape, KiBezier):
        return _map_bezier(sketch, shape)
    raise TypeError(f"Unsupported shape type: {type(shape).__name__}")


def _map_segment(sketch: Sketch, seg: KiSegment) -> MappedSegment:
    p1 = sketch.point(_to_mm(seg.start.x), _to_mm(seg.start.y))
    p2 = sketch.point(_to_mm(seg.end.x), _to_mm(seg.end.y))
    line = sketch.line(p1, p2)
    return MappedSegment(source=seg, start=p1, end=p2, line=line)


def _map_arc(sketch: Sketch, arc: KiArc) -> MappedArc:
    center_v = arc.center()
    if center_v is None:
        raise ValueError("Degenerate arc: cannot compute center")
    c = sketch.point(_to_mm(center_v.x), _to_mm(center_v.y))
    s = sketch.point(_to_mm(arc.start.x), _to_mm(arc.start.y))
    e = sketch.point(_to_mm(arc.end.x), _to_mm(arc.end.y))
    a = sketch.arc(c, s, e)
    return MappedArc(source=arc, center=c, start=s, end=e, arc=a)


def _map_circle(sketch: Sketch, circ: KiCircle) -> MappedCircle:
    c = sketch.point(_to_mm(circ.center.x), _to_mm(circ.center.y))
    radius_mm = _to_mm(circ.radius())
    circle = sketch.circle(c, radius_mm)
    return MappedCircle(source=circ, center=c, circle=circle)


def _map_rectangle(sketch: Sketch, rect: KiRectangle) -> MappedRectangle:
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

    return MappedRectangle(
        source=rect,
        top_left=tl, top_right=tr,
        bottom_right=br, bottom_left=bl,
        top=top, right=right,
        bottom=bottom, left=left,
    )


def _map_bezier(sketch: Sketch, bez: KiBezier) -> MappedBezier:
    p1 = sketch.point(_to_mm(bez.start.x), _to_mm(bez.start.y))
    p2 = sketch.point(_to_mm(bez.control1.x), _to_mm(bez.control1.y))
    p3 = sketch.point(_to_mm(bez.control2.x), _to_mm(bez.control2.y))
    p4 = sketch.point(_to_mm(bez.end.x), _to_mm(bez.end.y))
    cubic = sketch.cubic(p1, p2, p3, p4)
    return MappedBezier(
        source=bez, start=p1, control1=p2,
        control2=p3, end=p4, cubic=cubic,
    )
