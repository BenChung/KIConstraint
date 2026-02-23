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
from kipy.board_types import (
    Pad,
    PadStack,
    PadStackShape
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

@dataclass(frozen=True)
class MappedPad:
    source: Pad
    pad_stack: PadStack

    position: Point
    mapped_geometry
    constraints

def map_pad(sketch: Sketch, shape: Pad) -> MappedPad:
    """Map a KiCad Pad into solver entities within sketch"""
    geom = []
    cstrs = []
    center = sketch.point(_to_mm(shape.position.x), _to_mm(shape.position.y))
    for layer in shape.copper_layers:
        if layer.shape == PadStackShape.PSS_UNKNOWN or layer.shape == PSS_OVAL:
            # not supported
        else if layer.shape == PadStackShape.PSS_CIRCLE or (layer.shape == PadStackShape.PSS_CUSTOM and layer.custom_anchor_shape == PadStackShape.PSS_CIRCLE):
            sketch.circle(center, _to_mm(layer.size.x))
        else if layer.shape == PadStackShape.PSS_RECTANGLE or layer.shape == PadStackShape.PSS_ROUNDRECT or \
            (layer.shape == PadStackShape.PSS_CUSTOM and layer.custom_anchor_shape == PSS_RECTANGLE):
            # define a rectangle centered at center by adding a line, defining its midpoint as the center
            # then build the rectangle from that
            half_size = layer.size/2
            x = shape.position.x
            y = shape.position.y
            tl = sketch.point(_to_mm(x - half_size.x), _to_mm(y - half_size.y))
            tr = sketch.point(_to_mm(x + half_size.x), _to_mm(y - half_size.y))
            br = sketch.point(_to_mm(x + half_size.x), _to_mm(y + half_size.y))
            bl = sketch.point(_to_mm(x - half_size.x), _to_mm(y + half_size.y))
            construction = sketch.line(tl, br)
            top = sketch.line(tl, tr)
            bottom = sketch.line(bl, br)
            left = sketch.line(tl, bl)
            right = sketch.line(tr, br)
            cstrs.push(sketch.midpoint(center, construction))
            cstrs.push(sketch.perpendicular(top, left))
            cstrs.push(sketch.perpendicular(bottom, right))
            cstrs.push(sketch.perpendicular(left, bottom))
        else if layer.shape == PadStackShape.PSS_TRAPEZOID:
            trap_delta = layer.trap_delta / 2
            half_size = layer.size/2
            x = shape.position.x
            y = shape.position.y
            tl = sketch.point(_to_mm(x - half_size.x - trap_delta.y), _to_mm(y - half_size.y + trap_delta.x))
            tr = sketch.point(_to_mm(x + half_size.x + trap_delta.y), _to_mm(y - half_size.y - trap_delta.x))
            br = sketch.point(_to_mm(x + half_size.x - trap_delta.y), _to_mm(y + half_size.y + trap_delta.x))
            bl = sketch.point(_to_mm(x - half_size.x + trap_delta.y), _to_mm(y + half_size.y - trap_delta.x))
            top = sketch.line(tl, tr)
            bottom = sketch.line(bl, br)
            left = sketch.line(tl, bl)
            right = sketch.line(tr, br)
            if trap_delta.x != 0.0:
                # the trapezoid is skewed in the vertical axis
                mpl = sketch.point(_to_mm(x - half_size.x), _to_mm(y))
                mpr = sketch.point(_to_mm(x + half_size.x), _to_mm(y))
                construction = sketch.line(mpl, mpr)
                cstrs.push(sketch.midpoint(mpl, left))
                cstrs.push(sketch.midpoint(mpr, right))
                cstrs.push(sketch.perpendicular(construction, left))
                cstrs.push(sketch.midpoint(center, construction))
                cstrs.push(sketch.parallel(left, right))
                cstrs.push(sketch.equal(top, bottom))
            else:
                # the trapezoid is skewed in the horizontal axis
                mpt = sketch.point(_to_mm(x), _to_mm(y - half_size.y))
                mpb = sketch.point(_to_mm(x), _to_mm(y + half_size.y))
                construction = sketch.line(mpt, mpb)

                cstrs.push(sketch.midpoint(mpt, top))
                cstrs.push(sketch.midpoint(mpb, bottom))
                cstrs.push(sketch.perpendicular(construction, top))
                cstrs.push(sketch.midpoint(center, construction))
                cstrs.push(sketch.parallel(top, bottom))
                cstrs.push(sketch.equal(left, right))
        else if layer.shape == PadStackShape.PSS_CHAMFEREDRECT:
            corner_chamfer_fraction = layer.chamfer_ratio
            corner_chamfer_dist = min(layer.size.x, layer.size.y) * corner_chamfer_fraction
            half_size = layer.size/2
            x = shape.position.x
            y = shape.position.y
            tl = sketch.point(_to_mm(x - half_size.x), _to_mm(y - half_size.y))
            tr = sketch.point(_to_mm(x + half_size.x), _to_mm(y - half_size.y))
            br = sketch.point(_to_mm(x + half_size.x), _to_mm(y + half_size.y))
            bl = sketch.point(_to_mm(x - half_size.x), _to_mm(y + half_size.y))

            # build the corners
            vertical_construction_lines = []
            horizontal_construction_lines = []
            def build_chamfer(pt, adj_h, adj_v):
                p_h = sketch.point(_to_mm(x - half_size.x + adj_h), _to_mm(y - half_size.y))
                p_v = sketch.point(_to_mm(x - half_size.x), _to_mm(y - half_size.y + adj_v))
                chamfer = sketch.line(p_h, p_v)
                vcstr = sketch.line(pt, p_v)
                hcstr = sketch.line(pt, p_h)
                vertical_construction_lines.push(vcstr)
                horizontal_construction_lines.push(hcstr)
                cstrs.push(sketch.equal(vcstr, hcstr))
                return p_h, p_v

            tl_t = None
            tl_l = None
            if layer.chamfered_corners.top_left:
                tl_t, tl_l = build_chamfer(tl, corner_chamfer_dist, corner_chamfer_dist)
            else:
                tl_t = tl_l = tl
            
            tr_t = None
            tr_l = None
            if layer.chamfered_corners.top_left:
                tr_t, tr_l = build_chamfer(tr, -corner_chamfer_dist, corner_chamfer_dist)
            else:
                tr_t = tr_l = tr

            bl_l = None
            bl_b = None
            if layer.chamfered_corners.bottom_left:
                bl_b, bl_l = build_chamfer(bl, corner_chamfer_dist, -corner_chamfer_dist)
            else:
                bl_b = bl_l = bl

            br_r = None
            br_b = None
            if layer.chamfered_corners.bottom_right:
                br_b, br_r = build_chamfer(br, -corner_chamfer_dist, -corner_chamfer_dist)
            else:
                br_b = br_r = br
            
            if len(vertical_construction_lines) > 1:
                # make all the chamfers the same length
                for li in range(1, len(vertical_construction_lines)):
                    l = vertical_construction_lines[li]
                    ll = vertical_construction_lines[li-1]
                    cstrs.push(sketch.equal(l, li))
            top = sketch.line(tr_t, tl_t)
            left = sketch.line(tl_l, bl_l)
            right = sketch.line(tr_r, br_r)
            bottom = sketch.line(bl_l, br_r)

            
            tm = sketch.point(_to_mm(x), _to_mm(y - half_size.y))
            cstrs.push(sketch.midpoint(tm, top))
            
            lm = sketch.point(_to_mm(x - half_size.x), _to_mm(y))
            cstrs.push(sketch.midpoint(lm, left))
            
            rm = sketch.point(_to_mm(x + half_size.x), _to_mm(y))
            cstrs.push(sketch.midpoint(rm, right))
            
            bm = sketch.point(_to_mm(x), _to_mm(y + half_size.y))
            cstrs.push(sketch.midpoint(bm, bottom))
            
            construction_v = sketch.line(tm, bm)
            construction_h = sketch.line(lm, rm)
            cstrs.push(sketch.midpoint(center, construction_v))
            cstrs.push(sketch.midpoint(center, construction_h))
            cstrs.push(sketch.perpendicular(construction_h, construction_v))
            cstrs.push(sketch.parallel(construction_h, bottom))
            cstrs.push(sketch.parallel(construction_h, top))
            cstrs.push(sketch.parallel(construction_v, left))
            cstrs.push(sketch.parallel(construction_v, right))
            for l in vertical_construction_lines:
                cstrs.push(sketch.parallel(construction_v, l))
            for l in horizontal_construction_lines:
                cstrs.push(sketch.parallel(construction_h, l))


    return MappedPad(
        source=shape.pad, 
        pad_stack=shape.pad_stack,
        position=center,
        mapped_geometry=geom)
        


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
