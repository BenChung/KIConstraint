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
    PadStackLayer,
    PadStackShape,
)

from .solver.constraints import Constraint
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
class ChamferCorner:
    """Geometry for a single chamfered corner."""
    p_h: Point
    p_v: Point
    chamfer: Line
    h_construction: Line
    v_construction: Line


@dataclass(frozen=True)
class MappedPadCircle:
    source: PadStackLayer
    center: Point
    circle: Circle
    constraints: list[Constraint]

    @property
    def points(self) -> list[Point]:
        return [self.center]


@dataclass(frozen=True)
class MappedPadRectangle:
    source: PadStackLayer
    center: Point
    tl: Point
    tr: Point
    br: Point
    bl: Point
    top: Line
    right: Line
    bottom: Line
    left: Line
    construction: Line
    constraints: list[Constraint]

    @property
    def points(self) -> list[Point]:
        return [self.center, self.tl, self.tr, self.br, self.bl]


@dataclass(frozen=True)
class MappedPadTrapezoid:
    source: PadStackLayer
    center: Point
    tl: Point
    tr: Point
    br: Point
    bl: Point
    top: Line
    right: Line
    bottom: Line
    left: Line
    midpoint_a: Point
    midpoint_b: Point
    construction: Line
    constraints: list[Constraint]

    @property
    def points(self) -> list[Point]:
        return [self.center, self.tl, self.tr, self.br, self.bl,
                self.midpoint_a, self.midpoint_b]


@dataclass(frozen=True)
class MappedPadChamferedRect:
    source: PadStackLayer
    center: Point
    tl: Point
    tr: Point
    br: Point
    bl: Point
    top: Line
    right: Line
    bottom: Line
    left: Line
    chamfer_tl: ChamferCorner | None
    chamfer_tr: ChamferCorner | None
    chamfer_bl: ChamferCorner | None
    chamfer_br: ChamferCorner | None
    top_mid: Point
    left_mid: Point
    right_mid: Point
    bottom_mid: Point
    construction_v: Line
    construction_h: Line
    constraints: list[Constraint]

    @property
    def points(self) -> list[Point]:
        pts: list[Point] = [self.center, self.tl, self.tr, self.br, self.bl,
                            self.top_mid, self.left_mid, self.right_mid,
                            self.bottom_mid]
        for chamfer in (self.chamfer_tl, self.chamfer_tr,
                        self.chamfer_bl, self.chamfer_br):
            if chamfer is not None:
                pts.extend([chamfer.p_h, chamfer.p_v])
        return pts


MappedPadLayer = Union[
    MappedPadCircle, MappedPadRectangle,
    MappedPadTrapezoid, MappedPadChamferedRect,
]


@dataclass(frozen=True)
class MappedPad:
    source: Pad
    pad_stack: PadStack
    position: Point
    mapped_geometry: list[MappedPadLayer]
    constraints: list[Constraint]

def _map_pad_circle(
    sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
) -> MappedPadCircle:
    circle = sketch.circle(center, _to_mm(layer.size.x / 2))
    return MappedPadCircle(
        source=layer, center=center, circle=circle, constraints=[],
    )


def _map_pad_rectangle(
    sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
) -> MappedPadRectangle:
    half_size = layer.size / 2
    tl = sketch.point(_to_mm(x - half_size.x), _to_mm(y - half_size.y))
    tr = sketch.point(_to_mm(x + half_size.x), _to_mm(y - half_size.y))
    br = sketch.point(_to_mm(x + half_size.x), _to_mm(y + half_size.y))
    bl = sketch.point(_to_mm(x - half_size.x), _to_mm(y + half_size.y))
    construction = sketch.line(tl, br)
    top = sketch.line(tl, tr)
    bottom = sketch.line(bl, br)
    left = sketch.line(tl, bl)
    right = sketch.line(tr, br)
    constraints = [
        sketch.midpoint(center, construction),
        sketch.perpendicular(top, left),
        sketch.perpendicular(bottom, right),
        sketch.perpendicular(left, bottom),
    ]
    return MappedPadRectangle(
        source=layer, center=center,
        tl=tl, tr=tr, br=br, bl=bl,
        top=top, right=right, bottom=bottom, left=left,
        construction=construction, constraints=constraints,
    )


def _map_pad_trapezoid(
    sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
) -> MappedPadTrapezoid:
    trap_delta = layer.trapezoid_delta / 2
    half_size = layer.size / 2
    tl = sketch.point(_to_mm(x - half_size.x - trap_delta.y),
                      _to_mm(y - half_size.y + trap_delta.x))
    tr = sketch.point(_to_mm(x + half_size.x + trap_delta.y),
                      _to_mm(y - half_size.y - trap_delta.x))
    br = sketch.point(_to_mm(x + half_size.x - trap_delta.y),
                      _to_mm(y + half_size.y + trap_delta.x))
    bl = sketch.point(_to_mm(x - half_size.x + trap_delta.y),
                      _to_mm(y + half_size.y - trap_delta.x))
    top = sketch.line(tl, tr)
    bottom = sketch.line(bl, br)
    left = sketch.line(tl, bl)
    right = sketch.line(tr, br)
    constraints: list[Constraint] = []
    if trap_delta.x != 0.0:
        # the trapezoid is skewed in the vertical axis
        mp_a = sketch.point(_to_mm(x - half_size.x), _to_mm(y))
        mp_b = sketch.point(_to_mm(x + half_size.x), _to_mm(y))
        construction = sketch.line(mp_a, mp_b)
        constraints.extend([
            sketch.midpoint(mp_a, left),
            sketch.midpoint(mp_b, right),
            sketch.perpendicular(construction, left),
            sketch.midpoint(center, construction),
            sketch.parallel(left, right),
            sketch.equal(top, bottom),
        ])
    else:
        # the trapezoid is skewed in the horizontal axis
        mp_a = sketch.point(_to_mm(x), _to_mm(y - half_size.y))
        mp_b = sketch.point(_to_mm(x), _to_mm(y + half_size.y))
        construction = sketch.line(mp_a, mp_b)
        constraints.extend([
            sketch.midpoint(mp_a, top),
            sketch.midpoint(mp_b, bottom),
            sketch.perpendicular(construction, top),
            sketch.midpoint(center, construction),
            sketch.parallel(top, bottom),
            sketch.equal(left, right),
        ])
    return MappedPadTrapezoid(
        source=layer, center=center,
        tl=tl, tr=tr, br=br, bl=bl,
        top=top, right=right, bottom=bottom, left=left,
        midpoint_a=mp_a, midpoint_b=mp_b,
        construction=construction, constraints=constraints,
    )


def _build_chamfer_corner(
    sketch: Sketch, pt: Point, adj_h: float, adj_v: float,
) -> ChamferCorner:
    """Build the chamfer geometry for one corner."""
    p_h = sketch.point(pt.x + _to_mm(adj_h), pt.y)
    p_v = sketch.point(pt.x, pt.y + _to_mm(adj_v))
    chamfer = sketch.line(p_h, p_v)
    v_construction = sketch.line(pt, p_v)
    h_construction = sketch.line(pt, p_h)
    return ChamferCorner(
        p_h=p_h, p_v=p_v, chamfer=chamfer,
        h_construction=h_construction, v_construction=v_construction,
    )


def _map_pad_chamfered_rect(
    sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
) -> MappedPadChamferedRect:
    chamfer_dist = min(layer.size.x, layer.size.y) * layer.chamfer_ratio
    half_size = layer.size / 2

    tl = sketch.point(_to_mm(x - half_size.x), _to_mm(y - half_size.y))
    tr = sketch.point(_to_mm(x + half_size.x), _to_mm(y - half_size.y))
    br = sketch.point(_to_mm(x + half_size.x), _to_mm(y + half_size.y))
    bl = sketch.point(_to_mm(x - half_size.x), _to_mm(y + half_size.y))

    constraints: list[Constraint] = []
    v_construction_lines: list[Line] = []
    h_construction_lines: list[Line] = []

    # Build each corner's chamfer
    chamfer_tl = None
    if layer.chamfered_corners.top_left:
        chamfer_tl = _build_chamfer_corner(sketch, tl, chamfer_dist, chamfer_dist)
        v_construction_lines.append(chamfer_tl.v_construction)
        h_construction_lines.append(chamfer_tl.h_construction)
        constraints.append(sketch.equal(chamfer_tl.v_construction,
                                        chamfer_tl.h_construction))
        tl_t, tl_l = chamfer_tl.p_h, chamfer_tl.p_v
    else:
        tl_t = tl_l = tl

    chamfer_tr = None
    if layer.chamfered_corners.top_right:
        chamfer_tr = _build_chamfer_corner(sketch, tr, -chamfer_dist, chamfer_dist)
        v_construction_lines.append(chamfer_tr.v_construction)
        h_construction_lines.append(chamfer_tr.h_construction)
        constraints.append(sketch.equal(chamfer_tr.v_construction,
                                        chamfer_tr.h_construction))
        tr_t, tr_r = chamfer_tr.p_h, chamfer_tr.p_v
    else:
        tr_t = tr_r = tr

    chamfer_bl = None
    if layer.chamfered_corners.bottom_left:
        chamfer_bl = _build_chamfer_corner(sketch, bl, chamfer_dist, -chamfer_dist)
        v_construction_lines.append(chamfer_bl.v_construction)
        h_construction_lines.append(chamfer_bl.h_construction)
        constraints.append(sketch.equal(chamfer_bl.v_construction,
                                        chamfer_bl.h_construction))
        bl_b, bl_l = chamfer_bl.p_h, chamfer_bl.p_v
    else:
        bl_b = bl_l = bl

    chamfer_br = None
    if layer.chamfered_corners.bottom_right:
        chamfer_br = _build_chamfer_corner(sketch, br, -chamfer_dist, -chamfer_dist)
        v_construction_lines.append(chamfer_br.v_construction)
        h_construction_lines.append(chamfer_br.h_construction)
        constraints.append(sketch.equal(chamfer_br.v_construction,
                                        chamfer_br.h_construction))
        br_b, br_r = chamfer_br.p_h, chamfer_br.p_v
    else:
        br_b = br_r = br

    # Make all chamfers the same length
    for i in range(1, len(v_construction_lines)):
        constraints.append(sketch.equal(v_construction_lines[i],
                                        v_construction_lines[i - 1]))

    # Edge lines connecting chamfer endpoints
    top = sketch.line(tr_t, tl_t)
    left = sketch.line(tl_l, bl_l)
    right = sketch.line(tr_r, br_r)
    bottom = sketch.line(bl_b, br_b)

    # Midpoints and construction lines for centering
    tm = sketch.point(_to_mm(x), _to_mm(y - half_size.y))
    lm = sketch.point(_to_mm(x - half_size.x), _to_mm(y))
    rm = sketch.point(_to_mm(x + half_size.x), _to_mm(y))
    bm = sketch.point(_to_mm(x), _to_mm(y + half_size.y))

    construction_v = sketch.line(tm, bm)
    construction_h = sketch.line(lm, rm)

    constraints.extend([
        sketch.midpoint(tm, top),
        sketch.midpoint(lm, left),
        sketch.midpoint(rm, right),
        sketch.midpoint(bm, bottom),
        sketch.midpoint(center, construction_v),
        sketch.midpoint(center, construction_h),
        sketch.perpendicular(construction_h, construction_v),
        sketch.parallel(construction_h, bottom),
        sketch.parallel(construction_h, top),
        sketch.parallel(construction_v, left),
        sketch.parallel(construction_v, right),
    ])
    for line in v_construction_lines:
        constraints.append(sketch.parallel(construction_v, line))
    for line in h_construction_lines:
        constraints.append(sketch.parallel(construction_h, line))

    return MappedPadChamferedRect(
        source=layer, center=center,
        tl=tl, tr=tr, br=br, bl=bl,
        top=top, right=right, bottom=bottom, left=left,
        chamfer_tl=chamfer_tl, chamfer_tr=chamfer_tr,
        chamfer_bl=chamfer_bl, chamfer_br=chamfer_br,
        top_mid=tm, left_mid=lm, right_mid=rm, bottom_mid=bm,
        construction_v=construction_v, construction_h=construction_h,
        constraints=constraints,
    )


def _map_pad_layer(
    sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
) -> MappedPadLayer | None:
    """Map a single PadStackLayer into solver entities."""
    shape = layer.shape

    if shape == PadStackShape.PSS_CIRCLE or (
        shape == PadStackShape.PSS_CUSTOM
        and layer.custom_anchor_shape == PadStackShape.PSS_CIRCLE
    ):
        return _map_pad_circle(sketch, layer, center, x, y)

    if shape in (PadStackShape.PSS_RECTANGLE, PadStackShape.PSS_ROUNDRECT) or (
        shape == PadStackShape.PSS_CUSTOM
        and layer.custom_anchor_shape == PadStackShape.PSS_RECTANGLE
    ):
        return _map_pad_rectangle(sketch, layer, center, x, y)

    if shape == PadStackShape.PSS_TRAPEZOID:
        return _map_pad_trapezoid(sketch, layer, center, x, y)

    if shape == PadStackShape.PSS_CHAMFEREDRECT:
        return _map_pad_chamfered_rect(sketch, layer, center, x, y)

    # PSS_UNKNOWN, PSS_OVAL, etc. — not supported
    return None


def map_pad(sketch: Sketch, shape: Pad) -> MappedPad:
    """Map a KiCad Pad into solver entities within *sketch*."""
    center = sketch.point(_to_mm(shape.position.x), _to_mm(shape.position.y))
    layers: list[MappedPadLayer] = []
    for layer in shape.padstack.copper_layers:
        result = _map_pad_layer(
            sketch, layer, center, shape.position.x, shape.position.y,
        )
        if result is not None:
            layers.append(result)
    return MappedPad(
        source=shape,
        pad_stack=shape.padstack,
        position=center,
        mapped_geometry=layers,
        constraints=[c for lyr in layers for c in lyr.constraints],
    )


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
