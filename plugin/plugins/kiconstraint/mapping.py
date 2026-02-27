from __future__ import annotations

import math
from abc import ABC, abstractmethod
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
from kipy.board_types import (
    Pad,
    PadStack,
    PadStackLayer,
    PadStackShape,
)

from kipy.geometry import Vector2

from .solver.constraints import Constraint, SolveResult
from .solver.entities import Arc, Circle, Cubic, Line, Point
from .solver.sketch import Sketch

_NM_PER_MM = 1_000_000


def _to_mm(nm: int) -> float:
    return nm / _NM_PER_MM


# ---------------------------------------------------------------------------
# Mapped geometry base class
# ---------------------------------------------------------------------------


class MappedGeometry(ABC):
    """Base class for all mapped geometry objects.

    Every subclass must provide:
    - ``points``: all solver :class:`Point` entities in this geometry.
    - ``constraints``: solver constraints that maintain the shape
      (dataclass field, may be empty).
    """

    @property
    @abstractmethod
    def points(self) -> list[Point]: ...

    @property
    @abstractmethod
    def lines(self) -> list[Line]: ...

    # ``constraints`` is a dataclass field on every concrete subclass
    # rather than an abstract property, because @property descriptors
    # on a base class shadow dataclass fields of the same name.
    constraints: list[Constraint]


# ---------------------------------------------------------------------------
# Mapped shape dataclasses
# ---------------------------------------------------------------------------


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

@dataclass(frozen=True)
class ChamferCorner:
    """Geometry for a single chamfered corner."""
    p_h: Point
    p_v: Point
    chamfer: Line
    h_construction: Line
    v_construction: Line


@dataclass(frozen=True)
class MappedPadCircle(MappedGeometry):
    source: PadStackLayer
    center: Point
    circle: Circle
    constraints: list[Constraint]

    def write_back(self) -> None:
        d = self.circle.radius.value * 2
        self.source.size = Vector2.from_xy_mm(d, d)

    @property
    def points(self) -> list[Point]:
        return [self.center]

    @property
    def lines(self) -> list[Line]:
        return []


@dataclass(frozen=True)
class MappedPadRectangle(MappedGeometry):
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

    def write_back(self) -> None:
        width = math.hypot(self.tr.u - self.tl.u, self.tr.v - self.tl.v)
        height = math.hypot(self.bl.u - self.tl.u, self.bl.v - self.tl.v)
        self.source.size = Vector2.from_xy_mm(width, height)

    @property
    def points(self) -> list[Point]:
        return [self.center, self.tl, self.tr, self.br, self.bl]

    @property
    def lines(self) -> list[Line]:
        return [self.top, self.right, self.bottom, self.left]


@dataclass(frozen=True)
class MappedPadTrapezoid(MappedGeometry):
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

    def write_back(self) -> None:
        construction_len = math.hypot(
            self.midpoint_b.u - self.midpoint_a.u,
            self.midpoint_b.v - self.midpoint_a.v,
        )
        left_len = math.hypot(
            self.bl.u - self.tl.u, self.bl.v - self.tl.v,
        )
        right_len = math.hypot(
            self.br.u - self.tr.u, self.br.v - self.tr.v,
        )
        top_len = math.hypot(
            self.tr.u - self.tl.u, self.tr.v - self.tl.v,
        )
        bottom_len = math.hypot(
            self.br.u - self.bl.u, self.br.v - self.bl.v,
        )
        original_delta = self.source.trapezoid_delta
        if original_delta.x != 0 or original_delta.y == 0:
            # Vertical skew (or no skew):
            # construction connects midpoints of left/right edges.
            self.source.size = Vector2.from_xy_mm(
                construction_len, (left_len + right_len) / 2,
            )
            self.source.trapezoid_delta = Vector2.from_xy_mm(
                (right_len - left_len) / 2, 0,
            )
        else:
            # Horizontal skew:
            # construction connects midpoints of top/bottom edges.
            self.source.size = Vector2.from_xy_mm(
                (top_len + bottom_len) / 2, construction_len,
            )
            self.source.trapezoid_delta = Vector2.from_xy_mm(
                0, (top_len - bottom_len) / 2,
            )

    @property
    def points(self) -> list[Point]:
        return [self.center, self.tl, self.tr, self.br, self.bl,
                self.midpoint_a, self.midpoint_b]

    @property
    def lines(self) -> list[Line]:
        return [self.top, self.right, self.bottom, self.left]


@dataclass(frozen=True)
class MappedPadChamferedRect(MappedGeometry):
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

    def write_back(self) -> None:
        width = math.hypot(
            self.right_mid.u - self.left_mid.u,
            self.right_mid.v - self.left_mid.v,
        )
        height = math.hypot(
            self.bottom_mid.u - self.top_mid.u,
            self.bottom_mid.v - self.top_mid.v,
        )
        self.source.size = Vector2.from_xy_mm(width, height)
        # Derive chamfer_ratio from any existing chamfer's construction length.
        for chamfer in (self.chamfer_tl, self.chamfer_tr,
                        self.chamfer_bl, self.chamfer_br):
            if chamfer is not None:
                chamfer_dist = math.hypot(
                    chamfer.h_construction.p2.u - chamfer.h_construction.p1.u,
                    chamfer.h_construction.p2.v - chamfer.h_construction.p1.v,
                )
                self.source.chamfer_ratio = chamfer_dist / min(width, height)
                break

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

    @property
    def lines(self) -> list[Line]:
        result = [self.top, self.right, self.bottom, self.left]
        for chamfer in (self.chamfer_tl, self.chamfer_tr,
                        self.chamfer_bl, self.chamfer_br):
            if chamfer is not None:
                result.append(chamfer.chamfer)
        return result


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

    def write_back(self) -> None:
        self.source.position = _v2(self.position)
        for layer in self.mapped_geometry:
            layer.write_back()

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

    # Build each corner's chamfer.  Each entry is
    # (base_point, is_chamfered, h_sign, v_sign).
    corners = layer.chamfered_corners
    corner_specs = [
        (tl, corners.top_left,     1,  1),
        (tr, corners.top_right,   -1,  1),
        (bl, corners.bottom_left,  1, -1),
        (br, corners.bottom_right,-1, -1),
    ]
    chamfers: list[ChamferCorner | None] = []
    edge_h: list[Point] = []   # point along horizontal edge per corner
    edge_v: list[Point] = []   # point along vertical edge per corner

    for pt, is_chamfered, h_sign, v_sign in corner_specs:
        if is_chamfered:
            c = _build_chamfer_corner(
                sketch, pt, h_sign * chamfer_dist, v_sign * chamfer_dist,
            )
            v_construction_lines.append(c.v_construction)
            h_construction_lines.append(c.h_construction)
            constraints.append(sketch.equal(c.v_construction, c.h_construction))
            chamfers.append(c)
            edge_h.append(c.p_h)
            edge_v.append(c.p_v)
        else:
            chamfers.append(None)
            edge_h.append(pt)
            edge_v.append(pt)

    chamfer_tl, chamfer_tr, chamfer_bl, chamfer_br = chamfers

    # Make all chamfers the same length
    for i in range(1, len(v_construction_lines)):
        constraints.append(sketch.equal(v_construction_lines[i],
                                        v_construction_lines[i - 1]))

    # Edge lines connecting chamfer endpoints
    # indices: tl=0, tr=1, bl=2, br=3
    top = sketch.line(edge_h[1], edge_h[0])
    left = sketch.line(edge_v[0], edge_v[2])
    right = sketch.line(edge_v[1], edge_v[3])
    bottom = sketch.line(edge_h[2], edge_h[3])

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

_SHAPE_MAP: dict[type, type] = {
    KiSegment: MappedSegment,
    KiArc: MappedArc,
    KiCircle: MappedCircle,
    KiRectangle: MappedRectangle,
    KiBezier: MappedBezier,
}


# ---------------------------------------------------------------------------
# Mapping / write-back entry points
# ---------------------------------------------------------------------------


def _v2(point: Point) -> Vector2:
    """Convert a solved Point to a kipy Vector2."""
    return Vector2.from_xy_mm(point.u, point.v)


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
