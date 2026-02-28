from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from kipy.board_types import (
    Pad,
    PadStack,
    PadStackLayer,
    PadStackShape,
)
from kipy.geometry import Vector2

from ..solver.constraints import Constraint
from ..solver.entities import Circle, Line, Point
from ..solver.sketch import Sketch
from ._common import MappedGeometry, _to_mm, _v2


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

    @classmethod
    def create(
        cls, sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
    ) -> MappedPadCircle:
        circle = sketch.circle(center, _to_mm(layer.size.x / 2))
        return cls(source=layer, center=center, circle=circle, constraints=[])

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

    @classmethod
    def create(
        cls, sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
    ) -> MappedPadRectangle:
        half_size = layer.size * 0.5
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
        return cls(
            source=layer, center=center,
            tl=tl, tr=tr, br=br, bl=bl,
            top=top, right=right, bottom=bottom, left=left,
            construction=construction, constraints=constraints,
        )

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

    @classmethod
    def create(
        cls, sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
    ) -> MappedPadTrapezoid:
        trap_delta = layer.trapezoid_delta * 0.5
        half_size = layer.size * 0.5
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
        return cls(
            source=layer, center=center,
            tl=tl, tr=tr, br=br, bl=bl,
            top=top, right=right, bottom=bottom, left=left,
            midpoint_a=mp_a, midpoint_b=mp_b,
            construction=construction, constraints=constraints,
        )

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

    @staticmethod
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

    @classmethod
    def create(
        cls, sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
    ) -> MappedPadChamferedRect:
        chamfer_dist = min(layer.size.x, layer.size.y) * layer.chamfer_ratio
        half_size = layer.size * 0.5

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
                c = cls._build_chamfer_corner(
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

        return cls(
            source=layer, center=center,
            tl=tl, tr=tr, br=br, bl=bl,
            top=top, right=right, bottom=bottom, left=left,
            chamfer_tl=chamfer_tl, chamfer_tr=chamfer_tr,
            chamfer_bl=chamfer_bl, chamfer_br=chamfer_br,
            top_mid=tm, left_mid=lm, right_mid=rm, bottom_mid=bm,
            construction_v=construction_v, construction_h=construction_h,
            constraints=constraints,
        )

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


def _map_pad_layer(
    sketch: Sketch, layer: PadStackLayer, center: Point, x: int, y: int,
) -> MappedPadLayer | None:
    """Map a single PadStackLayer into solver entities."""
    shape = layer.shape

    if shape == PadStackShape.PSS_CIRCLE or (
        shape == PadStackShape.PSS_CUSTOM
        and layer.custom_anchor_shape == PadStackShape.PSS_CIRCLE
    ):
        return MappedPadCircle.create(sketch, layer, center, x, y)

    if shape in (PadStackShape.PSS_RECTANGLE, PadStackShape.PSS_ROUNDRECT) or (
        shape == PadStackShape.PSS_CUSTOM
        and layer.custom_anchor_shape == PadStackShape.PSS_RECTANGLE
    ):
        return MappedPadRectangle.create(sketch, layer, center, x, y)

    if shape == PadStackShape.PSS_TRAPEZOID:
        return MappedPadTrapezoid.create(sketch, layer, center, x, y)

    if shape == PadStackShape.PSS_CHAMFEREDRECT:
        return MappedPadChamferedRect.create(sketch, layer, center, x, y)

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
