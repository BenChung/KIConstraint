from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from kipy.board_types import (
    AlignedDimension,
    CenterDimension,
    Dimension,
    LeaderDimension,
    OrthogonalDimension,
)

from .mapping import MappedGeometry, _to_mm, _v2
from .solver.constraints import Constraint
from .solver.entities import Line, Point
from .solver.sketch import Sketch


@dataclass(frozen=True)
class MappedEdgeDimension:
    source: Dimension
    name: str
    line: Line
    constraints: str

    def map_back(self):
        self.source.start = _v2(self.line.p1)
        self.source.end = _v2(self.line.p2)
        return self.source


@dataclass(frozen=True)
class MappedPointDimension:
    source: Dimension
    name: str
    point: Point
    constraints: str

    def map_back(self):
        original_offset = self.source.end - self.source.start
        original_text_offset = self.source.text.position - self.source.start
        self.source.start = _v2(self.point)
        self.source.end = self.source.start + original_offset
        self.source.text.position = self.source.start + original_text_offset
        return self.source


@dataclass
class DimensionMapping:
    edges: dict[str, MappedEdgeDimension] = field(default_factory=dict)
    points: dict[str, MappedPointDimension] = field(default_factory=dict)

    def map_back(self):
        modified = []
        for edge_dim in self.edges.values():
            modified.append(edge_dim.map_back())
        for point_dim in self.points.values():
            modified.append(point_dim.map_back())
        return modified


def _find_point(
    u: float, v: float, all_points: list[Point], tolerance: float,
) -> Point | None:
    for pt in all_points:
        if math.hypot(pt.u - u, pt.v - v) <= tolerance:
            return pt
    return None


def _extract_name(dim: Dimension) -> str | None:
    if isinstance(dim, LeaderDimension):
        return dim.override_text.split(",")[0]
    else:
        prefix = dim.prefix
    if prefix.endswith(":"):
        return prefix[:-1].strip()
    return None


def map_dimensions(
    sketch: Sketch,
    dimensions: Sequence[Dimension],
    mapped: Sequence[MappedGeometry],
    tolerance: float = 1e-4,
) -> DimensionMapping:
    """Map named KiCAD dimensions to sketch Points and Lines.

    Dimensions opt in to naming via a prefix of the form ``name:``.
    The association is discovered by matching dimension reference
    positions against sketch point positions within *tolerance* (mm).
    """
    # Collect all points from mapped geometry.
    all_points: list[Point] = []
    for shape in mapped:
        all_points.extend(shape.points)

    # Collect all lines and build an edge index keyed by endpoint handles.
    edge_index: dict[frozenset[int], Line] = {}
    for shape in mapped:
        for line in shape.lines:
            key = frozenset({line.p1.handle, line.p2.handle})
            edge_index[key] = line

    result = DimensionMapping()

    for dim in dimensions:
        if isinstance(dim, CenterDimension):
            # add the point corresponding to the center
            pt = sketch.point(_to_mm(dim.center.x), _to_mm(dim.center.y))
            # lock its position
            sketch.dragged(pt)
            all_points.append(pt)

    for dim in dimensions:
        name = _extract_name(dim)
        if name is None:
            continue

        if isinstance(dim, (AlignedDimension, OrthogonalDimension)):
            start_mm = (_to_mm(dim.start.x), _to_mm(dim.start.y))
            end_mm = (_to_mm(dim.end.x), _to_mm(dim.end.y))
            p_start = _find_point(*start_mm, all_points, tolerance)
            p_end = _find_point(*end_mm, all_points, tolerance)

            if p_start is not None and p_end is not None:
                key = frozenset({p_start.handle, p_end.handle})
                line = edge_index.get(key)
                if line is not None:
                    result.edges[name] = MappedEdgeDimension(
                        source=dim, name=name, line=line, constraints=dim.suffix
                    )
                    continue
                # No connecting line — fall through to individual points.
                result.points[name + ":start"] = MappedPointDimension(
                    source=dim, name=name + ":start", point=p_start, constraints=""
                )
                result.points[name + ":end"] = MappedPointDimension(
                    source=dim, name=name + ":end", point=p_end, constraints=""
                )
            elif p_start is not None:
                result.points[name + ":start"] = MappedPointDimension(
                    source=dim, name=name + ":start", point=p_start, constraints=""
                )
            elif p_end is not None:
                result.points[name + ":end"] = MappedPointDimension(
                    source=dim, name=name + ":end", point=p_end, constraints=""
                )

        elif isinstance(dim, LeaderDimension):
            start_mm = (_to_mm(dim.start.x), _to_mm(dim.start.y))
            pt = _find_point(*start_mm, all_points, tolerance)
            if pt is not None:
                split_text = dim.override_text.split(",", 1)
                result.points[name] = MappedPointDimension(
                    source=dim, name=name, point=pt, constraints=split_text[1] if len(split_text) > 1 else ""
                )

        # RadialDimension is skipped — references circles/arcs.

    return result


# ---------------------------------------------------------------------------
# Suffix constraint language — value objects
# ---------------------------------------------------------------------------


class ConstraintSpec:
    """Base class for parsed constraint specifications."""
    def apply_to_point(
        self: ConstraintSpec,
        sketch: Sketch, 
        entry: MappedPointDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"point {entry.name!r}"
        raise ValueError(f"{ctx}: constraint {type(self).__name__} requires an edge, not a point")

    def apply_to_edge(
        self: ConstraintSpec,
        sketch: Sketch, 
        entry: MappedEdgeDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"edge {entry.name!r}"
        raise ValueError(f"{ctx}: constraint {type(self).__name__} requires a point, not an edge")

@dataclass(frozen=True)
class Distance(ConstraintSpec):
    value_mm: float
    def apply_to_edge(
        self: Distance,
        sketch: Sketch, 
        entry: MappedEdgeDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        return sketch.distance(entry.line.p1, entry.line.p2, self.value_mm)


@dataclass(frozen=True)
class Parallel(ConstraintSpec):
    other: str
    def apply_to_edge(
        self: Parallel,
        sketch: Sketch, 
        entry: MappedEdgeDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"edge {entry.name!r}"
        return sketch.parallel(entry.line, _resolve_edge(self.other, dim_map, ctx))


@dataclass(frozen=True)
class Perpendicular(ConstraintSpec):
    other: str
    def apply_to_edge(
        self: Perpendicular,
        sketch: Sketch, 
        entry: MappedEdgeDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"edge {entry.name!r}"
        return sketch.perpendicular(entry.line, _resolve_edge(self.other, dim_map, ctx))


@dataclass(frozen=True)
class Coincident(ConstraintSpec):
    other: str
    def apply_to_point(
        self: Coincident,
        sketch: Sketch, 
        entry: MappedPointDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"point {entry.name!r}"
        return sketch.coincident(entry.point, _resolve_point(self.other, dim_map, ctx))


@dataclass(frozen=True)
class Vertical(ConstraintSpec):
    def apply_to_edge(
        self: Vertical,
        sketch: Sketch, 
        entry: MappedEdgeDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"edge {entry.name!r}"
        return sketch.vertical(entry.line)


@dataclass(frozen=True)
class Horizontal(ConstraintSpec):
    def apply_to_edge(
        self: Horizontal,
        sketch: Sketch, 
        entry: MappedEdgeDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        return sketch.horizontal(entry.line)

@dataclass(frozen=True)
class Equal(ConstraintSpec):
    other: str
    def apply_to_edge(
        self: Equal,
        sketch: Sketch, 
        entry: MappedEdgeDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"edge {entry.name!r}"
        return sketch.equal(entry.line, _resolve_edge(self.other, dim_map, ctx))


@dataclass(frozen=True)
class Midpoint(ConstraintSpec):
    other: str
    def apply_to_point(
        self:Midpoint,
        sketch: Sketch, 
        entry: MappedPointDimension,
        dim_map: DimensionMapping
    ) -> Constraint:
        ctx = f"point {entry.name!r}"
        return sketch.midpoint(entry.point, _resolve_edge(self.other, dim_map, ctx))


# ---------------------------------------------------------------------------
# Suffix parsing
# ---------------------------------------------------------------------------

_DISTANCE_RE = re.compile(r"^=(\d+(?:\.\d+)?)mm$")
_FUNC_RE = re.compile(r"^(\w+)\(([^)]+)\)$")

_CONSTRUCTORS: dict[str, type[ConstraintSpec]] = {
    "p": Parallel, "par": Parallel,
    "x": Perpendicular, "perp": Perpendicular,
    "c": Coincident, "coin": Coincident,
    "e": Equal, "eq": Equal,
    "m": Midpoint, "mid": Midpoint,
}

_BARE: dict[str, ConstraintSpec] = {
    "v": Vertical(), "vert": Vertical(),
    "h": Horizontal(), "horiz": Horizontal(),
}


def _parse_token(token: str) -> ConstraintSpec:
    """Parse a single constraint token into a :class:`ConstraintSpec`."""
    token = token.strip()
    if not token:
        raise ValueError("empty constraint token")

    m = _DISTANCE_RE.match(token)
    if m:
        return Distance(float(m.group(1)))

    m = _FUNC_RE.match(token)
    if m:
        cls = _CONSTRUCTORS.get(m.group(1))
        if cls is None:
            raise ValueError(f"unknown constraint: {m.group(1)!r}")
        return cls(m.group(2).strip())

    bare = _BARE.get(token)
    if bare is not None:
        return bare

    raise ValueError(f"unrecognized constraint token: {token!r}")


def parse_suffix(suffix: str) -> list[ConstraintSpec]:
    """Parse a comma-separated suffix into constraint specs."""
    if not suffix.strip():
        return []
    return [_parse_token(t) for t in suffix.split(",")]


# ---------------------------------------------------------------------------
# Constraint application
# ---------------------------------------------------------------------------


def _resolve_edge(name: str, dim_map: DimensionMapping, context: str) -> Line:
    entry = dim_map.edges.get(name)
    if entry is None:
        raise ValueError(f"{context}: edge {name!r} not found in dimension mapping")
    return entry.line


def _resolve_point(name: str, dim_map: DimensionMapping, context: str) -> Point:
    entry = dim_map.points.get(name)
    if entry is None:
        raise ValueError(f"{context}: point {name!r} not found in dimension mapping")
    return entry.point


def apply_dimension_constraints(
    sketch: Sketch,
    dim_map: DimensionMapping,
) -> list[Constraint]:
    """Parse suffix constraints from mapped dimensions and apply them.

    Returns the list of all :class:`Constraint` objects created.
    """
    constraints: list[Constraint] = []

    for entry in dim_map.edges.values():
        for spec in parse_suffix(entry.constraints):
            constraints.append(spec.apply_to_edge(sketch, entry, dim_map))

    for entry in dim_map.points.values():
        for spec in parse_suffix(entry.constraints):
            constraints.append(spec.apply_to_point(sketch, entry, dim_map))

    return constraints
