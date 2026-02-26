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


# ---------------------------------------------------------------------------
# Mapping dataclasses (pass 1 output — name registry + map-back)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappedEdgeDimension:
    source: Dimension
    name: str
    line: Line

    def map_back(self):
        self.source.start = _v2(self.line.p1)
        self.source.end = _v2(self.line.p2)
        return self.source


@dataclass(frozen=True)
class MappedPointDimension:
    source: Dimension
    name: str
    point: Point

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
    extra_points: list[Point] = field(default_factory=list)

    def map_back(self):
        modified = []
        for edge_dim in self.edges.values():
            modified.append(edge_dim.map_back())
        for point_dim in self.points.values():
            modified.append(point_dim.map_back())
        return modified


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _get_suffix(dim: Dimension) -> str:
    """Extract constraint text from a dimension."""
    if isinstance(dim, LeaderDimension):
        parts = dim.override_text.split(",", 1)
        return parts[1] if len(parts) > 1 else ""
    return getattr(dim, "suffix", "")


def _build_lookup(
    mapped: Sequence[MappedGeometry],
    extra_points: Sequence[Point] = (),
) -> tuple[list[Point], dict[frozenset[int], Line]]:
    """Build point list and edge index from mapped geometry."""
    all_points: list[Point] = list(extra_points)
    for shape in mapped:
        all_points.extend(shape.points)

    edge_index: dict[frozenset[int], Line] = {}
    for shape in mapped:
        for line in shape.lines:
            key = frozenset({line.p1.handle, line.p2.handle})
            edge_index[key] = line

    return all_points, edge_index


# ---------------------------------------------------------------------------
# Pass 1: map_dimensions — build name registry
# ---------------------------------------------------------------------------


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

    This is pass 1: it only builds the name registry (for cross-reference
    resolution and map-back).  Constraint text is NOT stored here — it is
    read directly from dimensions in pass 2.
    """
    all_points, edge_index = _build_lookup(mapped)
    result = DimensionMapping()

    # First pass: add CenterDimension reference points.
    for dim in dimensions:
        if isinstance(dim, CenterDimension):
            pt = sketch.point(_to_mm(dim.center.x), _to_mm(dim.center.y))
            sketch.dragged(pt)
            all_points.append(pt)
            result.extra_points.append(pt)

    # Second pass: associate named dimensions with entities.
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
                        source=dim, name=name, line=line,
                    )
                    continue
                # No connecting line — register individual points.
                result.points[name + ":start"] = MappedPointDimension(
                    source=dim, name=name + ":start", point=p_start,
                )
                result.points[name + ":end"] = MappedPointDimension(
                    source=dim, name=name + ":end", point=p_end,
                )
            elif p_start is not None:
                result.points[name + ":start"] = MappedPointDimension(
                    source=dim, name=name + ":start", point=p_start,
                )
            elif p_end is not None:
                result.points[name + ":end"] = MappedPointDimension(
                    source=dim, name=name + ":end", point=p_end,
                )

        elif isinstance(dim, LeaderDimension):
            start_mm = (_to_mm(dim.start.x), _to_mm(dim.start.y))
            pt = _find_point(*start_mm, all_points, tolerance)
            if pt is not None:
                result.points[name] = MappedPointDimension(
                    source=dim, name=name, point=pt,
                )

        # RadialDimension is skipped — references circles/arcs.

    return result


# ---------------------------------------------------------------------------
# Suffix constraint language — value objects
# ---------------------------------------------------------------------------


class ConstraintSpec:
    """Base class for parsed constraint specifications."""

    def apply_to_line(
        self, sketch: Sketch, line: Line, name: str, dim_map: DimensionMapping,
    ) -> Constraint:
        raise ValueError(
            f"{name!r}: constraint {type(self).__name__} requires a point, not a line"
        )

    def apply_to_two_points(
        self, sketch: Sketch, p1: Point, p2: Point,
        name: str, dim_map: DimensionMapping,
    ) -> Constraint:
        raise ValueError(
            f"{name!r}: constraint {type(self).__name__}"
            " cannot apply between two unconnected points"
        )

    def apply_to_point(
        self, sketch: Sketch, pt: Point, name: str, dim_map: DimensionMapping,
    ) -> Constraint:
        raise ValueError(
            f"{name!r}: constraint {type(self).__name__} requires a line, not a point"
        )


@dataclass(frozen=True)
class Distance(ConstraintSpec):
    value_mm: float

    def apply_to_line(self, sketch, line, name, dim_map):
        return sketch.distance(line.p1, line.p2, self.value_mm)

    def apply_to_two_points(self, sketch, p1, p2, name, dim_map):
        return sketch.distance(p1, p2, self.value_mm)


@dataclass(frozen=True)
class Parallel(ConstraintSpec):
    other: str

    def apply_to_line(self, sketch, line, name, dim_map):
        return sketch.parallel(line, _resolve_line(self.other, dim_map, name))


@dataclass(frozen=True)
class Perpendicular(ConstraintSpec):
    other: str

    def apply_to_line(self, sketch, line, name, dim_map):
        return sketch.perpendicular(line, _resolve_line(self.other, dim_map, name))


@dataclass(frozen=True)
class Coincident(ConstraintSpec):
    other: str

    def apply_to_point(self, sketch, pt, name, dim_map):
        return sketch.coincident(pt, _resolve_point(self.other, dim_map, name))


@dataclass(frozen=True)
class Vertical(ConstraintSpec):
    def apply_to_line(self, sketch, line, name, dim_map):
        return sketch.vertical(line)


@dataclass(frozen=True)
class Horizontal(ConstraintSpec):
    def apply_to_line(self, sketch, line, name, dim_map):
        return sketch.horizontal(line)


@dataclass(frozen=True)
class Equal(ConstraintSpec):
    other: str

    def apply_to_line(self, sketch, line, name, dim_map):
        return sketch.equal(line, _resolve_line(self.other, dim_map, name))


@dataclass(frozen=True)
class Midpoint(ConstraintSpec):
    other: str

    def apply_to_point(self, sketch, pt, name, dim_map):
        return sketch.midpoint(pt, _resolve_line(self.other, dim_map, name))


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
# Reference resolution
# ---------------------------------------------------------------------------


def _resolve_line(name: str, dim_map: DimensionMapping, context: str) -> Line:
    entry = dim_map.edges.get(name)
    if entry is None:
        raise ValueError(f"{context!r}: edge {name!r} not found in dimension mapping")
    return entry.line


def _resolve_point(name: str, dim_map: DimensionMapping, context: str) -> Point:
    entry = dim_map.points.get(name)
    if entry is None:
        raise ValueError(f"{context!r}: point {name!r} not found in dimension mapping")
    return entry.point


# ---------------------------------------------------------------------------
# Pass 2: apply_dimension_constraints
# ---------------------------------------------------------------------------


def apply_dimension_constraints(
    sketch: Sketch,
    dimensions: Sequence[Dimension],
    dim_map: DimensionMapping,
    mapped: Sequence[MappedGeometry],
    tolerance: float = 1e-4,
) -> list[Constraint]:
    """Parse and apply constraints from all dimensions.

    Unlike :func:`map_dimensions` (pass 1), this processes **all** dimensions
    regardless of whether they are named.  Constraint text comes from each
    dimension's suffix (or ``override_text`` for LeaderDimension).  The
    *dim_map* registry (from pass 1) is used only for resolving
    cross-references like ``par(other_edge)``.

    Returns the list of all :class:`Constraint` objects created.
    """
    all_points, edge_index = _build_lookup(mapped, dim_map.extra_points)
    constraints: list[Constraint] = []

    for dim in dimensions:
        suffix = _get_suffix(dim)
        if not suffix.strip():
            continue

        name = _extract_name(dim) or "<unnamed>"
        specs = parse_suffix(suffix)

        if isinstance(dim, (AlignedDimension, OrthogonalDimension)):
            start_mm = (_to_mm(dim.start.x), _to_mm(dim.start.y))
            end_mm = (_to_mm(dim.end.x), _to_mm(dim.end.y))
            p_start = _find_point(*start_mm, all_points, tolerance)
            p_end = _find_point(*end_mm, all_points, tolerance)

            if p_start is not None and p_end is not None:
                key = frozenset({p_start.handle, p_end.handle})
                line = edge_index.get(key)
                if line is not None:
                    for spec in specs:
                        constraints.append(
                            spec.apply_to_line(sketch, line, name, dim_map)
                        )
                else:
                    for spec in specs:
                        constraints.append(
                            spec.apply_to_two_points(
                                sketch, p_start, p_end, name, dim_map,
                            )
                        )
            elif p_start is not None:
                for spec in specs:
                    constraints.append(
                        spec.apply_to_point(sketch, p_start, name, dim_map)
                    )
            elif p_end is not None:
                for spec in specs:
                    constraints.append(
                        spec.apply_to_point(sketch, p_end, name, dim_map)
                    )

        elif isinstance(dim, LeaderDimension):
            start_mm = (_to_mm(dim.start.x), _to_mm(dim.start.y))
            pt = _find_point(*start_mm, all_points, tolerance)
            if pt is not None:
                for spec in specs:
                    constraints.append(
                        spec.apply_to_point(sketch, pt, name, dim_map)
                    )

    return constraints
