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

from .mapping import MappedGeometry, _to_mm
from .solver.constraints import Constraint
from .solver.entities import Line, Point
from .solver.sketch import Sketch


@dataclass(frozen=True)
class MappedEdgeDimension:
    source: Dimension
    name: str
    line: Line


@dataclass(frozen=True)
class MappedPointDimension:
    source: Dimension
    name: str
    point: Point


@dataclass
class DimensionMapping:
    edges: dict[str, MappedEdgeDimension] = field(default_factory=dict)
    points: dict[str, MappedPointDimension] = field(default_factory=dict)


def _find_point(
    u: float, v: float, all_points: list[Point], tolerance: float,
) -> Point | None:
    for pt in all_points:
        if math.hypot(pt.u - u, pt.v - v) <= tolerance:
            return pt
    return None


def _extract_name(dim: Dimension) -> str | None:
    prefix = dim.prefix
    if prefix.endswith(":"):
        return prefix[:-1].strip()
    return None


def map_dimensions(
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
                # No connecting line — fall through to individual points.
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

        elif isinstance(dim, CenterDimension):
            center_mm = (_to_mm(dim.center.x), _to_mm(dim.center.y))
            pt = _find_point(*center_mm, all_points, tolerance)
            if pt is not None:
                result.points[name] = MappedPointDimension(
                    source=dim, name=name, point=pt,
                )

        # RadialDimension is skipped — references circles/arcs.

    return result


# ---------------------------------------------------------------------------
# Suffix constraint language
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    "par": "p",
    "perp": "x",
    "coin": "c",
    "vert": "v",
    "horiz": "h",
    "eq": "e",
    "mid": "m",
}

_DISTANCE_RE = re.compile(r"^=(\d+(?:\.\d+)?)mm$")
_FUNC_RE = re.compile(r"^(\w+)\(([^)]+)\)$")
_BARE_RE = re.compile(r"^(\w+)$")

# Constraints that require an edge (Line).
_EDGE_CONSTRAINTS = {"p", "x", "v", "h", "e", "="}


def _parse_token(token: str) -> tuple[str, str | float | None]:
    """Parse a single constraint token into ``(kind, arg)``.

    Returns one of:
    - ``("=", 3.0)``
    - ``("p", "other_name")``
    - ``("v", None)``
    """
    token = token.strip()
    if not token:
        raise ValueError("empty constraint token")

    m = _DISTANCE_RE.match(token)
    if m:
        return ("=", float(m.group(1)))

    m = _FUNC_RE.match(token)
    if m:
        name = _ALIASES.get(m.group(1), m.group(1))
        return (name, m.group(2).strip())

    m = _BARE_RE.match(token)
    if m:
        name = _ALIASES.get(m.group(1), m.group(1))
        return (name, None)

    raise ValueError(f"unrecognized constraint token: {token!r}")


def _parse_suffix(suffix: str) -> list[tuple[str, str | float | None]]:
    """Parse a comma-separated suffix into a list of constraint specs."""
    if not suffix.strip():
        return []
    return [_parse_token(t) for t in suffix.split(",")]


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

    # Process edge dimensions.
    for name, entry in dim_map.edges.items():
        suffix = entry.source.suffix
        for kind, arg in _parse_suffix(suffix):
            ctx = f"edge {name!r}"
            if kind == "=":
                assert isinstance(arg, float)
                constraints.append(
                    sketch.distance(entry.line.p1, entry.line.p2, arg)
                )
            elif kind == "p":
                other = _resolve_edge(arg, dim_map, ctx)
                constraints.append(sketch.parallel(entry.line, other))
            elif kind == "x":
                other = _resolve_edge(arg, dim_map, ctx)
                constraints.append(sketch.perpendicular(entry.line, other))
            elif kind == "v":
                constraints.append(sketch.vertical(entry.line))
            elif kind == "h":
                constraints.append(sketch.horizontal(entry.line))
            elif kind == "e":
                other = _resolve_edge(arg, dim_map, ctx)
                constraints.append(sketch.equal(entry.line, other))
            elif kind == "m":
                # mid(point_name) → named point is midpoint of this edge.
                pt = _resolve_point(arg, dim_map, ctx)
                constraints.append(sketch.midpoint(pt, entry.line))
            elif kind == "c":
                raise ValueError(
                    f"{ctx}: 'coin' is not applicable to edges"
                )
            else:
                raise ValueError(f"{ctx}: unknown constraint {kind!r}")

    # Process point dimensions.
    for name, entry in dim_map.points.items():
        suffix = entry.source.suffix
        for kind, arg in _parse_suffix(suffix):
            ctx = f"point {name!r}"
            if kind == "c":
                pt = _resolve_point(arg, dim_map, ctx)
                constraints.append(sketch.coincident(entry.point, pt))
            elif kind == "m":
                # mid(edge_name) → this point is midpoint of named edge.
                edge = _resolve_edge(arg, dim_map, ctx)
                constraints.append(sketch.midpoint(entry.point, edge))
            elif kind in _EDGE_CONSTRAINTS:
                raise ValueError(
                    f"{ctx}: constraint {kind!r} requires an edge, not a point"
                )
            else:
                raise ValueError(f"{ctx}: unknown constraint {kind!r}")

    return constraints
