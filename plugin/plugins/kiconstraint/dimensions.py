from __future__ import annotations

import math
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
from .solver.entities import Line, Point


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
