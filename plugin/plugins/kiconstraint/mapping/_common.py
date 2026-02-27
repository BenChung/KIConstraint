from __future__ import annotations

from abc import ABC, abstractmethod

from kipy.geometry import Vector2

from ..solver.constraints import Constraint
from ..solver.entities import Line, Point

_NM_PER_MM = 1_000_000


def _to_mm(nm: int) -> float:
    return nm / _NM_PER_MM


def _v2(point: Point) -> Vector2:
    """Convert a solved Point to a kipy Vector2."""
    return Vector2.from_xy_mm(point.u, point.v)


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
