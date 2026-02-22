from __future__ import annotations

import slvs


class Entity:
    """Base wrapper around a raw Slvs_Entity dict."""

    __slots__ = ("_raw",)

    def __init__(self, raw: slvs.Slvs_Entity) -> None:
        self._raw = raw

    @property
    def handle(self) -> int:
        return self._raw["h"]

    @property
    def group(self) -> int:
        return self._raw["group"]

    def _param(self, index: int) -> float:
        return slvs.get_param_value(self._raw["param"][index])


class Point(Entity):
    """2D or 3D point."""

    @property
    def u(self) -> float:
        return self._param(0)

    @property
    def v(self) -> float:
        return self._param(1)

    # 3D aliases
    @property
    def x(self) -> float:
        return self._param(0)

    @property
    def y(self) -> float:
        return self._param(1)

    @property
    def z(self) -> float:
        return self._param(2)


class Normal(Entity):
    """Orientation / normal (internal use for workplanes, circles, arcs)."""


class Distance(Entity):
    """Scalar distance value."""

    @property
    def value(self) -> float:
        return self._param(0)


class Line(Entity):
    """Line segment defined by two points."""

    __slots__ = ("_raw", "p1", "p2")

    def __init__(self, raw: slvs.Slvs_Entity, p1: Point, p2: Point) -> None:
        super().__init__(raw)
        self.p1 = p1
        self.p2 = p2


class Circle(Entity):
    """Circle defined by a center point and a radius distance."""

    __slots__ = ("_raw", "center", "radius")

    def __init__(self, raw: slvs.Slvs_Entity, center: Point, radius: Distance) -> None:
        super().__init__(raw)
        self.center = center
        self.radius = radius


class Arc(Entity):
    """Arc defined by center, start, and end points."""

    __slots__ = ("_raw", "center", "start", "end")

    def __init__(self, raw: slvs.Slvs_Entity, center: Point, start: Point, end: Point) -> None:
        super().__init__(raw)
        self.center = center
        self.start = start
        self.end = end


class Cubic(Entity):
    """Cubic Bezier curve defined by four control points."""

    __slots__ = ("_raw", "p1", "p2", "p3", "p4")

    def __init__(self, raw: slvs.Slvs_Entity, p1: Point, p2: Point, p3: Point, p4: Point) -> None:
        super().__init__(raw)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4


class Workplane(Entity):
    """Custom workplane entity."""
