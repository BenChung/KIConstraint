"""Test action for KiConstraint — verifies the plugin can connect to KiCAD."""

from kipy import KiCad
from kiconstraint.solver import Sketch
from kiconstraint.dimensions import map_dimensions
from kiconstraint.mapping import map_shape


def main():
    kicad = KiCad()
    print(f"Connected to KiCad {kicad.get_version()}")

    board = kicad.get_board()
    print(f"Board: {board.name}")

    sketch = Sketch()

    graphics = board.get_shapes()
    mapped = []
    for graphic in graphics:
        mapped.append(map_shape(sketch, graphic))
    print(f"Mapped {len(mapped)} board shapes")

    dimensions = board.get_dimensions()
    dim_map = map_dimensions(dimensions, mapped)
    print(f"Named {len(dim_map.edges)} edges, {len(dim_map.points)} points")

    result = sketch.solve()
    print(f"Solve: ok={result.ok}, dof={result.dof}")


if __name__ == "__main__":
    main()
