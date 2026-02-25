"""Test action for KiConstraint — verifies the plugin can connect to KiCAD."""

from kipy import KiCad
from kiconstraint.solver import Sketch
from kiconstraint.dimensions import apply_dimension_constraints, map_dimensions
from kiconstraint.mapping import map_shape, write_back_shapes
from kipy.proto.common.types import KiCadObjectType


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

    items = board.get_items(types=[KiCadObjectType.KOT_PCB_DIMENSION])
    print(f"Found {len(items)} items")
    
    dimensions = board.get_dimensions()
    print(f"Found {len(dimensions)} dimensions")
    dim_map = map_dimensions(sketch, dimensions, mapped)
    print(f"Named {len(dim_map.edges)} edges, {len(dim_map.points)} points")

    dim_constraints = apply_dimension_constraints(sketch, dim_map)
    print(f"Applied {len(dim_constraints)} dimension constraints")

    result = sketch.solve()
    print(f"Solve: ok={result.ok}, dof={result.dof}")

    if result.ok:
        modified = write_back_shapes(mapped, result)
        mod_dims = dim_map.map_back()
        board.update_items(modified + mod_dims)
        print(f"Wrote back {len(modified)} shapes")
    else:
        print("Solve failed — skipping write-back")


if __name__ == "__main__":
    main()
