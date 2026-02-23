"""Test action for KiConstraint — verifies the plugin can connect to KiCAD."""

from kipy import KiCad
from kiconstrint.solver.sketch import (
    Sketch
)
from kiconstraint import (
    map_shape
)


def main():
    kicad = KiCad()
    print(f"Connected to KiCad {kicad.get_version()}")

    board = kicad.get_board()
    print(f"Board: {board.name}")

    graphics = board.get_shapes()
    mapped = []
    sketch = Sketch()
    for graphic in graphics:
        mapped.append(map_shape(sketch, graphic))

    tracks = board.get_tracks()
    print(f"Tracks: {len(tracks)}")

    nets = board.get_nets()
    print(f"Nets: {len(nets)}")

    footprints = board.get_footprints()
    print(f"Footprints: {len(footprints)}")


if __name__ == "__main__":
    main()
