"""Test action for KiConstraint — verifies the plugin can connect to KiCAD."""

from kipy import KiCad


def main():
    kicad = KiCad()
    print(f"Connected to KiCad {kicad.get_version()}")

    board = kicad.get_board()
    print(f"Board: {board.name}")

    tracks = board.get_tracks()
    print(f"Tracks: {len(tracks)}")

    nets = board.get_nets()
    print(f"Nets: {len(nets)}")

    footprints = board.get_footprints()
    print(f"Footprints: {len(footprints)}")


if __name__ == "__main__":
    main()
