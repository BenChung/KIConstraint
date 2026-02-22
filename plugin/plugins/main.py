"""KiConstraint plugin entrypoint for KiCAD IPC API."""

from kipy import KiCad


def main():
    kicad = KiCad()
    board = kicad.get_board()
    # TODO: implement plugin logic
    print(f"KiConstraint loaded, board: {board.name}")


if __name__ == "__main__":
    main()
