"""Extract geometry and dimensions from the current KiCad board into a JSON
fixture file for offline testing.

Usage::

    python extract_board.py [output.json]

Connects to a running KiCad instance, reads the board's graphic shapes,
dimensions, and pads, then serialises the raw protobuf data to JSON.

The resulting file can be loaded in tests with :func:`load_fixture`.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from google.protobuf.json_format import MessageToDict
from kipy import KiCad
from kipy.board_types import (
    AlignedDimension,
    BoardShape,
    CenterDimension,
    Dimension,
    LeaderDimension,
    OrthogonalDimension,
    Pad,
    RadialDimension,
)
from kipy.common_types import (
    Arc,
    Bezier,
    Circle,
    GraphicShape,
    Rectangle,
    Segment,
)


_SHAPE_NAMES = {
    Segment: "segment",
    Arc: "arc",
    Circle: "circle",
    Rectangle: "rectangle",
    Bezier: "bezier",
}

_DIM_NAMES = {
    AlignedDimension: "aligned",
    OrthogonalDimension: "orthogonal",
    LeaderDimension: "leader",
    CenterDimension: "center",
    RadialDimension: "radial",
}


def _shape_proto(shape: BoardShape) -> dict[str, Any]:
    """Serialise a board shape's protobuf to a JSON-friendly dict."""
    return MessageToDict(shape._proto, preserving_proto_field_name=True)


def _dim_proto(dim: Dimension) -> dict[str, Any]:
    """Serialise a dimension's protobuf to a JSON-friendly dict."""
    return MessageToDict(dim._proto, preserving_proto_field_name=True)


def _pad_proto(pad: Pad) -> dict[str, Any]:
    """Serialise a pad's protobuf to a JSON-friendly dict."""
    return MessageToDict(pad._proto, preserving_proto_field_name=True)


def extract(output_path: str) -> None:
    kicad = KiCad()
    print(f"Connected to KiCad {kicad.get_version()}")

    board = kicad.get_board()
    print(f"Board: {board.name}")

    shapes = board.get_shapes()
    dimensions = board.get_dimensions()
    pads = board.get_pads()

    fixture: dict[str, Any] = {
        "board_name": board.name,
        "shapes": [],
        "dimensions": [],
        "pads": [],
    }

    for shape in shapes:
        kind = "unknown"
        for cls, name in _SHAPE_NAMES.items():
            if isinstance(shape, cls):
                kind = name
                break
        fixture["shapes"].append({
            "type": kind,
            "proto": _shape_proto(shape),
        })

    for dim in dimensions:
        kind = "unknown"
        for cls, name in _DIM_NAMES.items():
            if isinstance(dim, cls):
                kind = name
                break
        fixture["dimensions"].append({
            "type": kind,
            "proto": _dim_proto(dim),
        })

    for pad in pads:
        fixture["pads"].append({
            "proto": _pad_proto(pad),
        })

    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)

    print(
        f"Wrote {len(fixture['shapes'])} shapes, "
        f"{len(fixture['dimensions'])} dimensions, "
        f"{len(fixture['pads'])} pads to {output_path}"
    )


# ---------------------------------------------------------------------------
# Loader for tests
# ---------------------------------------------------------------------------

def load_fixture(path: str) -> dict[str, Any]:
    """Load a board fixture from a JSON file.

    Returns a dict with ``shapes``, ``dimensions``, and ``pads`` lists.
    Each item contains a ``type`` string and a reconstructed kipy wrapper
    object under the ``obj`` key.
    """
    from google.protobuf.json_format import ParseDict
    from kipy.proto.board import board_types_pb2
    from kipy.board_types import (
        AlignedDimension as AD,
        CenterDimension as CD,
        LeaderDimension as LD,
        OrthogonalDimension as OD,
        Pad as PadCls,
        RadialDimension as RD,
        to_concrete_board_shape,
    )

    with open(path) as f:
        raw = json.load(f)

    result: dict[str, Any] = {
        "board_name": raw.get("board_name", ""),
        "shapes": [],
        "dimensions": [],
        "pads": [],
    }

    _DIM_CLS = {
        "aligned": AD,
        "orthogonal": OD,
        "leader": LD,
        "center": CD,
        "radial": RD,
    }

    for entry in raw.get("shapes", []):
        proto = ParseDict(entry["proto"], board_types_pb2.BoardGraphicShape())
        from kipy.board_types import BoardShape as BS
        board_shape = BS(proto)
        concrete = to_concrete_board_shape(board_shape)
        if concrete is not None:
            result["shapes"].append(concrete)

    for entry in raw.get("dimensions", []):
        proto = ParseDict(entry["proto"], board_types_pb2.Dimension())
        dim_cls = _DIM_CLS.get(entry["type"])
        if dim_cls is not None:
            result["dimensions"].append(dim_cls(proto))

    for entry in raw.get("pads", []):
        proto = ParseDict(entry["proto"], board_types_pb2.Pad())
        result["pads"].append(PadCls(proto))

    return result


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "board_fixture.json"
    extract(output)
