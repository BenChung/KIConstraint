from __future__ import annotations

from dataclasses import dataclass

import slvs


@dataclass(frozen=True)
class Constraint:
    """Thin wrapper around a solver constraint."""

    handle: int
    type_code: int


class SolveResult:
    """Wrapper around the raw solve result."""

    def __init__(self, raw: slvs.Slvs_SolveResult) -> None:
        self._raw = raw

    @property
    def ok(self) -> bool:
        return self._raw["result"] in (
            slvs.ResultFlag.OKAY,
            slvs.ResultFlag.REDUNDANT_OKAY,
        )

    @property
    def dof(self) -> int:
        return self._raw["dof"]

    @property
    def result_code(self) -> int:
        return self._raw["result"]
