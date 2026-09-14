from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from surgicalvision.constants import (
    EXPECTED_KNOT_SEQUENCE,
    EXPECTED_NEEDLE_SEQUENCE,
    EXPECTED_SUTURE_SEQUENCE,
)

JIGSAWS_DIR = Path(__file__).resolve().parent / "jigsaws"


@dataclass(frozen=True)
class DemoCase:
    id: str
    title: str
    description: str
    filename: str
    expected_sequence: tuple[str, ...]
    expected_duration_s: float
    dataset: str = "JIGSAWS"
    attribution: str = (
        "JIGSAWS clip from Johns Hopkins University / Intuitive Surgical "
        "(subject D, trial 005, capture 1)."
    )

    def path(self) -> Path:
        return JIGSAWS_DIR / self.filename

    def as_public(self) -> dict[str, str | float | list[str]]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "dataset": self.dataset,
            "expected_duration_s": self.expected_duration_s,
            "expected_sequence": list(self.expected_sequence),
        }


CASES: dict[str, DemoCase] = {
    "suturing": DemoCase(
        id="suturing",
        title="Suturing",
        description="da Vinci suturing on the JIGSAWS bench — reach, throw, knot, release.",
        filename="suturing.avi",
        expected_sequence=EXPECTED_SUTURE_SEQUENCE,
        expected_duration_s=70.0,
    ),
    "knot_tying": DemoCase(
        id="knot_tying",
        title="Knot tying",
        description="Knot-tying trial from the same JIGSAWS subject and capture.",
        filename="knot_tying.avi",
        expected_sequence=EXPECTED_KNOT_SEQUENCE,
        expected_duration_s=45.0,
    ),
    "needle_passing": DemoCase(
        id="needle_passing",
        title="Needle passing",
        description="Needle passing across the numbered rings — transfer and pull-through.",
        filename="needle_passing.avi",
        expected_sequence=EXPECTED_NEEDLE_SEQUENCE,
        expected_duration_s=75.0,
    ),
}


def get_case(case_id: str) -> DemoCase:
    try:
        return CASES[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown demo case: {case_id}") from exc
