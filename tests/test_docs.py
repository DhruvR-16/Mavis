"""
Keeps the README's documented thresholds honest.

The README used to restate thresholds in prose in two places. Both copies went
stale when the tolerances were retuned, leaving the Key Features section
advertising numbers ~3x tighter than the engine actually used — and a sentence
claiming 35° was "above the measured 35.5° median", which it is not.

The numbers now live in exactly one table, and this test asserts that table
still matches what the code compares against. Prose that repeats a threshold is
prose that will eventually contradict the code.
"""

import os
import re

import pytest

from analyzer.base_analyzer import (
    ANGLE_TOLERANCE_DEG,
    DRIFT_TOLERANCE_BODY,
    FAULT_CONFIRM_FRAMES,
    SYMMETRY_TOLERANCE,
    TEMPO_MIN_SECONDS,
)
from exercises.bicep.analyzer import (
    DOWN_ANGLE_IDEAL,
    TORSO_TOLERANCE_DEG,
    UP_ANGLE_IDEAL,
)

README = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "readme.md"
)


@pytest.fixture(scope="module")
def readme() -> str:
    with open(README) as f:
        return f.read()


@pytest.fixture(scope="module")
def threshold_rows(readme) -> dict:
    """
    Maps the first cell of each row in the thresholds table to its 'Now' cell.

    Only the row label and the current-value column are read, so rewording the
    prose or the 'basis' column will not break these tests.
    """
    rows = {}
    for line in readme.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower() in {"check", ""} or set(cells[0]) <= {"-", " "}:
            continue
        rows[cells[0].lower()] = cells[2]
    return rows


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def test_thresholds_table_exists(threshold_rows):
    assert "elbow drift" in threshold_rows, "thresholds table is missing or reworded"


def test_documented_drift_matches_code(threshold_rows):
    assert DRIFT_TOLERANCE_BODY in numbers_in(threshold_rows["elbow drift"])


def test_documented_asymmetry_is_the_effective_value(threshold_rows):
    # The code compares against base + tolerance. Documenting the bare base
    # value is what produced the "35° is above 35.5°" contradiction.
    effective = SYMMETRY_TOLERANCE + ANGLE_TOLERANCE_DEG
    assert effective in numbers_in(threshold_rows["press asymmetry"])


def test_documented_tempo_matches_code(threshold_rows):
    assert TEMPO_MIN_SECONDS in numbers_in(threshold_rows["tempo"])


def test_documented_torso_tolerance_matches_code(threshold_rows):
    assert TORSO_TOLERANCE_DEG in numbers_in(threshold_rows["torso swing"])


def test_documented_rom_matches_code(threshold_rows):
    stated = numbers_in(threshold_rows.get("contraction / extension", ""))
    row = threshold_rows.get("contraction / extension", "")
    if "unchanged" in row.lower() and not stated:
        pytest.skip("row defers to the 'before' column")
    assert UP_ANGLE_IDEAL + ANGLE_TOLERANCE_DEG in stated
    assert DOWN_ANGLE_IDEAL - ANGLE_TOLERANCE_DEG in stated


def test_documented_fault_confirmation_matches_code(threshold_rows):
    assert FAULT_CONFIRM_FRAMES in numbers_in(threshold_rows["fault confirmation"])


def test_no_stale_thresholds_outside_the_before_column(readme):
    """
    The retired values may appear only in the 'Before tuning' column.

    Anywhere else they are a stale claim about current behaviour.
    """
    stale = {
        "18% of your shoulder width": "old drift tolerance",
        "imbalance beyond 20°": "old asymmetry tolerance",
        "faster than **0.8 s**": "old eccentric-only tempo rule",
    }
    for phrase, what in stale.items():
        assert phrase not in readme, f"README still advertises the {what}"


def test_symmetry_claim_is_not_self_contradictory(readme):
    # The measured median for real presses; the effective threshold has to sit
    # above it or the "tolerances clear normal variation" claim is false.
    measured_median = 35.5
    assert SYMMETRY_TOLERANCE + ANGLE_TOLERANCE_DEG > measured_median
