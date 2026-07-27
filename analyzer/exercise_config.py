"""
Loads the shared exercise definitions from exercises.json at the repo root.

The same file is read by the web engine (frontend/src/engine/config.ts), so a
threshold only ever exists in one place. Before this, each runtime carried its
own hand-maintained copy of the numbers and they had already drifted apart.
"""

import json
import os
from functools import lru_cache

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exercises.json"
)


@lru_cache(maxsize=1)
def _load() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"Shared exercise config not found at {CONFIG_PATH}. "
            "It is required by both the Python and web engines."
        )


def tolerances() -> dict:
    return _load()["tolerances"]


def calibration() -> dict:
    return _load()["calibration"]


def exercise(key: str) -> dict:
    """Returns the definition for 'bicep' / 'shoulder'."""
    exercises = _load()["exercises"]
    if key not in exercises:
        raise KeyError(f"Unknown exercise '{key}'. Known: {sorted(exercises)}")
    return exercises[key]
