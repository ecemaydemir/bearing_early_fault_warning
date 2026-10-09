"""IMS (NASA / University of Cincinnati) bearing run-to-failure dataset metadata and loaders.

Each test is a folder of 1-second vibration snapshots sampled at 20 kHz
(20 480 samples per file), recorded every 5-10 minutes until a bearing failed.
File names are timestamps (``YYYY.MM.DD.HH.MM.SS``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FS = 20_480  # sampling rate [Hz]
SHAFT_RPM = 2000
SHAFT_HZ = SHAFT_RPM / 60.0

# Rexnord ZA-2115 double-row bearing (from the dataset readme)
N_ROLLERS = 16
ROLLER_D = 0.331  # in
PITCH_D = 2.815  # in
CONTACT_ANGLE_DEG = 15.17


def fault_frequencies(fr: float = SHAFT_HZ) -> dict[str, float]:
    """Characteristic defect frequencies [Hz] for the ZA-2115 bearing."""
    ratio = ROLLER_D / PITCH_D * np.cos(np.deg2rad(CONTACT_ANGLE_DEG))
    return {
        "bpfo": N_ROLLERS / 2 * fr * (1 - ratio),  # outer race
        "bpfi": N_ROLLERS / 2 * fr * (1 + ratio),  # inner race
        "bsf": PITCH_D / (2 * ROLLER_D) * fr * (1 - ratio**2),  # rolling element
        "ftf": fr / 2 * (1 - ratio),  # cage
    }


@dataclass(frozen=True)
class TestSpec:
    name: str
    folder: str  # relative to the raw data root
    channels: dict[int, tuple[int, ...]]  # bearing -> column indices in the raw file
    failed: dict[int, str] = field(default_factory=dict)  # bearing -> failure mode


TESTS: dict[str, TestSpec] = {
    "test1": TestSpec(
        name="test1",
        folder="1st_test",
        channels={1: (0, 1), 2: (2, 3), 3: (4, 5), 4: (6, 7)},
        failed={3: "inner race defect", 4: "roller element defect"},
    ),
    "test2": TestSpec(
        name="test2",
        folder="2nd_test",
        channels={1: (0,), 2: (1,), 3: (2,), 4: (3,)},
        failed={1: "outer race failure"},
    ),
    "test3": TestSpec(
        name="test3",
        # the official 3rd_test.rar extracts to 4th_test/txt
        folder="4th_test/txt",
        channels={1: (0,), 2: (1,), 3: (2,), 4: (3,)},
        failed={3: "outer race failure"},
    ),
}


def parse_timestamp(name: str) -> datetime:
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")


def list_files(raw_root: Path, spec: TestSpec) -> list[Path]:
    folder = Path(raw_root) / spec.folder
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {folder}")
    return files


def load_snapshot(path: Path) -> np.ndarray:
    """Return an (n_samples, n_channels) float32 array for one snapshot file."""
    return pd.read_csv(path, sep="\t", header=None, dtype=np.float32).to_numpy()
