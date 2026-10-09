"""Turn the long feature tables into per-bearing matrices with train/calibration splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

# Strictly positive, heavy-tailed indicators are modelled in log space.
LOG_FEATURES = [f for f in FEATURE_NAMES if f != "skewness"]
DEAD_RMS = 0.01  # snapshots with RMS below this are sensor drop-outs (e.g. after shutdown)


def load_features(features_dir: Path, test: str) -> pd.DataFrame:
    df = pd.read_csv(Path(features_dir) / f"{test}_features.csv.gz", parse_dates=["timestamp"])
    # Drop snapshots where the acquisition was dead on any channel
    dead = df.groupby("timestamp")["rms"].min() < DEAD_RMS
    return df[~df["timestamp"].isin(dead[dead].index)].reset_index(drop=True)


@dataclass
class BearingData:
    bearing: int
    timestamps: pd.DatetimeIndex
    hours: np.ndarray  # hours since start of the test
    life: np.ndarray  # fraction of the test's snapshots seen so far, 0..1
    raw: pd.DataFrame  # untransformed features (channel-suffixed columns)
    X: np.ndarray  # log-transformed, standardized with train statistics
    train: np.ndarray  # boolean masks
    calib: np.ndarray


def wide_by_bearing(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """One row per timestamp, columns ``<feature>_ch<k>`` for each of the bearing's channels."""
    out = {}
    for bearing, g in df.groupby("bearing"):
        w = g.pivot(index="timestamp", columns="channel", values=FEATURE_NAMES)
        w.columns = [f"{f}_ch{c}" for f, c in w.columns]
        out[int(bearing)] = w.sort_index()
    return out


def prepare(df: pd.DataFrame, train_end: float, calib_end: float) -> dict[int, BearingData]:
    t0 = df["timestamp"].min()
    result = {}
    for bearing, w in wide_by_bearing(df).items():
        hours = ((w.index - t0).total_seconds() / 3600).to_numpy()
        # Splits use the snapshot index, not wall-clock time: test1 was recorded in
        # bursts with multi-day gaps, so a time-based split would leave almost no
        # training data.
        life = np.arange(len(w)) / len(w)
        train = life < train_end
        calib = (life >= train_end) & (life < calib_end)

        Z = w.copy()
        log_cols = [c for c in Z.columns if c.rsplit("_ch", 1)[0] in LOG_FEATURES]
        Z[log_cols] = np.log(Z[log_cols].clip(lower=1e-9))
        mu, sd = Z[train].mean(), Z[train].std().replace(0, 1)
        X = ((Z - mu) / sd).to_numpy(dtype=np.float32)

        result[bearing] = BearingData(bearing, w.index, hours, life, w, X, train, calib)
    return result
