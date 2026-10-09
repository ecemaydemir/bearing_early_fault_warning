# Excerpt from the full (private) project, shared for illustration only. All rights reserved.

"""From anomaly scores to operator-facing alarms, and how to evaluate them."""
from __future__ import annotations

import numpy as np


def fit_threshold(calib_scores: np.ndarray, quantile: float, margin: float) -> float:
    """High quantile of healthy calibration scores, pushed out by ``margin`` times its
    distance from the median. Works for any score scale or sign."""
    q, med = np.quantile(calib_scores, quantile), np.median(calib_scores)
    return float(q + margin * (q - med))


def k_of_n(exceed: np.ndarray, k: int, n: int) -> np.ndarray:
    """Alarm at t when at least k of the last n snapshots (inclusive) exceed the threshold.

    This debounces single noisy spikes, the main source of nuisance alarms in practice.
    """
    c = np.convolve(exceed.astype(int), np.ones(n, dtype=int))[: len(exceed)]
    return c >= k


def alarm_episodes(alarm: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of contiguous alarm runs, end exclusive."""
    a = np.concatenate([[0], alarm.astype(int), [0]])
    d = np.diff(a)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def first_persistent_alarm(alarm: np.ndarray, hours: np.ndarray, start_h: float) -> float | None:
    """Time of the first alarm after ``start_h`` that is never followed by a full clear.

    A real degradation does not heal, so the warning we report to the operator is the
    start of the final alarm episode that lasts until the end of the record. Earlier
    episodes that cleared are counted separately (as false or intermittent alarms).
    """
    eps = [(s, e) for s, e in alarm_episodes(alarm) if hours[s] >= start_h]
    if not eps:
        return None
    return float(hours[eps[-1][0]]) if eps[-1][1] == len(alarm) else None


def first_alarm(alarm: np.ndarray, hours: np.ndarray, start_h: float) -> float | None:
    idx = np.flatnonzero(alarm & (hours >= start_h))
    return float(hours[idx[0]]) if len(idx) else None
