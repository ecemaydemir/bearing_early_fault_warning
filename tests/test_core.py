import numpy as np
import pytest

from bearing_efw.alarm import alarm_episodes, first_persistent_alarm, fit_threshold, k_of_n
from bearing_efw.dataset import FS, fault_frequencies
from bearing_efw.features import FEATURE_NAMES, snapshot_features


def test_fault_frequencies_match_literature():
    f = fault_frequencies()
    # Published values for the IMS rig (Rexnord ZA-2115 at 2000 rpm)
    assert f["bpfo"] == pytest.approx(236.4, abs=0.5)
    assert f["bpfi"] == pytest.approx(296.9, abs=0.5)
    assert f["bsf"] == pytest.approx(139.9, abs=0.5)


def test_features_of_gaussian_noise():
    x = np.random.default_rng(0).standard_normal(FS)
    feats = snapshot_features(x)
    assert list(feats) == FEATURE_NAMES
    assert feats["rms"] == pytest.approx(1.0, rel=0.02)
    assert feats["kurtosis"] == pytest.approx(3.0, abs=0.15)


def test_envelope_picks_up_outer_race_impacts():
    """Synthetic outer-race defect: resonance bursts repeating at BPFO."""
    rng = np.random.default_rng(1)
    t = np.arange(FS) / FS
    bpfo = fault_frequencies()["bpfo"]
    impacts = np.zeros(FS)
    impacts[(np.arange(0, 1, 1 / bpfo) * FS).astype(int)] = 1.0
    ring = np.exp(-t[:200] * 2000) * np.sin(2 * np.pi * 4000 * t[:200])
    faulty = np.convolve(impacts, ring)[:FS] * 3 + 0.3 * rng.standard_normal(FS)
    healthy = 0.3 * rng.standard_normal(FS)
    f_faulty, f_healthy = snapshot_features(faulty), snapshot_features(healthy)
    assert f_faulty["env_bpfo"] > 5 * f_healthy["env_bpfo"]
    assert f_faulty["env_bpfo"] > f_faulty["env_bpfi"]
    assert f_faulty["kurtosis"] > f_healthy["kurtosis"]


def test_k_of_n_debounces_single_spikes():
    exceed = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    assert not k_of_n(exceed, 2, 3)[:5].any()
    assert k_of_n(exceed, 2, 3)[6]


def test_persistent_alarm_ignores_episodes_that_clear():
    alarm = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    hours = np.arange(10, dtype=float)
    assert alarm_episodes(alarm) == [(1, 3), (6, 10)]
    assert first_persistent_alarm(alarm, hours, 0.0) == 6.0
    assert first_persistent_alarm(alarm[:8].copy() & False, hours[:8], 0.0) is None


def test_threshold_is_above_calibration_quantile():
    s = np.random.default_rng(0).normal(5, 1, 1000)
    thr = fit_threshold(s, 0.995, 0.5)
    assert thr > np.quantile(s, 0.995)
    # sign-agnostic: a shifted score gets a shifted threshold
    assert fit_threshold(s - 10, 0.995, 0.5) == pytest.approx(thr - 10)
