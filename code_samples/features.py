"""Per-snapshot condition indicators.

Every 1-second snapshot is reduced to a small vector of physically meaningful
features per channel: time-domain statistics, broadband spectral energy and
envelope-spectrum energy at the bearing defect frequencies.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats

from .dataset import FS, fault_frequencies

# Broadband FFT bands [Hz]. Upper limit is the Nyquist frequency (10.24 kHz).
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10240)]
# Envelope analysis: band-pass around the structural resonance, then demodulate
ENVELOPE_BAND = (2000, 8000)
FAULT_TOL_HZ = 3.0  # half-width of the window around each defect frequency
N_HARMONICS = 3

_FAULT_F = fault_frequencies()
_SOS = signal.butter(4, ENVELOPE_BAND, btype="bandpass", fs=FS, output="sos")


def time_features(x: np.ndarray) -> dict[str, float]:
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    return {
        "rms": rms,
        "peak": peak,
        "p2p": float(x.max() - x.min()),
        "kurtosis": float(stats.kurtosis(x, fisher=False)),
        "skewness": float(stats.skew(x)),
        "crest": peak / rms,
        "shape": rms / mean_abs,
        "impulse": peak / mean_abs,
    }


def _band_energy(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    return float(psd[m].sum())


def spectral_features(x: np.ndarray) -> dict[str, float]:
    x = x - x.mean()
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    psd = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 / len(x)
    total = psd.sum()
    out = {f"band_{lo // 1000}_{hi // 1000}k": _band_energy(freqs, psd, lo, hi) / total for lo, hi in BANDS}
    out["spectral_centroid"] = float((freqs * psd).sum() / total)

    # Envelope spectrum: impacts from a localized defect modulate the resonance
    # at the defect frequency, which shows up after demodulation.
    env = np.abs(signal.hilbert(signal.sosfiltfilt(_SOS, x)))
    env -= env.mean()
    env_spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    env_total = env_spec[(freqs > 5) & (freqs < 1000)].sum()
    for name in ("bpfo", "bpfi", "bsf"):
        e = sum(
            _band_energy(freqs, env_spec, k * _FAULT_F[name] - FAULT_TOL_HZ, k * _FAULT_F[name] + FAULT_TOL_HZ)
            for k in range(1, N_HARMONICS + 1)
        )
        out[f"env_{name}"] = e / env_total
    return out


def snapshot_features(x: np.ndarray) -> dict[str, float]:
    """All features for a single-channel 1-second snapshot."""
    return {**time_features(x), **spectral_features(x)}


FEATURE_NAMES = list(snapshot_features(np.random.default_rng(0).standard_normal(FS)).keys())
