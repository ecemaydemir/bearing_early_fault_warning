"""Signal-level figures for the technical report (needs the raw data in data/raw).

Usage:
    python scripts/make_report_figures.py --raw data/raw --out reports/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_efw import plots  # noqa: E402  (sets the shared matplotlib style)
from bearing_efw.dataset import FS, TESTS, fault_frequencies, list_files, load_snapshot  # noqa: E402
from bearing_efw.features import ENVELOPE_BAND  # noqa: E402

plt = plots.plt


def envelope_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sos = signal.butter(4, ENVELOPE_BAND, btype="bandpass", fs=FS, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x - x.mean())))
    env -= env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    return np.fft.rfftfreq(len(env), 1 / FS), spec


def main(raw: Path, out: Path) -> None:
    files = list_files(raw, TESTS["test2"])
    # healthy (start), degraded (~hour 100, after the alarm), late (~hour 160)
    picks = {"Healthy (hour 0)": files[0], "Degraded (hour 100)": files[600], "Near failure (hour 160)": files[960]}
    sig = {k: load_snapshot(p)[:, 0] for k, p in picks.items()}
    colors = ["#2a78d6", "#eda100", "#e34948"]

    # 1. raw waveforms
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
    t = np.arange(FS) / FS
    n = int(0.1 * FS)
    for ax, (k, x), c in zip(axes, sig.items(), colors):
        ax.plot(t[:n] * 1000, x[:n], color=c, lw=0.7)
        ax.set_title(f"{k}: RMS {np.sqrt(np.mean((x - x.mean()) ** 2)):.3f} g", loc="left")
        ax.set_ylabel("acc. [g]")
    axes[-1].set_xlabel("time [ms] (first 100 ms of a 1-s snapshot)")
    fig.suptitle("test2, bearing 1: raw vibration", y=1.0, fontsize=10)
    fig.savefig(out / "report_waveforms.png")
    plt.close(fig)

    # 2. amplitude spectrum
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for (k, x), c in zip(sig.items(), colors):
        f = np.fft.rfftfreq(len(x), 1 / FS)
        a = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) / len(x)
        ax.semilogy(f, signal.medfilt(a, 31), color=c, lw=0.9, label=k)
    ax.axvspan(*ENVELOPE_BAND, color=plots.SPLIT, alpha=0.1, lw=0)
    ax.text(np.mean(ENVELOPE_BAND), ax.get_ylim()[1], "envelope band", ha="center", va="top", color=plots.INK_2, fontsize=8)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("amplitude (median-smoothed)")
    ax.set_title("Amplitude spectrum: the degraded bearing excites the 2-8 kHz structural resonances", loc="left")
    ax.legend(loc="upper right")
    fig.savefig(out / "report_spectrum.png")
    plt.close(fig)

    # 3. envelope spectrum with defect frequencies
    ff = fault_frequencies()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    for ax, k, c in zip(axes, ["Healthy (hour 0)", "Degraded (hour 100)"], [colors[0], colors[1]]):
        f, s = envelope_spectrum(sig[k])
        m = f < 1000
        ax.plot(f[m], s[m], color=c, lw=0.8)
        for h in (1, 2, 3):
            ax.axvline(h * ff["bpfo"], color=plots.ALARM, ls="--", lw=0.8)
            ax.axvline(h * ff["bpfi"], color="#4a3aa7", ls=":", lw=0.8)
        ax.set_title(k, loc="left")
        ax.set_ylabel("envelope amplitude")
    axes[0].plot([], [], color=plots.ALARM, ls="--", label=f"BPFO × 1..3 ({ff['bpfo']:.1f} Hz)")
    axes[0].plot([], [], color="#4a3aa7", ls=":", label=f"BPFI × 1..3 ({ff['bpfi']:.1f} Hz)")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("frequency [Hz]")
    fig.suptitle("test2, bearing 1: envelope spectrum reveals impacts at the outer-race defect frequency", y=1.0, fontsize=10)
    fig.savefig(out / "report_envelope.png")
    plt.close(fig)
    print("written:", [p.name for p in sorted(out.glob("report_*.png"))])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    p.add_argument("--out", type=Path, default=ROOT / "reports/figures")
    a = p.parse_args()
    main(a.raw, a.out)
