"""Static figures for the README and reports."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BEARING_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
ALARM = "#e34948"  # status: critical
SPLIT = "#8a8984"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "font.size": 9,
        "lines.linewidth": 1.2,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _splits(ax, data):
    d = next(iter(data.values()))
    ax.axvspan(0, d.hours[d.train][-1], color=SPLIT, alpha=0.08, lw=0)
    ax.axvspan(d.hours[d.train][-1], d.hours[d.calib][-1], color=SPLIT, alpha=0.16, lw=0)


def _gapped(hours: np.ndarray, y: np.ndarray, max_gap_h: float = 2.0):
    """Insert NaNs at recording gaps so lines are not drawn across missing days."""
    y = np.asarray(y, dtype=float)
    gaps = np.flatnonzero(np.diff(hours) > max_gap_h) + 1
    return np.insert(hours.astype(float), gaps, np.nan), np.insert(y, gaps, np.nan)


def health_indicators(data, test, spec, cfg, out: Path):
    """RMS, kurtosis and outer/inner race envelope energy for all bearings over the test."""
    feats = [("rms", "RMS [g]"), ("kurtosis", "Kurtosis"), ("env_bpfo", "Envelope energy @ BPFO"), ("env_bpfi", "Envelope energy @ BPFI")]
    duration_h = max(d.hours[-1] for d in data.values())
    fig, axes = plt.subplots(len(feats), 1, figsize=(10, 9), sharex=True)
    for ax, (f, label) in zip(axes, feats):
        _splits(ax, data)
        for b, d in data.items():
            y = d.raw[f"{f}_ch0"].rolling(5, min_periods=1).median()
            lab = f"Bearing {b}" + (f" ({spec.failed[b]})" if b in spec.failed else "")
            ax.plot(*_gapped(d.hours, y), color=BEARING_COLORS[b], label=lab, lw=1.0 if b in spec.failed else 0.8)
        ax.set_ylabel(label)
        if f in ("rms", "kurtosis"):
            ax.set_yscale("log")
    axes[0].legend(ncol=2, loc="upper left")
    axes[0].set_title(f"{test}: condition indicators (shaded = train / calibration period)")
    axes[-1].set_xlabel("Hours since start of test")
    fig.savefig(out)
    plt.close(fig)


def detector_panel(results, data, test, spec, cfg, out: Path):
    """One row per detector: anomaly score of every bearing, threshold and alarm onset."""
    names = list(results)
    duration_h = max(d.hours[-1] for d in data.values())
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.3 * len(names)), sharex=True)
    failed = sorted(spec.failed)
    for ax, name in zip(np.atleast_1d(axes), names):
        _splits(ax, data)
        for b, d in data.items():
            r = results[name][b]
            # normalize by the bearing's own threshold so all bearings share one axis
            base = np.median(r["score"][d.calib])
            y = (r["score"] - base) / (r["threshold"] - base)
            ax.plot(*_gapped(d.hours, y), color=BEARING_COLORS[b], lw=1.0 if b in spec.failed else 0.7,
                    alpha=1.0 if b in spec.failed else 0.55, label=f"Bearing {b}")
            if b in spec.failed and r["persistent_alarm_h"] is not None:
                ax.axvline(r["persistent_alarm_h"], color=ALARM, lw=1.4)
                ax.annotate(f"alarm B{b}: {duration_h - r['persistent_alarm_h']:.0f} h before end",
                            (r["persistent_alarm_h"], 0.97 - 0.14 * failed.index(b)), xycoords=("data", "axes fraction"),
                            ha="right", va="top", color=INK, fontsize=8, xytext=(-4, 0), textcoords="offset points")
        ax.axhline(1, color=INK_2, ls="--", lw=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("score / threshold")
        ax.set_title(name, loc="left")
    np.atleast_1d(axes)[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.35))
    np.atleast_1d(axes)[-1].set_xlabel("Hours since start of test")
    fig.suptitle(f"{test}: anomaly scores (dashed = alarm threshold, red = persistent alarm on failed bearing)", y=1.02, fontsize=10, color=INK)
    fig.savefig(out)
    plt.close(fig)


def lead_time_bars(summary, out: Path):
    """Lead time of the persistent alarm on each failed bearing, per detector."""
    rows = summary[summary["failed"]].assign(case=lambda d: d.test + " B" + d.bearing.astype(str))
    table = rows.pivot(index="case", columns="detector", values="lead_time_h")[list(dict.fromkeys(rows.detector))]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    n = len(table.columns)
    w = 0.8 / n
    x = np.arange(len(table))
    for i, det in enumerate(table.columns):
        vals = table[det].fillna(0).to_numpy()
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=colors[i], label=det)
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in vals], fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, table.index)
    ax.set_ylabel("Warning lead time [h]")
    ax.set_title("Hours between the persistent alarm and the end of the test (higher = earlier warning)", loc="left")
    ax.legend(ncol=n, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.savefig(out)
    plt.close(fig)
