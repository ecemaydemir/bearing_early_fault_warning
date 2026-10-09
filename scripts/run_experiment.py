"""Fit every detector on each test's healthy period, raise alarms and evaluate early warning.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_efw import plots  # noqa: E402
from bearing_efw.alarm import alarm_episodes, first_alarm, first_persistent_alarm, fit_threshold, k_of_n  # noqa: E402
from bearing_efw.dataset import TESTS  # noqa: E402
from bearing_efw.models import build_models  # noqa: E402
from bearing_efw.preprocess import load_features, prepare  # noqa: E402


DEFECTS = {"env_bpfo": "outer race", "env_bpfi": "inner race", "env_bsf": "rolling element"}


def diagnose(d, alarm_h: float | None, n: int = 12) -> str:
    """Which defect frequency carries the most excess envelope energy right after the alarm.

    Uses the standardized features (z-scores vs. the bearing's own healthy baseline),
    averaged over the first ``n`` snapshots of the alarm and over the bearing's channels.
    """
    if alarm_h is None:
        return ""
    cols = list(d.raw.columns)
    after = np.flatnonzero(d.hours >= alarm_h)[:n]
    z = {k: np.mean([d.X[after, cols.index(c)].mean() for c in cols if c.startswith(k + "_")]) for k in DEFECTS}
    best = max(z, key=z.get)
    return f"{DEFECTS[best]} (z={z[best]:.1f})" if z[best] > 3 else "no clear defect frequency"


def run_test(test: str, cfg: dict, features_dir: Path, fig_dir: Path) -> pd.DataFrame:
    spec = TESTS[test]
    data = prepare(load_features(features_dir, test), cfg["split"]["train_end"], cfg["split"]["calib_end"])
    columns = list(next(iter(data.values())).raw.columns)
    duration_h = max(d.hours[-1] for d in data.values())

    results, rows = {}, []
    for model in build_models(cfg, columns):
        model.fit([d.X[d.train] for d in data.values()])
        results[model.name] = {}
        for b, d in data.items():
            score = model.score(d.X)
            thr = fit_threshold(score[d.calib], cfg["threshold"]["quantile"], cfg["threshold"]["margin"])
            alarm = k_of_n(score > thr, cfg["alarm"]["k"], cfg["alarm"]["n"])
            calib_end_h = d.hours[d.calib][-1]
            persistent = first_persistent_alarm(alarm, d.hours, calib_end_h)
            in_healthy = (d.life >= cfg["split"]["calib_end"]) & (d.life < cfg["healthy_end"][test])
            false_eps = [e for e in alarm_episodes(alarm & in_healthy)]
            results[model.name][b] = {"score": score, "threshold": thr, "alarm": alarm, "persistent_alarm_h": persistent}
            rows.append(
                {
                    "test": test,
                    "detector": model.name,
                    "bearing": b,
                    "failed": b in spec.failed,
                    "failure_mode": spec.failed.get(b, ""),
                    "first_alarm_h": first_alarm(alarm, d.hours, calib_end_h),
                    "persistent_alarm_h": persistent,
                    "lead_time_h": None if persistent is None else duration_h - persistent,
                    "lead_time_pct_life": None if persistent is None else 100 * (duration_h - persistent) / duration_h,
                    "diagnosis": diagnose(d, persistent),
                    "false_alarm_episodes": len(false_eps),
                    "healthy_eval_hours": float(np.ptp(d.hours[in_healthy])) if in_healthy.any() else 0.0,
                }
            )
        print(f"  {test} / {model.name} done", flush=True)

    plots.health_indicators(data, test, spec, cfg, fig_dir / f"{test}_indicators.png")
    plots.detector_panel(results, data, test, spec, cfg, fig_dir / f"{test}_detectors.png")

    scores = pd.concat(
        [
            pd.DataFrame({"test": test, "detector": name, "bearing": b, "timestamp": data[b].timestamps,
                          "hours": data[b].hours, "score": r["score"], "threshold": r["threshold"], "alarm": r["alarm"]})
            for name, per_b in results.items()
            for b, r in per_b.items()
        ]
    )
    return pd.DataFrame(rows), scores, duration_h


def main(config: Path, features_dir: Path, reports: Path) -> None:
    cfg = yaml.safe_load(config.read_text())
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summaries, all_scores, durations = [], [], {}
    for test in TESTS:
        s, sc, dur = run_test(test, cfg, features_dir, fig_dir)
        summaries.append(s)
        all_scores.append(sc)
        durations[test] = dur
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(reports / "summary.csv", index=False, float_format="%.2f")
    pd.concat(all_scores).to_csv(reports / "scores.csv.gz", index=False, float_format="%.5g")
    plots.lead_time_bars(summary, fig_dir / "lead_times.png")

    # Compact headline table: one row per detector
    headline = []
    for det, g in summary.groupby("detector", sort=False):
        fails = g[g.failed]
        row = {"detector": det}
        for r in fails.itertuples():
            row[f"{r.test} B{r.bearing} lead [h]"] = None if pd.isna(r.lead_time_h) else round(r.lead_time_h, 1)
        fa = g.groupby("test").apply(lambda x: x.false_alarm_episodes.sum(), include_groups=False)
        hrs = g.groupby("test").healthy_eval_hours.sum()
        row["false alarms / 1000 bearing-h"] = round(1000 * fa.sum() / hrs.sum(), 2)
        row["false alarm episodes (t1/t2/t3)"] = "/".join(str(int(fa.get(t, 0))) for t in TESTS)
        headline.append(row)
    headline = pd.DataFrame(headline)
    (reports / "results.md").write_text(headline.to_markdown(index=False) + "\n")
    (reports / "durations.json").write_text(json.dumps({k: round(v, 1) for k, v in durations.items()}, indent=2))
    print(headline.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--features", type=Path, default=ROOT / "data/features")
    p.add_argument("--reports", type=Path, default=ROOT / "reports")
    a = p.parse_args()
    main(a.config, a.features, a.reports)
