"""Reduce the raw IMS snapshots (~6 GB) to compact feature tables (~MBs).

Usage:
    python scripts/build_features.py --raw data/raw --out data/features
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_efw.dataset import TESTS, list_files, load_snapshot, parse_timestamp  # noqa: E402
from bearing_efw.features import snapshot_features  # noqa: E402


def _process(args: tuple[str, Path]) -> list[dict]:
    test, path = args
    spec = TESTS[test]
    data = load_snapshot(path)
    ts = parse_timestamp(path.name)
    rows = []
    for bearing, cols in spec.channels.items():
        for axis, col in enumerate(cols):
            rows.append({"timestamp": ts, "bearing": bearing, "channel": axis, **snapshot_features(data[:, col])})
    return rows


def build(raw: Path, out: Path, tests: list[str], workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for test in tests:
        files = list_files(raw, TESTS[test])
        print(f"{test}: {len(files)} snapshots", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            rows = [r for chunk in ex.map(_process, [(test, f) for f in files], chunksize=16) for r in chunk]
        df = pd.DataFrame(rows).sort_values(["bearing", "channel", "timestamp"])
        df.to_csv(out / f"{test}_features.csv.gz", index=False, float_format="%.6g")
        print(f"  -> {out / f'{test}_features.csv.gz'} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/features"))
    p.add_argument("--tests", nargs="+", default=list(TESTS))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    build(a.raw, a.out, a.tests, a.workers)
