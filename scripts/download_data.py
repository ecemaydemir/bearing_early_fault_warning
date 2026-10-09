"""Download and extract the IMS bearing dataset (~1 GB download, ~6 GB extracted).

Source: NASA Prognostics Center of Excellence data repository (mirror on the PHM
Society's public S3 bucket). Extraction needs ``py7zr`` and an ``unar``/``unrar``
binary for the inner .rar archives (``brew install unar`` / ``apt install unar``).

Usage:
    python scripts/download_data.py --out data/raw
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "bearings.zip"
    if not zpath.exists() and not (out / "2nd_test").exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)

    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        import py7zr

        with py7zr.SevenZipFile(out / "4. Bearings" / "IMS.7z") as z:
            z.extractall(out)
        shutil.rmtree(out / "4. Bearings")
        zpath.unlink()

    tool = shutil.which("unar") or shutil.which("unrar")
    if tool is None:
        raise SystemExit("Install `unar` (or `unrar`) to extract the .rar archives.")
    for rar in sorted(out.glob("*_test.rar")):
        cmd = [tool, "-q", "-f", "-o", str(out), str(rar)] if tool.endswith("unar") else [tool, "x", "-o+", str(rar), str(out)]
        subprocess.run(cmd, check=True)
        rar.unlink()
    print("Done:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    main(p.parse_args().out)
