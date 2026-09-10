"""Fetch vendored endgame tables (build-machine only).

The platform has no network, so everything must ship inside submission.zip.
Artifacts land in tb/ (gitignored) and are packed via `make zip` defaults.

Source: Syzygy 3+4-man (70 files, ~4.3MB): jshriver/syzygy HuggingFace
mirror of the standard Ronald de Man tables. Full 3-4-5 set is ~940MB
(unshippable); 5-man excluded. Measured 2026-09: 3-man 25KB, 4-man
WDL 1.20MB + DTZ 2.92MB.

Usage: uv run python scripts/fetch_tables.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ROOT / "tb"

HF_BASE = "https://huggingface.co/datasets/jshriver/syzygy/resolve/main"

# 5 three-man + 30 four-man tables, WDL (.rtbw) + DTZ (.rtbz) each.
# Exact table list verified against the mirror index (70 files total).
TABLES_3MAN = ["KBvK", "KNvK", "KPvK", "KQvK", "KRvK"]
TABLES_4MAN = [
    "KBBvK", "KBNvK", "KBPvK", "KBvKB", "KBvKN", "KBvKP", "KNNvK", "KNPvK",
    "KNvKN", "KNvKP", "KPPvK", "KPvKP", "KQBvK", "KQNvK", "KQPvK", "KQQvK",
    "KQRvK", "KQvKB", "KQvKN", "KQvKP", "KQvKQ", "KQvKR", "KRBvK", "KRNvK",
    "KRPvK", "KRRvK", "KRvKB", "KRvKN", "KRvKP", "KRvKR",
]

EXPECTED_COUNT = (len(TABLES_3MAN) + len(TABLES_4MAN)) * 2


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "aichessathon-starter-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, temporary.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        if temporary.stat().st_size == 0:
            raise OSError(f"empty download: {url}")
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_tables() -> None:
    names: list[str] = []
    for base in TABLES_3MAN + TABLES_4MAN:
        names += [f"{base}.rtbw", f"{base}.rtbz"]
    assert len(names) == EXPECTED_COUNT, f"{len(names)} != {EXPECTED_COUNT}"
    missing = [
        name
        for name in names
        if not (TB_DIR / name).is_file() or (TB_DIR / name).stat().st_size == 0
    ]
    if not missing:
        print(f"tb/: all {len(names)} files present, skipping download")
        return
    print(f"tb/: downloading {len(missing)} missing files ...")
    for i, name in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {name}")
        _download(f"{HF_BASE}/{name}", TB_DIR / name)
    total = sum((TB_DIR / name).stat().st_size for name in names)
    print(f"tb/: {len(names)} files, {total:,} bytes total")


def main() -> None:
    fetch_tables()


if __name__ == "__main__":
    sys.exit(main())
