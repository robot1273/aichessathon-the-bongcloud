"""Fetch vendored endgame tables + opening book (build-machine only).

The platform has no network, so everything must ship inside submission.zip.
Artifacts land in tb/ and book/ (both gitignored) and are packed via
`make zip --include tb --include book`.

Sources:
- Syzygy 3+4-man (70 files, ~4.3MB): jshriver/syzygy HuggingFace mirror of
  the standard Ronald de Man tables. Full 3-4-5 set is ~940MB (unshippable);
  5-man excluded. Measured 2026-09: 3-man 25KB, 4-man WDL 1.20MB + DTZ 2.92MB.
- gm2001.bin (0.49MB, 30,416 entries): GM games 2001-2013, >=2530 Elo,
  compiled by Oliver Deville, via ChrisWhittington/polyglot-books releases.
  Master-game statistics (not an engine book). Hit rate on the 8 sample
  rated openings: 3/8 (curated sidelines often miss any book).

Usage: uv run python scripts/fetch_tables.py [--tables-only | --book-only]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ROOT / "tb"
BOOK_DIR = ROOT / "book"

HF_BASE = "https://huggingface.co/datasets/jshriver/syzygy/resolve/main"
BOOK_URL = "https://github.com/ChrisWhittington/polyglot-books/releases/latest/download/gm2001.zip"

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
    req = urllib.request.Request(url, headers={"User-Agent": "aichessathon-starter-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)


def fetch_tables() -> None:
    names: list[str] = []
    for base in TABLES_3MAN + TABLES_4MAN:
        names += [f"{base}.rtbw", f"{base}.rtbz"]
    assert len(names) == EXPECTED_COUNT, f"{len(names)} != {EXPECTED_COUNT}"
    missing = [n for n in names if not (TB_DIR / n).is_file()]
    if not missing:
        print(f"tb/: all {len(names)} files present, skipping download")
        return
    print(f"tb/: downloading {len(missing)} missing files ...")
    for i, name in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {name}")
        _download(f"{HF_BASE}/{name}", TB_DIR / name)
    total = sum(p.stat().st_size for p in TB_DIR.glob("*.rtb*"))
    print(f"tb/: {len(names)} files, {total:,} bytes total")


def fetch_book() -> None:
    dest = BOOK_DIR / "book.bin"
    if dest.is_file():
        print(f"book/: {dest.name} present ({dest.stat().st_size:,} bytes), skipping")
        return
    import tempfile
    import zipfile

    print("book/: downloading gm2001.zip ...")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "gm2001.zip"
        _download(BOOK_URL, zpath)
        with zipfile.ZipFile(zpath) as archive:
            data = archive.read("gm2001.bin")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"book/: wrote {dest} ({len(data):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TB + book artifacts.")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--book-only", action="store_true")
    args = parser.parse_args()
    if args.book_only:
        fetch_book()
    elif args.tables_only:
        fetch_tables()
    else:
        fetch_tables()
        fetch_book()


if __name__ == "__main__":
    sys.exit(main())
