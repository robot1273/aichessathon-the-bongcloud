"""Fetch the vendored Polyglot opening book (build-machine only).

The platform has no network, so the book must ship inside submission.zip.
The agent enforces the competition's fullmove-20 boundary before every probe.

Source: CodeKiddy Polyglot collection, mirrored by Chris Whittington.
Usage: uv run python scripts/fetch_book.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = ROOT / "book"
BOOK_PATH = BOOK_DIR / "book.bin"

BOOK_URL = (
    "https://github.com/ChrisWhittington/polyglot-books/"
    "releases/download/v1.0/codekiddy.zip"
)
ARCHIVE_SHA256 = "483113fa5e14267439ee12e6a92624b8c7521b8aeb1a7b82344e6b6eb5c57664"
BOOK_SHA256 = "46e8ad19a960bbc491a64d5b54ff3648b9ac2c10666cf0fc29dcfc08e249e79c"
BOOK_MEMBER = "codekiddy.bin"
BOOK_SIZE = 16_484_048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(destination: Path) -> None:
    request = urllib.request.Request(
        BOOK_URL, headers={"User-Agent": "aichessathon-starter-fetch/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)


def _valid_book(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == BOOK_SIZE and _sha256(path) == BOOK_SHA256


def fetch_book() -> None:
    if _valid_book(BOOK_PATH):
        print(f"book/: {BOOK_PATH.name} present, skipping download")
        return

    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "codekiddy.zip"
        extracted_path = Path(temporary) / BOOK_MEMBER
        print("book/: downloading CodeKiddy Polyglot book ...")
        _download(archive_path)
        if _sha256(archive_path) != ARCHIVE_SHA256:
            raise OSError("opening-book archive checksum mismatch")
        with (
            zipfile.ZipFile(archive_path) as archive,
            archive.open(BOOK_MEMBER) as source,
            extracted_path.open("wb") as destination,
        ):
            shutil.copyfileobj(source, destination)
        if not _valid_book(extracted_path):
            raise OSError("opening-book payload checksum mismatch")
        extracted_path.replace(BOOK_PATH)

    print(f"book/: {BOOK_PATH.stat().st_size:,} bytes")


def main() -> None:
    fetch_book()


if __name__ == "__main__":
    sys.exit(main())
