"""
compress_data.py — MAINTAINER TOOL. Readers do not run this.

Packs the raw Lee County taxroll into committable .gz archives and writes a
manifest of checksums. Readers unpack it again with `extract_data.py`.

    uv run python scripts/compress_data.py


Why this exists
---------------
The raw roll is about 1.7 GB, and **GitHub refuses any push containing a file
over 100 MB.** Three of the five files are over it:

    lee_2025par.dbf    1,084 MB    11x over
    lee_2025par.shp      293 MB     3x over
    NAL46F202501.csv     272 MB     3x over

Gzip gets every one of them under the limit, because fixed-width DBF records
and comma-separated text are enormously repetitive. Measured 2026-10-26:

    NAL46F202501.csv      272 MB  ->  47 MB    5.8x
    SDF46F202501.csv       18 MB  ->   3 MB    5.6x
    lee_2025par.dbf     1,084 MB  ->  64 MB   16.9x
    lee_2025par.shp       293 MB  ->  83 MB    3.5x
    lee_2025par.shx         4 MB  ->   2 MB    2.3x
                        ────────     ──────
                        1,671 MB     199 MB

So the repository carries ~199 MB of archives, every file comfortably under
100 MB, and a reader gets the genuine county data back byte for byte.

    The .shp is the tight one at 83 MB — it is packed coordinate doubles,
    which barely compress. If a future roll grows by ~20%, that file will
    cross 100 MB and this approach needs revisiting (split archives, or a
    trimmed sample). Worth re-checking each roll year.


It also writes the missing .prj
-------------------------------
The shapefile Lee County supplied has **no .prj file**, so every tool that
opens it — geopandas, QGIS, ArcGIS — reports "no coordinate system" and
refuses to reproject. The notebook works around it by hardcoding EPSG:2237.

This script generates the .prj, which makes the dataset self-describing for
everyone else too. It is a ~400 byte text file and is committed directly
rather than archived.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

# Lee County is in the WEST zone of the Florida State Plane system.
# A different Florida county may be East (EPSG:2236) or North (EPSG:2238) —
# using the wrong one puts every parcel in the wrong place.
LEE_COUNTY_EPSG = 2237

RAW_FILES = [
    "NAL46F202501.csv",
    "SDF46F202501.csv",
    "lee_2025par.shp",
    "lee_2025par.shx",
    "lee_2025par.dbf",
]

GITHUB_FILE_LIMIT = 100 * 1024**2
GITHUB_WARN_LIMIT = 50 * 1024**2
CHUNK = 8 * 1024**2

# Same guard as extract_data.py — a Windows legacy code page turns any
# non-ASCII character in a print() into a fatal UnicodeEncodeError. See the
# longer note in extract_data.py for why this matters so much there.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass


def find_project_root(marker: str = "pyproject.toml") -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / marker).exists():
            return candidate
    raise SystemExit("Could not find the project root (no pyproject.toml above here).")


def mb(n: int) -> float:
    return n / 1024**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_prj(data_dir: Path) -> str | None:
    """Generate the .prj the county did not ship. Needs pyproj (dev-only)."""
    prj = data_dir / "lee_2025par.prj"
    try:
        from pyproj import CRS
    except ImportError:
        print("  ! pyproj not available, skipping .prj generation")
        return None

    crs = CRS.from_epsg(LEE_COUNTY_EPSG)
    # ESRI WKT1 is what shapefile readers expect in a .prj.
    wkt = crs.to_wkt(version="WKT1_ESRI")
    prj.write_bytes(wkt.encode("utf-8"))
    print(f"  wrote {prj.name}  ({len(wkt)} bytes)  -> {crs.name}")
    return prj.name


def main() -> int:
    root = find_project_root()
    data = root / "data"
    if not data.is_dir():
        raise SystemExit(f"No data/ directory at {data}")

    print("=" * 78)
    print("Packing the Lee County taxroll for GitHub")
    print("=" * 78)

    missing = [f for f in RAW_FILES if not (data / f).exists()]
    if missing:
        print("\nThese raw files are missing, so there is nothing to pack:")
        for f in missing:
            print(f"  - {f}")
        print("\nThis script is for the maintainer, who has the full roll.")
        return 1

    print("\nGenerating the missing projection file:")
    prj_name = write_prj(data)

    manifest: dict[str, object] = {
        "dataset": "Lee County, Florida - 2025 certified tax roll",
        "source_epsg": LEE_COUNTY_EPSG,
        "note": "Unpack with: uv run python scripts/extract_data.py",
        "committed_plain": [prj_name] if prj_name else [],
        "files": {},
    }

    print(f"\n{'file':<26}{'raw':>12}{'gzipped':>12}{'ratio':>8}   GitHub")
    print("-" * 78)

    total_raw = total_gz = 0
    over_limit = []

    for name in RAW_FILES:
        src = data / name
        dst = data / (name + ".gz")
        raw_size = src.stat().st_size

        t0 = time.perf_counter()
        with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo, length=CHUNK)
        elapsed = time.perf_counter() - t0

        gz_size = dst.stat().st_size
        total_raw += raw_size
        total_gz += gz_size

        if gz_size >= GITHUB_FILE_LIMIT:
            flag = "TOO BIG"
            over_limit.append(name)
        elif gz_size >= GITHUB_WARN_LIMIT:
            flag = "ok (>50MB warn)"
        else:
            flag = "ok"

        print(f"{name:<26}{mb(raw_size):>10,.1f}MB{mb(gz_size):>10,.1f}MB"
              f"{raw_size/gz_size:>7.1f}x   {flag}")
        print(f"{'':<26}compressed in {elapsed:.1f}s, checksumming...", flush=True)

        manifest["files"][name] = {
            "archive": name + ".gz",
            "raw_bytes": raw_size,
            "archive_bytes": gz_size,
            "raw_sha256": sha256(src),
            "archive_sha256": sha256(dst),
        }

    print("-" * 78)
    print(f"{'TOTAL':<26}{mb(total_raw):>10,.1f}MB{mb(total_gz):>10,.1f}MB"
          f"{total_raw/total_gz:>7.1f}x")

    manifest_path = data / "data_manifest.json"
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_path.write_bytes(payload)
    print(f"\nwrote {manifest_path.name} ({len(payload):,} bytes)")

    if over_limit:
        print("\n!! These archives are still over GitHub's 100 MB hard limit:")
        for f in over_limit:
            print(f"   - {f}.gz")
        print("   They CANNOT be pushed. See the note at the top of this file.")
        return 1

    print("\nEvery archive is under 100 MB. Safe to commit.")
    print("Remember: .gitignore allows data/*.gz and data/*.prj, and keeps the")
    print("extracted raw files out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
