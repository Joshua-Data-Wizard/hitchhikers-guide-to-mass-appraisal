"""
compress_data.py — MAINTAINER TOOL. Readers do not run this.

Takes the raw Lee County taxroll, removes owners' names and mailing addresses,
packs it into committable .gz archives, and writes a manifest of checksums.
Readers unpack it again with `extract_data.py`.

    uv run python scripts/compress_data.py

Put the county's files in data/ first. They are changed in place.


It removes owners' names and mailing addresses
----------------------------------------------
Florida's roll files are public, and the Department of Revenue publishes them
with every owner's name and mailing address. Nothing in this project uses
those columns, so they are not shipped in it. Fourteen go, from both the NAL
and the shapefile's attribute table:

    OWN_NAME  OWN_ADDR1  OWN_ADDR2  OWN_CITY  OWN_STATE  OWN_ZIPCD  OWN_STATE_DOM
    FIDU_NAME FIDU_ADDR1 FIDU_ADDR2 FIDU_CITY FIDU_STATE FIDU_ZIPCD FIDU_CD

(In the .dbf, OWN_STATE_DOM is cut to OWN_STATE_, because dBase field names
stop at ten characters.) The property's own address, PHY_ADDR1 and PHY_ADDR2,
stays. It describes the land, and the chat assistant uses it.

Everything else stays exactly as it was published, value for value:

  - The NAL is cut as raw text. Every line is split into its exact fields,
    quotes and all, the fourteen are dropped, and the rest are joined with the
    same commas and line ending. Python's csv module is NOT used to write it,
    because it re-quotes fields and changes every line of this file.
  - The .dbf is cut byte by byte. Its records are fixed width, so the fourteen
    fields are sliced out of each one and the header is rewritten to match.
    The .shp, .shx and .prj are not touched, so every shape stays identical.

Each step checks its own work before replacing anything: every kept value must
read back exactly as it was, or the original file is left alone. And if any
OWN_ or FIDU_ column survives, from a future roll with a new one, the script
stops and says so, rather than shipping it.

Running it again on files already cut changes nothing. The manifest records the
county's original checksum for each file and the columns removed, so anyone can
download the roll from the Department of Revenue, run this, and get our files
exactly.


Why compressed
--------------
**GitHub refuses any push containing a file over 100 MB.** Gzip gets every
file under it, because fixed-width DBF records and comma-separated text are
enormously repetitive. The sizes, after removing the owner columns, are in
data/README.md and in the table this script prints.

    The .shp is the tight one at about 83 MB. It is packed coordinate
    doubles, which barely compress. If a future roll grows by ~20%, that file
    will cross 100 MB and this approach needs revisiting (split archives, or a
    trimmed sample). Worth re-checking each roll year.

An archive is only rebuilt when its file changed. Gzip stamps the time into
every archive it writes, so rebuilding an unchanged file would still give git
a new 80 MB blob to carry forever. New archives are stamped with time zero, so
packing the same file twice gives the same bytes.


It also writes the missing .prj
-------------------------------
The shapefile as supplied has **no .prj file**, so every tool that
opens it — geopandas, QGIS, ArcGIS — reports "no coordinate system" and
refuses to reproject. The notebook works around it by hardcoding EPSG:2237.

This script generates the .prj, which makes the dataset self-describing for
everyone else too. It is a ~400 byte text file and is committed directly
rather than archived.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import os
import re
import shutil
import struct
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

# Owners' names and mailing addresses: removed before anything is packed.
NAL_PERSONAL_COLUMNS = [
    "OWN_NAME", "OWN_ADDR1", "OWN_ADDR2", "OWN_CITY", "OWN_STATE", "OWN_ZIPCD", "OWN_STATE_DOM",
    "FIDU_NAME", "FIDU_ADDR1", "FIDU_ADDR2", "FIDU_CITY", "FIDU_STATE", "FIDU_ZIPCD", "FIDU_CD",
]
# The same fourteen in the .dbf, where a field name stops at ten characters.
DBF_PERSONAL_COLUMNS = [name[:10] for name in NAL_PERSONAL_COLUMNS]
STRIP = {
    "NAL46F202501.csv": ("csv", NAL_PERSONAL_COLUMNS),
    "lee_2025par.dbf": ("dbf", DBF_PERSONAL_COLUMNS),
}
# Anything that still looks like this after the strip stops the script.
LOOKS_PERSONAL = re.compile(r"^(OWN_|FIDU_)")

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


# ---------------------------------------------------------------------------
#  Removing the owner columns
# ---------------------------------------------------------------------------

def split_raw_csv_line(line: str) -> list[str]:
    """Split one CSV line into its exact raw fields, quotes and all.

    A comma inside quotes belongs to the field. Joining the result with commas
    gives back the line exactly, which is what lets a kept field keep its bytes.
    """
    fields, current, quoted = [], [], False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        if ch == "," and not quoted:
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    fields.append("".join(current))
    return fields


def strip_csv(path: Path, drop: list[str]) -> list[str]:
    """Remove columns from a CSV as raw text. Returns the names removed, or [] if already gone."""
    with open(path, "r", encoding="ascii", newline="") as f:
        header = split_raw_csv_line(f.readline().rstrip("\r\n"))
    names = [h.strip('"') for h in header]
    present = [d for d in drop if d in names]
    if not present:
        return []
    if len(present) != len(drop):
        raise SystemExit(f"{path.name}: only some of the owner columns are here ({present}). "
                         "Check the file's layout before going further.")
    gone = {names.index(d) for d in drop}

    tmp = path.with_name(path.name + ".stripping")
    with open(path, "r", encoding="ascii", newline="") as fi, open(tmp, "w", encoding="ascii", newline="") as fo:
        for number, line in enumerate(fi, start=1):
            body = line.rstrip("\r\n")
            raw = split_raw_csv_line(body)
            if len(raw) != len(names) or ",".join(raw) != body:
                tmp.unlink()
                raise SystemExit(f"{path.name} line {number}: could not split it exactly. Nothing was changed.")
            fo.write(",".join(r for i, r in enumerate(raw) if i not in gone) + line[len(body):])

    # Check every kept value reads back exactly as it was, before replacing anything.
    # zip_longest, not zip: zip stops quietly at the shorter file, and a cut-short
    # file would pass. A check that passes on missing lines is no check.
    with open(path, "r", encoding="ascii", newline="") as old, open(tmp, "r", encoding="ascii", newline="") as new:
        for number, (a, b) in enumerate(itertools.zip_longest(csv.reader(old), csv.reader(new)), start=1):
            if a is None or b is None or [v for i, v in enumerate(a) if i not in gone] != b:
                tmp.unlink()
                raise SystemExit(f"{path.name} line {number}: a kept value changed. Nothing was changed.")
    os.replace(tmp, path)
    return present


def read_dbf_header(f) -> tuple[bytes, int, int, int, list[tuple[str, bytes, int]]]:
    head = f.read(32)
    count, header_len, record_len = struct.unpack("<IHH", head[4:12])
    fields = []
    while True:
        descriptor = f.read(32)
        if descriptor[:1] == b"\x0d":
            break
        fields.append((descriptor[:11].split(b"\x00")[0].decode("ascii"), descriptor, descriptor[16]))
    return head, count, header_len, record_len, fields


def strip_dbf(path: Path, drop: list[str]) -> list[str]:
    """Remove fields from a dBase file, byte by byte. Returns the names removed, or [] if already gone."""
    with open(path, "rb") as f:
        head, count, header_len, record_len, fields = read_dbf_header(f)
    names = [n for n, _, _ in fields]
    present = [d for d in drop if d in names]
    if not present:
        return []
    if len(present) != len(drop):
        raise SystemExit(f"{path.name}: only some of the owner fields are here ({present}). "
                         "Check the file's layout before going further.")

    slices, offset = [], 1
    for name, descriptor, width in fields:
        slices.append((name, descriptor, offset, width))
        offset += width
    if offset != record_len:
        raise SystemExit(f"{path.name}: the field widths do not add up to the record length. Nothing was changed.")
    keep = [s for s in slices if s[0] not in drop]
    new_record_len = 1 + sum(width for _, _, _, width in keep)
    new_header_len = 32 + 32 * len(keep) + 1

    tmp = path.with_name(path.name + ".stripping")
    with open(path, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(head[:8] + struct.pack("<HH", new_header_len, new_record_len) + head[12:32])
        fo.write(b"".join(descriptor for _, descriptor, _, _ in keep) + b"\x0d")
        fi.seek(header_len)
        for _ in range(count):
            record = fi.read(record_len)
            fo.write(record[:1] + b"".join(record[o:o + w] for _, _, o, w in keep))
        fo.write(fi.read())  # the end-of-file marker, exactly as it was

    # Check every kept field of every record is byte-identical, before replacing anything.
    with open(path, "rb") as old, open(tmp, "rb") as new:
        old.seek(header_len)
        _, new_count, new_header, new_len, new_fields = read_dbf_header(new)
        if new_count != count or [n for n, _, _ in new_fields] != [s[0] for s in keep]:
            tmp.unlink()
            raise SystemExit(f"{path.name}: the new header does not match. Nothing was changed.")
        new.seek(new_header)
        for _ in range(count):
            a, b = old.read(record_len), new.read(new_len)
            if a[:1] + b"".join(a[o:o + w] for _, _, o, w in keep) != b:
                tmp.unlink()
                raise SystemExit(f"{path.name}: a kept field changed. Nothing was changed.")
    os.replace(tmp, path)
    return present


def columns_left(path: Path, kind: str) -> list[str]:
    if kind == "csv":
        with open(path, "r", encoding="ascii", newline="") as f:
            return [h.strip('"') for h in split_raw_csv_line(f.readline().rstrip("\r\n"))]
    with open(path, "rb") as f:
        return [n for n, _, _ in read_dbf_header(f)[4]]


def pack(src: Path, dst: Path) -> None:
    """Gzip at level 9, stamped with time zero, so the same file always packs to the same bytes."""
    tmp = dst.with_name(dst.name + ".packing")
    with open(src, "rb") as fi, open(tmp, "wb") as raw, \
            gzip.GzipFile(filename=src.name, mode="wb", fileobj=raw, compresslevel=9, mtime=0) as fo:
        shutil.copyfileobj(fi, fo, length=CHUNK)
    os.replace(tmp, dst)


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

    manifest_path = data / "data_manifest.json"
    previous = json.loads(manifest_path.read_bytes().decode("utf-8"))["files"] if manifest_path.exists() else {}

    print("\nRemoving owners' names and mailing addresses:")
    provenance: dict[str, dict[str, object]] = {}
    for name, (kind, drop) in STRIP.items():
        path = data / name
        source = sha256(path)
        removed = strip_csv(path, drop) if kind == "csv" else strip_dbf(path, drop)
        if removed:
            provenance[name] = {"source_sha256": source, "removed_columns": removed}
            print(f"  {name:<24} removed {len(removed)} columns")
        else:
            earlier = previous.get(name, {})
            provenance[name] = {k: earlier[k] for k in ("source_sha256", "removed_columns") if k in earlier}
            print(f"  {name:<24} already removed, nothing to do")
        survivors = [c for c in columns_left(path, kind) if LOOKS_PERSONAL.match(c)]
        if survivors:
            raise SystemExit(f"  !! {name} still has {survivors}. Add them to the lists at the top of "
                             "this script, and read what they are before shipping anything.")

    print("\nGenerating the missing projection file:")
    prj_name = write_prj(data)

    manifest: dict[str, object] = {
        "dataset": "Lee County, Florida - 2025 certified tax roll, owners' names and mailing addresses removed",
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
        raw_sha = sha256(src)

        earlier = previous.get(name, {})
        unchanged = (dst.exists() and earlier.get("raw_sha256") == raw_sha
                     and earlier.get("archive_sha256") == sha256(dst))
        t0 = time.perf_counter()
        if not unchanged:
            pack(src, dst)
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
        print(f"{'':<26}" + ("unchanged, archive kept as it was" if unchanged else f"packed in {elapsed:.1f}s"), flush=True)

        manifest["files"][name] = {
            "archive": name + ".gz",
            "raw_bytes": raw_size,
            "archive_bytes": gz_size,
            "raw_sha256": raw_sha,
            "archive_sha256": sha256(dst),
            **provenance.get(name, {}),
        }

    print("-" * 78)
    print(f"{'TOTAL':<26}{mb(total_raw):>10,.1f}MB{mb(total_gz):>10,.1f}MB"
          f"{total_raw/total_gz:>7.1f}x")

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
