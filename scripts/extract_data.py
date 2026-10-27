"""
extract_data.py — unpack the Lee County taxroll. 🐬

    uv run python scripts/extract_data.py

Run this ONCE after `uv sync`, before you open the notebook. It unpacks the
compressed taxroll in data/ into the real CSV and shapefile the notebook
reads. Takes well under a minute and about 1.7 GB of disk.

    uv run python scripts/extract_data.py --check    see what would happen
    uv run python scripts/extract_data.py --force    re-extract everything
    uv run python scripts/extract_data.py --clean    delete extracted files


DON'T PANIC — why isn't the data just sitting there?
----------------------------------------------------
Because GitHub refuses any single file larger than 100 MB, and the raw roll
is about 1.7 GB — the parcel attribute table alone (`lee_2025par.dbf`) is
1.08 GB. So the repository stores the data gzipped, which fits comfortably,
and this script puts it back byte for byte.

Nothing is downloaded. Everything you need is already in your clone.

The files this writes are in .gitignore, so they will never show up as
changes you might accidentally commit.


This script uses only the Python standard library
--------------------------------------------------
No pandas, no geopandas, nothing from pyproject.toml. That is deliberate:
unpacking your data is the very first thing you do, and it should not be
able to fail because of an unrelated dependency problem.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

CHUNK = 8 * 1024**2

# ─────────────────────────────────────────────────────────────────────────
#  Make printing unbreakable before we print anything.
#
#  Windows consoles still default to a legacy code page (cp1252), and on one
#  of those `print("🐬")` raises UnicodeEncodeError and kills the script
#  outright — before a single byte of data is extracted.
#
#  This is the FIRST command a reader runs, on exactly the kind of machine
#  that has that code page. It must not be able to fail over a decoration.
#
#  Keeping the stream's own encoding and only relaxing the error handler is
#  the quiet fix: ASCII stays perfect, a UTF-8 terminal still shows the
#  dolphins, and a legacy console degrades to "?" instead of a traceback.
# ─────────────────────────────────────────────────────────────────────────
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError):   # ancient Python, or a piped stream
        pass


def _console_speaks_unicode() -> bool:
    """Can this console actually render a dolphin, or will it print '?'."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "🐬—✅⚠️🚀🗺️".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# `errors="replace"` above stops the crash. This stops the MESS: rather than
# littering a legacy console with "?" and "�", we pick symbols it can
# actually print. Modern terminals get the dolphins; cp1252 gets clean ASCII.
_UNI = _console_speaks_unicode()
DOLPHIN = "🐬" if _UNI else "~"
TICK    = "✅" if _UNI else "[OK]"
WARN    = "⚠️ " if _UNI else "[!] "
ROCKET  = "🚀" if _UNI else ""
MAP     = "🗺️ " if _UNI else ""
DASH    = "—" if _UNI else "-"


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Climb from the current folder until we find the top of the project."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / marker).exists():
            return candidate
    raise SystemExit(
        "Could not find the project root — no pyproject.toml here or above.\n"
        "Run this from inside the repository:\n"
        "    uv run python scripts/extract_data.py"
    )


def mb(n: int) -> float:
    return n / 1024**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def free_space(path: Path) -> int:
    return shutil.disk_usage(path).free


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unpack the compressed Lee County taxroll.")
    ap.add_argument("--check", action="store_true",
                    help="report what would be done, change nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the file is already there and valid")
    ap.add_argument("--clean", action="store_true",
                    help="delete the extracted files (archives are kept)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip checksums; faster, but will not catch a damaged file")
    args = ap.parse_args()

    root = find_project_root()
    data = root / "data"
    manifest_path = data / "data_manifest.json"

    print("=" * 74)
    print(f" {DOLPHIN}  Unpacking the Lee County taxroll")
    print("=" * 74)
    print(f"project root : {root}")
    print(f"data folder  : {data}")

    if not manifest_path.exists():
        print(f"\n!! No {manifest_path.name} found in data/.")
        print("   Your clone may be incomplete - try `git pull`, or see")
        print("   data/README.md for where to get the roll directly.")
        return 1

    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    files: dict[str, dict] = manifest["files"]
    print(f"manifest     : {len(files)} files, "
          f"{manifest.get('dataset', 'unknown dataset')}")

    # ---- --clean --------------------------------------------------------
    if args.clean:
        freed = 0
        for name in files:
            target = data / name
            if target.exists():
                freed += target.stat().st_size
                target.unlink()
                print(f"  deleted {name}")
        print(f"\nFreed {mb(freed):,.1f} MB. The .gz archives are untouched -")
        print("re-run this script without --clean to get the data back.")
        return 0

    # ---- plan -----------------------------------------------------------
    todo: list[str] = []
    ok: list[str] = []
    problems: list[str] = []

    for name, info in files.items():
        archive = data / info["archive"]
        target = data / name

        if not archive.exists():
            problems.append(f"{info['archive']} is missing from data/")
            continue

        if target.exists() and not args.force:
            if target.stat().st_size == info["raw_bytes"]:
                ok.append(name)
                continue
            print(f"  {WARN}{name} exists but is the wrong size "
                  f"({mb(target.stat().st_size):,.1f} MB, expected "
                  f"{mb(info['raw_bytes']):,.1f} MB) - will re-extract")
        todo.append(name)

    if problems:
        print("\n!! Cannot continue:")
        for p in problems:
            print(f"   - {p}")
        return 1

    if ok:
        print(f"\nAlready extracted ({len(ok)}): {', '.join(ok)}")
    if not todo:
        print(f"\n{TICK} Everything is already unpacked. Nothing to do.")
        print(f"   Open the notebook in notebooks/ and go. {ROCKET}")
        return 0

    needed = sum(files[n]["raw_bytes"] for n in todo)
    available = free_space(data)
    print(f"\nTo extract ({len(todo)}): {', '.join(todo)}")
    print(f"disk needed  : {mb(needed):,.1f} MB")
    print(f"disk free    : {mb(available):,.1f} MB")

    if available < needed * 1.1:
        print("\n!! Not enough free disk space (want ~10% headroom).")
        return 1

    if args.check:
        print("\n--check given, so nothing was written.")
        return 0

    # ---- extract --------------------------------------------------------
    print()
    t_all = time.perf_counter()
    for i, name in enumerate(todo, 1):
        info = files[name]
        archive = data / info["archive"]
        target = data / name
        tmp = data / (name + ".partial")

        print(f"[{i}/{len(todo)}] {name}  "
              f"({mb(info['archive_bytes']):,.1f} MB -> "
              f"{mb(info['raw_bytes']):,.1f} MB)", flush=True)

        t0 = time.perf_counter()
        try:
            # Write to .partial first, then rename. An interrupted run can
            # then never leave a half-written file that looks complete.
            with gzip.open(archive, "rb") as fi, open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo, length=CHUNK)
        except (OSError, EOFError, gzip.BadGzipFile) as e:
            tmp.unlink(missing_ok=True)
            print(f"        FAILED: {type(e).__name__}: {e}")
            print(f"        The archive may be damaged. Try: git checkout -- "
                  f"data/{info['archive']}")
            return 1
        except KeyboardInterrupt:
            tmp.unlink(missing_ok=True)
            print("\n        interrupted; removed the partial file.")
            return 130

        if tmp.stat().st_size != info["raw_bytes"]:
            print(f"        FAILED: expected {info['raw_bytes']:,} bytes, "
                  f"got {tmp.stat().st_size:,}")
            tmp.unlink(missing_ok=True)
            return 1

        if not args.skip_verify:
            digest = sha256(tmp)
            if digest != info["raw_sha256"]:
                print("        FAILED checksum - the extracted file does not "
                      "match the manifest.")
                print(f"        expected {info['raw_sha256'][:16]}...")
                print(f"        got      {digest[:16]}...")
                tmp.unlink(missing_ok=True)
                return 1
            note = "verified"
        else:
            note = "size ok (checksum skipped)"

        tmp.replace(target)
        print(f"        done in {time.perf_counter()-t0:.1f}s {DASH} {note}")

    print(f"\n{'=' * 74}")
    print(f" {TICK} All set in {time.perf_counter()-t_all:.1f}s. "
          f"{mb(needed):,.1f} MB of Lee County, Florida.")
    print("=" * 74)

    prj = manifest.get("committed_plain") or []
    for name in prj:
        if name and (data / name).exists():
            print(f"\n {MAP} {name} is committed directly {DASH} it is the")
            print("     projection definition the county's shapefile shipped without.")

    print(f"\nNext: open notebooks/ and run the first cell. {ROCKET}")
    print("Thanks to Lee County, Florida for permission to share this data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
