"""
preflight.py - run this before you commit.

    uv run python scripts/preflight.py

Three checks, each of which has already caught a real problem in this
repository. All of them are fast and none of them touch the network.

    1. SECRETS      no API key, token or private key in anything git can stage,
                    including a notebook's saved outputs
    2. FILE SIZES   nothing over GitHub's 100 MB hard limit
    3. CONSOLE      every print() in every script survives a Windows console

Exit code is 0 if everything passes and 1 if anything fails, so this works as
a pre-commit hook or a CI step.

    uv run python scripts/preflight.py --only secrets
    uv run python scripts/preflight.py --quiet


Why these three, specifically
-----------------------------
Each one is here because of something that actually happened:

  SECRETS     An earlier version of the app printed the user's API key to the
              terminal, and shipped a hardcoded username and password in its
              source. Neither is a thing you notice by reading 3,900 lines.

  FILE SIZES  The raw Lee County taxroll is 1.67 GB, with three files over
              GitHub's 100 MB hard limit. A single `git add .` would have
              staged all of it and had the push rejected - after which the
              blobs live in local history and need surgery to remove.

  CONSOLE     A dolphin emoji in extract_data.py's first print() raised
              UnicodeEncodeError on a Windows cp1252 console and killed the
              script before it extracted a single byte. It worked perfectly on
              the author's machine, which is UTF-8. That is the whole problem:
              the development box is not the reader's box.

Standard library only, deliberately - a check you cannot run is not a check.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

# A Windows console using the legacy code page cannot print these, and a
# print() that raises takes the whole script down. See the CONSOLE note above.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

GITHUB_HARD_LIMIT = 100 * 1024**2
GITHUB_WARN_LIMIT = 50 * 1024**2

SECRET_PATTERNS = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI API key",    re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("Google API key",    re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("HuggingFace token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("AWS access key",    re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token",      re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("private key",       re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    ("bearer token",      re.compile(r"Bearer\s+[A-Za-z0-9._\-]{25,}")),
]


def find_project_root(marker: str = "pyproject.toml") -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / marker).exists():
            return candidate
    raise SystemExit("Could not find the project root - run this inside the repo.")


def mb(n: int) -> float:
    return n / 1024**2


def stageable_files(root: Path) -> list[Path]:
    """Everything git would include: tracked files plus un-ignored new ones."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, check=True).stdout
        return [root / line for line in out.splitlines() if (root / line).is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ! git not available; falling back to a filesystem walk")
        skip = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}
        return [p for p in root.rglob("*")
                if p.is_file() and not skip & set(p.parts)]


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 1 - secrets
# ─────────────────────────────────────────────────────────────────────────
def check_secrets(root: Path, files: list[Path], verbose: bool) -> list[str]:
    problems: list[str] = []
    scanned = 0

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1

        # Scan the source, and for a notebook ALSO scan its saved outputs -
        # a printed key hides there, not in the code.
        blobs = [("source", text)]
        if path.suffix == ".ipynb":
            try:
                notebook = json.loads(text)
                chunks = []
                for i, cell in enumerate(notebook.get("cells", [])):
                    for output in cell.get("outputs", []):
                        for key in ("text", "data"):
                            value = output.get(key)
                            if isinstance(value, list):
                                chunks.append(f"[cell {i}] " + "".join(map(str, value)))
                            elif isinstance(value, dict):
                                chunks.append("[cell %d] %s" % (i, json.dumps(
                                    {k: v for k, v in value.items() if "image" not in k})))
                            elif value:
                                chunks.append(f"[cell {i}] {value}")
                blobs.append(("outputs", "\n".join(chunks)))
            except (ValueError, TypeError):
                problems.append(f"{path.relative_to(root)}: could not parse as a notebook")

        for where, blob in blobs:
            for label, pattern in SECRET_PATTERNS:
                for match in pattern.finditer(blob):
                    context = blob[max(0, match.start() - 50):match.end() + 15]
                    problems.append(
                        f"{path.relative_to(root)} ({where}): {label} -> "
                        f"...{context.replace(chr(10), ' ')}...")

    if verbose:
        print(f"  scanned {scanned} files (notebook outputs included)")
    return problems


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 2 - file sizes
# ─────────────────────────────────────────────────────────────────────────
def check_file_sizes(root: Path, files: list[Path], verbose: bool) -> list[str]:
    problems: list[str] = []
    big = sorted(((p.stat().st_size, p) for p in files
                  if p.stat().st_size >= GITHUB_WARN_LIMIT), reverse=True)

    for size, path in big:
        rel = path.relative_to(root)
        if size >= GITHUB_HARD_LIMIT:
            problems.append(
                f"{rel}: {mb(size):,.1f} MB exceeds GitHub's 100 MB HARD limit - "
                f"the push will be rejected")
        elif verbose:
            headroom = (GITHUB_HARD_LIMIT - size) / GITHUB_HARD_LIMIT * 100
            print(f"  {rel}: {mb(size):,.1f} MB - over the 50 MB advisory, "
                  f"{headroom:.0f}% headroom left before the hard limit")

    if verbose and not big:
        print("  no file is over 50 MB")
    elif verbose:
        print(f"  {len(big)} file(s) over the 50 MB advisory, none over 100 MB"
              if not problems else "")
    return problems


# ─────────────────────────────────────────────────────────────────────────
#  CHECK 3 - console safety
# ─────────────────────────────────────────────────────────────────────────
def check_console_safety(root: Path, files: list[Path], verbose: bool) -> list[str]:
    """Every string literal reachable from a print() must be pure ASCII.

    Emoji are welcome in Markdown, notebooks and Streamlit - all UTF-8 by
    definition. They are not safe in terminal output, where a legacy Windows
    code page turns one into a fatal UnicodeEncodeError. Scripts that want
    decoration should pick it at runtime based on what the console can encode.
    """
    problems: list[str] = []
    checked = 0

    for path in sorted(f for f in files if f.suffix == ".py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError) as e:
            problems.append(f"{path.relative_to(root)}: could not parse - {e}")
            continue
        checked += 1

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                continue
            for arg in node.args:
                # ast.walk reaches the literal pieces of an f-string too.
                for sub in ast.walk(arg):
                    if (isinstance(sub, ast.Constant)
                            and isinstance(sub.value, str)
                            and not sub.value.isascii()):
                        problems.append(
                            f"{path.relative_to(root)}:{node.lineno}: non-ASCII in "
                            f"print() -> {sub.value.strip()[:45]!r}")

    if verbose:
        print(f"  checked every print() in {checked} Python files")
    return problems


CHECKS = {
    "secrets":  ("no API keys, tokens or private keys anywhere git can see", check_secrets),
    "sizes":    ("nothing over GitHub's 100 MB hard limit", check_file_sizes),
    "console":  ("every print() survives a Windows console", check_console_safety),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-commit checks for this repository.")
    ap.add_argument("--only", choices=sorted(CHECKS), help="run a single check")
    ap.add_argument("--quiet", action="store_true", help="only report failures")
    args = ap.parse_args()

    root = find_project_root()
    files = stageable_files(root)
    verbose = not args.quiet

    print("=" * 74)
    print(" Preflight - checks to run before committing")
    print("=" * 74)
    print(f"repo  : {root}")
    print(f"files : {len(files)} that git would include")

    selected = [args.only] if args.only else list(CHECKS)
    failures = 0

    for name in selected:
        description, fn = CHECKS[name]
        print(f"\n--- {name.upper()}: {description}")
        problems = fn(root, files, verbose)
        if problems:
            failures += len(problems)
            for p in problems:
                print(f"  FAIL  {p}")
        else:
            print(f"  PASS")

    print("\n" + "=" * 74)
    if failures:
        print(f" {failures} problem(s) found. Do not commit until these are fixed.")
    else:
        print(" All checks passed. Safe to commit.")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
