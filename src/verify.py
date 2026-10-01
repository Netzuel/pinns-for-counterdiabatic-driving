"""Audit the public files for private data and verify the bundled data hashes.

    python src/verify.py            # audit and check datasets/checksums.json
    python src/verify.py --update   # rewrite datasets/checksums.json

The audit scans the working tree, not Git history. It reports file locations and
categories without printing matched values. Text, notebooks, SVG, HDF5 strings
and attributes, and the contents of PyTorch checkpoints (zip archives) are
included. A pattern scan reduces risk; it does not prove that no private
information remains.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import h5py
import numpy as np

# Folders skipped when the tree is not a Git checkout (caches, environments,
# build output, and user training runs).
EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "runs",
}
# Categories of private information; each regex runs on raw bytes.
PATTERNS = {
    "private-home-path": re.compile(
        rb"(?:/(?:Users|home|lhome|scratch|lustre)/[A-Za-z0-9._-]+|[A-Za-z]:\\(?:Users|Documents and Settings)\\)"
    ),
    "machine-temp-path": re.compile(
        rb"/(?:private/)?var/folders/[A-Za-z0-9_+-]+|/private/(?:tmp|var)/[A-Za-z0-9._-]+"
        rb"|/Volumes/[A-Za-z0-9._ -]+|file:///[A-Za-z]"
    ),
    "email-address": re.compile(
        rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
    ),
    "personal-hostname": re.compile(rb"(?i)\b[a-z0-9_-]*(?:MacBook|DESKTOP|LAPTOP)[-_][a-z0-9_-]+"),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access-token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}"
        rb"|sk-[A-Za-z0-9_-]{40,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
    ),
    "credential-assignment": re.compile(
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*[\"'][^\"'\s]{12,}[\"']"
    ),
}
ALLOWED: set[tuple[str, bytes]] = set()  # (category, exact match) pairs that are public
CHECKSUMS = Path("datasets/checksums.json")
# Glob patterns, relative to datasets/, of the files whose hashes are recorded.
HASHED = ("hamiltonians.h5", "pretrained/*.pt", "results/*.h5", "results/*.csv", "experiments.json")


def public_files(root: Path):
    """Yield every file that would be published.

    In a Git checkout these are the tracked files plus untracked files that are
    not ignored, exactly what ``git add -A`` would commit. Otherwise the tree
    is walked, skipping the folders in EXCLUDED.
    """
    try:
        listing = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        names = sorted({name for name in listing.decode().split("\0") if name})
        yield from (root / name for name in names if (root / name).is_file())
        return
    except (OSError, subprocess.CalledProcessError):
        pass
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(
            name for name in folders if name not in EXCLUDED and not name.endswith(".egg-info")
        )
        for name in sorted(names):
            if name != ".DS_Store":
                yield Path(directory) / name


def _hdf5_strings(path: Path) -> bytes:
    """Collect dataset names, attributes, and string datasets of an HDF5 file."""
    chunks = []

    def visit(name, item):
        chunks.append(name.encode())
        for key, value in item.attrs.items():
            chunks.extend([str(key).encode(), str(value).encode()])
        if isinstance(item, h5py.Dataset) and item.dtype.kind in "SUO":
            chunks.append(str(np.asarray(item[()])).encode())

    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            chunks.extend([str(key).encode(), str(value).encode()])
        handle.visititems(visit)
    return b"\n".join(chunks)


def audit(root: Path) -> tuple[list[dict], int]:
    """Return privacy findings (file, category, count) and the number of files scanned."""
    findings, count = [], 0
    for path in public_files(root):
        count += 1
        relative = path.relative_to(root).as_posix()
        payloads = [(relative, path.read_bytes())]
        if path.suffix in {".h5", ".hdf5"}:
            payloads.append((relative + "#hdf5", _hdf5_strings(path)))
        if path.suffix == ".pt" and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                payloads += [(f"{relative}#{n}", archive.read(n)) for n in archive.namelist()]
        for location, data in payloads:
            for label, pattern in PATTERNS.items():
                hits = {m.group(0) for m in pattern.finditer(data)} - {
                    value for kind, value in ALLOWED if kind == label
                }
                if hits:
                    findings.append({"file": location, "category": label, "count": len(hits)})
    return findings, count


def checksums(root: Path) -> dict[str, str]:
    """SHA-256 of every bundled data file, keyed by its path inside datasets/."""
    base = root / "datasets"
    files = sorted({p for pattern in HASHED for p in base.glob(pattern)})
    return {
        p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }


def main() -> int:
    """Print a JSON report; exit 1 on any finding or checksum mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--update", action="store_true", help="rewrite datasets/checksums.json")
    args = parser.parse_args()
    root = args.root.resolve()
    current = checksums(root)
    if args.update:
        (root / CHECKSUMS).write_text(json.dumps(current, indent=2) + "\n")
    expected = json.loads((root / CHECKSUMS).read_text())
    mismatched = sorted(
        k for k in expected.keys() | current.keys() if expected.get(k) != current.get(k)
    )
    findings, scanned = audit(root)
    report = {
        "scanned_files": scanned,
        "privacy_findings": findings,
        "hashed_files": len(current),
        "checksum_mismatches": mismatched,
    }
    print(json.dumps(report, indent=2))
    return 1 if findings or mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
