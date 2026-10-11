#!/usr/bin/env python3
"""Export only SHA256SUMS-listed release files; never Git history or local output."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "outputs/scd-anonymous.zip")
args = parser.parse_args()
entries = []
for line in (ROOT / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise SystemExit(f"Unsafe or missing release path: {name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"Release changed: {name}; update the manifest after review")
    entries.append((name, data))
entries.append(("SHA256SUMS", (ROOT / "SHA256SUMS").read_bytes()))
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name, data in entries:
        info = zipfile.ZipInfo("scd/" + name, date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, data)
print(f"Exported {len(entries)} files to {args.output.name}")
