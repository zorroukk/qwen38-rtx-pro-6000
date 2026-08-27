#!/usr/bin/env python3
"""Reproduce the sidecar-aware overlay from the sealed online-W4 base."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


EXPECTED_BASE_SHA256 = "95bb2059669f9a66abe1be8e271037eeaa11ea534ace808f1c171ca78fee101f"
EXPECTED_PATCH_SHA256 = "938c43e9d283883ffc285c4e5e3a03967ebc0167480767606131fe361f1d8644"
EXPECTED_OUTPUT_SHA256 = "b54de0a07a16a7a3070aabead3c53b80f108d89528c30763ca8e032f330d97ac"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    patch = args.patch.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    require_digest(base, EXPECTED_BASE_SHA256, "base overlay")
    require_digest(patch, EXPECTED_PATCH_SHA256, "sidecar patch")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        subprocess.run(
            [
                "patch",
                "--batch",
                "--fuzz=0",
                "--posix",
                "--output",
                str(temporary),
                str(base),
                str(patch),
            ],
            check=True,
        )
        require_digest(temporary, EXPECTED_OUTPUT_SHA256, "generated overlay")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"SIDECAR_OVERLAY_REPRODUCED {EXPECTED_OUTPUT_SHA256} {output}")


if __name__ == "__main__":
    main()
