"""Fetch the checksum-registered paper benchmark repeat evidence through gh."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry():
    registry = json.loads((BENCHMARK_ROOT / "LARGE_ASSET.json").read_text())
    destination = Path(registry["restore_relative_path"])
    if destination.is_absolute() or ".." in destination.parts:
        raise ValueError("The registered destination must be relative to in_silico.")
    if Path(registry["asset_name"]).name != registry["asset_name"]:
        raise ValueError("The registered asset name must be a plain filename.")
    if not re.fullmatch(r"[0-9a-f]{64}", registry["sha256"]):
        raise ValueError("Invalid registered SHA-256.")
    if not isinstance(registry["bytes"], int) or registry["bytes"] <= 0:
        raise ValueError("Invalid registered byte count.")
    return registry


def matches(path, registry):
    return (path.is_file() and path.stat().st_size == registry["bytes"]
            and sha256(path) == registry["sha256"])


def fetch_repeat_evidence(output=None):
    registry = load_registry()
    destination = (Path(output).expanduser().absolute() if output is not None
                   else BENCHMARK_ROOT / registry["restore_relative_path"])
    if destination.exists() or destination.is_symlink():
        if matches(destination, registry):
            print("Already verified:", destination)
            return destination
        raise FileExistsError("Existing destination has different bytes; choose a new --output path.")
    if shutil.which("gh") is None:
        raise RuntimeError("Install GitHub CLI and authenticate with access to the paper repository.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download on the destination filesystem. No incomplete file is published.
    with tempfile.TemporaryDirectory(prefix=".repeat-evidence-", dir=destination.parent) as temporary:
        target = Path(temporary) / registry["asset_name"]
        subprocess.run([
            "gh", "release", "download", registry["tag"],
            "--repo", registry["repository"], "--pattern", registry["asset_name"],
            "--dir", temporary,
        ], check=True)
        if not matches(target, registry):
            raise ValueError("Downloaded asset does not match the registered byte count and SHA-256.")
        # A hard-link publication is atomic and cannot replace a file that
        # another process created during the download. The temporary name is
        # removed on context exit, leaving an ordinary single-link output file.
        try:
            os.link(target, destination)
        except FileExistsError:
            if not matches(destination, registry):
                raise FileExistsError("Destination changed during download; no file was overwritten.")
    print("Downloaded and verified:", destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional destination; defaults to the registered in_silico results path.")
    args = parser.parse_args()
    try:
        fetch_repeat_evidence(args.output)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
