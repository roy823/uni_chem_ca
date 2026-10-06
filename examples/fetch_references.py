"""Download pinned, checksum-verified public reference files into this project."""

import hashlib
import json
import tarfile
from pathlib import Path

import requests


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/dc_r2scan_manifest.json").read_text(encoding="utf8"))
    for item in manifest["files"]:
        path = root / Path(item["path"])
        if not path.resolve().is_relative_to(root):
            raise ValueError("Download path outside project")
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            continue
        response = requests.get(item["url"], timeout=120)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != item["sha256"]:
            raise ValueError("Reference checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    small = json.loads((root / "data/h2o13_manifest.json").read_text())
    archive = root / "data/raw/h2o13.tar.bz2"
    if not archive.exists():
        response = requests.get(small["url"], timeout=120)
        response.raise_for_status()
        archive.write_bytes(response.content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != small["sha256"]:
        raise ValueError("H2O-13 checksum mismatch")
    with tarfile.open(archive) as source:
        member = "h2o-13/watergrid_60_HOH_180__0.7_rOH_1.8_vario_PBE0_AV5Z_delta_PS.xyz"
        path = root / "data/raw/h2o13/PBE0.xyz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.extractfile(member).read())
    print("Pinned public reference files verified; run unichem prepare-reference")


if __name__ == "__main__":
    main()
