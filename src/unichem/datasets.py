"""Source-grounded H/O reference data; no chemical identities in model inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOHR_ANGSTROM = 0.529177210903
HARTREE_EV = 27.211386245988


@dataclass(frozen=True)
class ReferenceFrame:
    atomic_numbers: np.ndarray
    positions: np.ndarray  # bohr
    box: np.ndarray  # bohr, orthogonal lengths
    energy: float  # Hartree
    forces: np.ndarray  # Hartree/bohr
    source: str
    group: str
    split: str
    index: int
    net_charge: int = 0
    external_field: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    density: np.ndarray | None = None
    density_grid_shape: tuple[int, int, int] | None = None


def load_deepmd(
    directory: str | Path,
    *,
    source: str,
    group: str,
    split: str,
    type_map: tuple[int, ...],
    net_charge=0,
    external_field=(0.0, 0.0, 0.0),
):
    """Strict DeepMD npy reader. Units follow the published DeepMD format.

    The caller must specify the source's type ordering and field metadata.
    Set order is preserved; author training/validation partitions are retained.
    """
    root = Path(directory)
    types = np.loadtxt(root / "type.raw", dtype=int).reshape(-1)
    if types.min() < 0 or types.max() >= len(type_map):
        raise ValueError("Invalid type.raw")
    numbers = np.array(type_map, dtype=np.int64)[types]
    if not np.isin(numbers, [1, 8]).all():
        raise ValueError("Only H/O reference frames accepted")
    offset = 0
    for folder in sorted(root.glob("set.*")):
        positions = np.load(folder / "coord.npy", allow_pickle=False).reshape(-1, len(types), 3)
        forces = np.load(folder / "force.npy", allow_pickle=False).reshape(positions.shape)
        energies = np.load(folder / "energy.npy", allow_pickle=False).reshape(-1)
        cells = np.load(folder / "box.npy", allow_pickle=False).reshape(-1, 3, 3)
        if not len(energies) == len(positions) == len(cells):
            raise ValueError("Frame count mismatch")
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(forces).all()
            or not np.isfinite(energies).all()
        ):
            raise ValueError("Nonfinite reference labels")
        off = cells - np.eye(3) * cells.diagonal(axis1=-2, axis2=-1)[:, None, :]
        if np.abs(off).max() > 1e-8:
            raise ValueError("Triclinic reference unsupported")
        for i in range(len(energies)):
            yield ReferenceFrame(
                numbers.copy(),
                positions[i] / BOHR_ANGSTROM,
                cells[i].diagonal() / BOHR_ANGSTROM,
                float(energies[i] / HARTREE_EV),
                forces[i] * BOHR_ANGSTROM / HARTREE_EV,
                source,
                group,
                split,
                offset + i,
                net_charge,
                external_field,
            )
        offset += len(energies)


def geometry_hash(frame):
    # Same atom order from the same source is sufficient for cross-split exact
    # duplicate detection; no claim of general molecular graph isomorphism.
    p = np.remainder(frame.positions, frame.box)
    return hashlib.sha256(
        frame.atomic_numbers.tobytes() + np.round(p, 7).tobytes() + np.round(frame.box, 7).tobytes()
    ).hexdigest()


def coordination(frame):
    """Read-only geometric coverage diagnostics; never training features."""
    o = np.where(frame.atomic_numbers == 8)[0]
    h = np.where(frame.atomic_numbers == 1)[0]
    delta = frame.positions[h, None, :] - frame.positions[o][None, :, :]
    delta -= frame.box * np.round(delta / frame.box)
    distances = np.linalg.norm(delta, axis=-1) * BOHR_ANGSTROM
    nearest = distances.argmin(1)
    counts = np.bincount(nearest, minlength=len(o))
    sorted_d = np.sort(distances, axis=1)
    return {
        "two_coordinate_oxygen": int((counts == 2).sum()),
        "three_coordinate_oxygen": int((counts == 3).sum()),
        "one_coordinate_oxygen": int((counts == 1).sum()),
        "shared_proton_candidates": int(
            ((sorted_d[:, 1] - sorted_d[:, 0] < 0.2) & (sorted_d[:, 1] < 1.5)).sum()
        ),
        "max_nearest_oh_angstrom": float(distances.min(1).max()),
    }


def build_public_dataset(raw_root="data/raw/dc_r2scan", output="data/ho_reference"):
    root = Path(raw_root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ["training", "validation"]:
        for directory in sorted((root / split).iterdir()):
            if directory.is_dir() and (directory / "type.raw").exists():
                # Folder names alone do not identify field direction or whether
                # labels include field energy. Quarantine until source metadata
                # is independently verified; no fabricated zero-field labels.
                field = None if directory.name.startswith("E") else (0.0, 0.0, 0.0)
                frames.extend(
                    load_deepmd(
                        directory,
                        source="Paesani DC-r2SCAN water autoionization",
                        group=directory.name,
                        split=split,
                        type_map=(8, 1),
                        external_field=field,
                    )
                )
    validation_hashes = {geometry_hash(f) for f in frames if f.split == "validation"}
    excluded = [
        f for f in frames if f.split == "training" and geometry_hash(f) in validation_hashes
    ]
    frames = [
        f for f in frames if not (f.split == "training" and geometry_hash(f) in validation_hashes)
    ]
    groups = {}
    hashes = {}
    duplicates = []
    coverage = {
        "three_coordinate_oxygen": 0,
        "one_coordinate_oxygen": 0,
        "shared_proton_candidates": 0,
        "max_nearest_oh_angstrom": 0.0,
    }
    for frame in frames:
        key = frame.split + "/" + frame.group
        groups[key] = groups.get(key, 0) + 1
        digest = geometry_hash(frame)
        if digest in hashes and hashes[digest] != frame.split:
            duplicates.append((key, frame.index))
        hashes[digest] = frame.split
        c = coordination(frame)
        for k, value in coverage.items():
            coverage[k] = max(value, c[k]) if k.startswith("max") else value + c[k]
    if duplicates:
        raise ValueError(f"Cross-split duplicate reference frames: {duplicates[:5]}")
    report = {
        "source": "https://github.com/paesanilab/Data_Repository/tree/main/DC-r2SCAN_water_autoionization",
        "method": "DC-r2SCAN",
        "input_units": "angstrom, eV, eV/angstrom",
        "internal_units": "bohr, Hartree, Hartree/bohr",
        "type_map": [8, 1],
        "label_kind": "published quantum energy and force; no density/1-RDM labels",
        "author_split_assignments_preserved": True,
        "cross_split_exact_duplicates": len(duplicates),
        "training_frames_quarantined_for_validation_overlap": len(excluded),
        "groups": groups,
        "frames": len(frames),
        "coverage_observer": coverage,
        "coverage_limits": [
            "Isolated H3O+/OH-, small Zundel/Eigen clusters not separately labeled",
            "Geometric coordination is not a quantum species label",
            "External-field frames quarantined pending field direction and label convention",
            "No 1-RDM/electron-density supervision supplied by this dataset",
            "Bulk frames cannot be cropped into independently DFT-labeled clusters",
        ],
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    (out / "quarantined_training_frames.json").write_text(
        json.dumps(
            [
                {"group": f.group, "index": f.index, "geometry_sha256": geometry_hash(f)}
                for f in excluded
            ],
            indent=2,
        ),
        encoding="utf8",
    )
    # Variable-size groups stored independently to avoid padding fake atoms.
    for split_group in groups:
        selected = [f for f in frames if f.split + "/" + f.group == split_group]
        folder = out / split_group
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "frames.npz",
            atomic_numbers=selected[0].atomic_numbers,
            positions=np.stack([f.positions for f in selected]),
            boxes=np.stack([f.box for f in selected]),
            energy=np.array([f.energy for f in selected]),
            forces=np.stack([f.forces for f in selected]),
            metadata=np.asarray(
                json.dumps(
                    {
                        "source": selected[0].source,
                        "group": selected[0].group,
                        "split": selected[0].split,
                        "method": "DC-r2SCAN",
                        "units": "atomic",
                        "net_charge": 0,
                        "external_field": selected[0].external_field,
                    }
                )
            ),
        )
    return report


def load_reference_groups(root="data/ho_reference", *, split="training", max_frames=None):
    frames = []
    for path in sorted((Path(root) / split).glob("*/frames.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if meta["external_field"] is None:
                continue
            count = (
                len(data["energy"]) if max_frames is None else min(max_frames, len(data["energy"]))
            )
            indices = np.linspace(0, len(data["energy"]) - 1, count, dtype=int)
            for i in indices:
                frames.append(
                    ReferenceFrame(
                        data["atomic_numbers"].copy(),
                        data["positions"][i].copy(),
                        data["boxes"][i].copy(),
                        float(data["energy"][i]),
                        data["forces"][i].copy(),
                        meta["source"],
                        meta["group"],
                        meta["split"],
                        int(i),
                        meta["net_charge"],
                        tuple(meta["external_field"]),
                        data["density"][i].copy() if "density" in data else None,
                        tuple(meta["density_grid_shape"]) if "density_grid_shape" in meta else None,
                    )
                )
    return frames


def build_monomer_dataset(path="data/raw/h2o13/PBE0.xyz", output="data/pbe0_monomers"):
    from ase.io import read

    atoms = read(path, index=":")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    partitions = {"training": [], "validation": [], "test": [], "stretched_test": []}
    seen = set()
    for index, a in enumerate(atoms):
        numbers = a.get_atomic_numbers()
        if sorted(numbers.tolist()) != [1, 1, 8]:
            raise ValueError("Unexpected monomer composition")
        # Source geometries are isolated quantum calculations. The lattice in
        # the extxyz is a storage box, not evidence of periodic DFT.
        positions = a.positions / BOHR_ANGSTROM
        digest = hashlib.sha256(np.round(positions, 7).tobytes()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate source monomer geometry")
        seen.add(digest)
        if max(float(a.info["OH1"]), float(a.info["OH2"])) >= 1.5:
            split = "stretched_test"
        else:
            bucket = int(digest[:8], 16) % 10
            split = "training" if bucket < 8 else "validation" if bucket == 8 else "test"
        partitions[split].append(
            (
                index,
                numbers,
                positions,
                float(a.info["PBE0_energy"]) / HARTREE_EV,
                a.arrays["PBE0_force"] * BOHR_ANGSTROM / HARTREE_EV,
            )
        )
    for split, items in partitions.items():
        folder = out / split / "monomer"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "frames.npz",
            atomic_numbers=items[0][1],
            positions=np.stack([x[2] for x in items]),
            boxes=np.tile(np.array([40.0, 40.0, 40.0]), (len(items), 1)),
            energy=np.array([x[3] for x in items]),
            forces=np.stack([x[4] for x in items]),
            source_indices=np.array([x[0] for x in items]),
            metadata=np.asarray(
                json.dumps(
                    {
                        "source": "H2O-13, Bartok et al. PRB 88 054104 (2013)",
                        "method": "PBE0/aug-cc-pV5Z",
                        "boundary": "isolated molecule",
                        "model_boundary": "periodic vacuum approximation; box convergence required",
                        "group": "monomer",
                        "split": split,
                        "units": "atomic",
                        "net_charge": 0,
                        "external_field": [0.0, 0.0, 0.0],
                    }
                )
            ),
        )
    report = {
        "source": "https://qmml.org/Datasets/h2o-13.tar.bz2",
        "method": "PBE0/aug-cc-pV5Z",
        "frames": len(atoms),
        "partitions": {k: len(v) for k, v in partitions.items()},
        "stretched_holdout_min_oh_angstrom": 1.5,
        "no_density_labels": True,
        "input_units": "eV, eV/angstrom, angstrom (QUIP extxyz)",
        "boundary": "isolated reference, periodic-vacuum model must be converged",
        "no_mixed_methods": True,
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def import_generated_references(
    root="data/generated_pbe0", output="data/ho_pbe0", *, split_policy="suite"
):
    if split_policy not in ("suite", "manifest"):
        raise ValueError("Explicit suite or manifest split policy required")
    source = Path(root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    rows = []
    for path in sorted(source.glob("[0-9][0-9][0-9][0-9].npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if not meta["scf_converged"]:
                raise ValueError("Unconverged DFT frame")
            with np.load(path.with_name(path.stem + ".density.npz"), allow_pickle=False) as rho:
                rows.append(
                    (
                        meta,
                        data["numbers"].copy(),
                        data["positions"].copy(),
                        float(data["energy"]),
                        data["forces"].copy(),
                        rho["density"].copy(),
                    )
                )
    if len(rows) != len(json.loads((source / "jobs.json").read_text())):
        raise ValueError("Reference suite incomplete; do not train a partial coverage claim")
    # Hold out one geometry per small family, two Zundel shifts, and the whole
    # second O-O distance in the neutral pair scan. Family names are metadata.
    by_family = {}
    for row in rows:
        by_family.setdefault(row[0]["group"], []).append(row)
    for family, items in by_family.items():
        for index, item in enumerate(items):
            meta = item[0]
            if family == "neutral_pair_scan":
                validation = index >= len(items) // 2
            elif family == "zundel":
                validation = index >= len(items) - 2
            else:
                validation = index == len(items) - 1
            if split_policy == "manifest":
                split = meta["design_split"]
                if split not in ("training", "validation", "test"):
                    raise ValueError("Invalid designed reference split")
            else:
                split = "validation" if validation else "training"
            field = tuple(meta["external_field_au"])
            # Field-specific groups retain a single explicit field vector.
            group = family + ("_" + str(meta["index"]) if any(field) else "")
            groups.setdefault((split, group), []).append(item)
    for (split, group), items in groups.items():
        meta = items[0][0]
        folder = out / split / group
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": "Generated PySCF " + meta["pyscf_version"],
            "method": meta["method"],
            "group": group,
            "split": split,
            "units": "atomic",
            "boundary": "isolated",
            "net_charge": meta["net_charge"],
            "external_field": meta["external_field_au"],
            "density_grid_shape": [48, 48, 48],
            "density_kind": meta["density_kind"],
        }
        np.savez_compressed(
            folder / "frames.npz",
            atomic_numbers=items[0][1],
            positions=np.stack([x[2] for x in items]),
            boxes=np.tile([28.8] * 3, (len(items), 1)),
            energy=np.array([x[3] for x in items]),
            forces=np.stack([x[4] for x in items]),
            density=np.stack([x[5] for x in items]),
            source_indices=np.array([x[0]["index"] for x in items]),
            metadata=np.asarray(json.dumps(metadata)),
        )
    report = {
        "method": "PBE0/aug-cc-pVDZ",
        "units": "atomic",
        "boundary": "isolated",
        "frames": len(rows),
        "split_policy": split_policy,
        "split_counts": {
            s: sum(len(v) for (split, _), v in groups.items() if split == s)
            for s in ("training", "validation", "test")
        },
        "groups": {s + "/" + g: len(v) for (s, g), v in groups.items()},
        "all_scf_converged": True,
        "density_convention": "valence with lowest oxygen 1s MOs removed",
        "max_field_force_fd_error": max(x[0].get("field_force_fd_error", 0) for x in rows),
        "categories": sorted(by_family),
        "density_integrals": [x[0]["density_integral_grid"] for x in rows],
        "status": "small reference suite, not statistically comprehensive reactive training set",
        "source_files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.npz")
        },
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
