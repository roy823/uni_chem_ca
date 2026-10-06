"""Bounded CA execution on water-coordinate inputs; not chemical validation.

The test field is an untrained numerical fixture. All contact pairs are evolved
in one shared system. Field preparation and neighbour construction are timed
separately from CA updates; no global HF solver is used by this experiment.
"""

import argparse
import json
import math
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from unichem import make_world
from unichem.checkpoint import save_cells
from unichem.grid import Grid
from unichem.runtime.local_cells import CellSnapshot, FieldSnapshot, LocalCellEngine
from unichem.spatial.neighbors import CellList
from unichem.units import BOHR_PER_ANGSTROM


def dimensions(count):
    candidates = []
    for a in range(1, round(count ** (1 / 3)) + 2):
        if count % a:
            continue
        for b in range(a, math.isqrt(count // a) + 1):
            if count % (a * b) == 0:
                c = count // (a * b)
                candidates.append((c / a, a, b, c))
    _, a, b, c = min(candidates)
    return a, b, c


def state_for(count, device):
    shape = dimensions(count)
    spacing = 3.2 * BOHR_PER_ANGSTROM
    origin = (np.indices(shape).reshape(3, -1).T + 0.5) * spacing
    length = 0.9894093164864126 * BOHR_PER_ANGSTROM
    angle = np.deg2rad(100.02687873278352)
    molecule = np.array(
        [[0, 0, 0], [length, 0, 0], [length * np.cos(angle), length * np.sin(angle), 0]]
    )
    positions = (origin[:, None, :] + molecule).reshape(-1, 3)
    atoms = make_world(["O", "H", "H"] * count, positions).atoms
    atoms = replace(
        atoms, **{f.name: getattr(atoms, f.name).to(device) for f in dataclass_fields(atoms)}
    )
    channels = torch.tensor([5, 1, 1] * count, device=device)
    occupation = atoms.numbers.to(torch.float64) / (2 * channels)
    active = torch.arange(5, device=device)[None] < channels[:, None]
    diagonal = occupation[:, None] * active
    density = torch.diag_embed(diagonal).to(torch.complex128)[:, None].repeat(1, 2, 1, 1)
    state = CellSnapshot(atoms, density, channels)
    grid = Grid((4, 4, 4), tuple(n * spacing / 4 for n in shape))
    return state, grid


def fixture(state, pairs):
    n, _, width, _ = state.density.shape
    channels = torch.arange(width, device=state.density.device, dtype=torch.float64)
    diagonal = 0.02 * state.atoms.numbers[:, None] + 0.015 * channels[None]
    onsite = torch.diag_embed(diagonal)
    active = channels[None] < state.channels[:, None]
    hopping = 0.05 * torch.exp(-pairs.distance / 5)
    hopping = hopping[:, None, None] * (
        active[pairs.first, :, None] & active[pairs.second, None, :]
    )
    return FieldSnapshot(
        torch.zeros(n, 3, device=state.density.device, dtype=torch.float64), onsite, hopping
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waters", type=int, nargs="+", default=[10, 100, 1000, 10000])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="outputs/local_ca_capacity")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("Requested CUDA device is unavailable")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for count in args.waters:
        state, grid = state_for(count, device)
        state.validate()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        pairs = CellList(3.1 * BOHR_PER_ANGSTROM)(grid, state.atoms.positions)
        synchronize(device)
        neighbor_seconds = perf_counter() - start
        start = perf_counter()
        field = fixture(state, pairs)
        synchronize(device)
        field_seconds = perf_counter() - start
        engine = LocalCellEngine(dt=0.1, max_neighbors=32, max_channels=5)
        initial_number = state.populations().sum(0)
        state = engine.step(state, pairs, field)
        synchronize(device)
        samples = []
        for _ in range(args.steps):
            start = perf_counter()
            state = engine.step(state, pairs, field)
            synchronize(device)
            samples.append(perf_counter() - start)
        eig = torch.linalg.eigvalsh(state.density)
        row = {
            "water_coordinate_inputs": count,
            "cells": state.n_cells,
            "contact_pairs": len(pairs.first),
            "maximum_degree": int(
                torch.bincount(
                    torch.cat((pairs.first, pairs.second)), minlength=state.n_cells
                ).max()
            ),
            "neighbor_seconds": neighbor_seconds,
            "fixture_field_seconds": field_seconds,
            "ca_step_seconds": samples,
            "median_ca_step_seconds": float(np.median(samples)),
            "microseconds_per_cell_step": float(np.median(samples)) * 1e6 / state.n_cells,
            "max_spin_number_error": float(
                (state.populations().sum(0) - initial_number).abs().max()
            ),
            "initial_spin_numbers": initial_number.tolist(),
            "initial_number_error": float(abs(initial_number.sum() - state.atoms.numbers.sum())),
            "minimum_local_occupation": float(eig.min()),
            "maximum_local_occupation": float(eig.max()),
            "local_density_bytes": state.density.numel() * state.density.element_size(),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        rows.append(row)
        save_cells(state, root / f"cells_{device.type}_{count}.npz")
        np.savez_compressed(
            root / f"state_{device.type}_{count}.npz",
            positions_bohr=state.atoms.positions.cpu().numpy(),
            numbers=state.atoms.numbers.cpu().numpy(),
            local_populations=state.populations().cpu().numpy(),
        )
        report = {
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "threads": args.threads,
            "state_channels": 5,
            "max_neighbors": 32,
            "steps_after_warmup": args.steps,
            "field_source": "Untrained nonchemical numerical fixture; zero nuclear forces",
            "quantum_closure": "Local reduced density, product-state pair closure and convex channel mixture; not global HF",
            "chemistry_validated": False,
            "rows": rows,
        }
        (root / f"report_{device.type}.json").write_text(
            json.dumps(report, indent=2), encoding="utf8"
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
