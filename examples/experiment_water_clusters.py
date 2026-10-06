"""Internal water-cluster performance and reaction observations, without learning.

No reference solver or reaction label enters the Hamiltonian. Prepared ionic and
shared-proton initial states are controls, not evidence of spontaneous water
autoionization. All chemical classifications below are post-processing only.
"""

import argparse
import json
import os
import platform
import resource
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from pyscf import lib

from unichem import Evaluator, SynchronousEngine, make_world
from unichem.checkpoint import save_world
from unichem.errors import StateError
from unichem.quantum.gaussian import GaussianIntegrals
from unichem.quantum.observables import atomic_populations, bond_indices
from unichem.runtime.scf import SCFSolver
from unichem.units import ATOMIC_TIME_PER_FS, BOHR_PER_ANGSTROM

EV_PER_HARTREE = 27.211386245988
KB_HARTREE_PER_K = 3.166811563e-6


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def environment(threads):
    torch.set_num_threads(threads)
    lib.num_threads(threads)
    return {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "threads": threads,
        "torch": torch.__version__,
        "method": "all-electron HF",
        "basis": "sto-3g",
        "learned_energy": False,
        "reaction_rules": False,
    }


def water_chain(count, *, spacing=2.7):
    positions = []
    angle = np.deg2rad(100.02687873278352)
    length = 0.9894093164864126
    for i in range(count):
        origin = np.array([spacing * i, 0.0, 0.0])
        positions.extend(
            [
                origin,
                origin + [length, 0, 0],
                origin + length * np.array([np.cos(angle), (-1) ** i * np.sin(angle), 0]),
            ]
        )
    return make_world(["O", "H", "H"] * count, np.array(positions) * BOHR_PER_ANGSTROM)


def thermal_momenta(world, temperature, seed):
    rng = np.random.default_rng(seed)
    masses = world.atoms.masses.numpy()
    positions = world.atoms.positions.numpy()
    center = np.sum(masses[:, None] * positions, axis=0) / masses.sum()
    relative = positions - center
    momentum = rng.normal(size=positions.shape) * np.sqrt(
        masses[:, None] * KB_HARTREE_PER_K * temperature
    )
    momentum -= masses[:, None] * momentum.sum(0) / masses.sum()
    inertia = sum(m * (np.dot(r, r) * np.eye(3) - np.outer(r, r)) for m, r in zip(masses, relative))
    omega = np.linalg.solve(inertia, np.cross(relative, momentum).sum(0))
    momentum -= masses[:, None] * np.cross(omega, relative)
    kinetic = np.sum(momentum**2 / (2 * masses[:, None]))
    target = 0.5 * (3 * len(masses) - 6) * KB_HARTREE_PER_K * temperature
    momentum *= np.sqrt(target / kinetic)
    return replace(world, atoms=replace(world.atoms, momenta=torch.tensor(momentum)))


def scenario_world(name, seed, count):
    if name == "oriented_ion_pair":
        if count != 2:
            raise ValueError("The oriented ion-pair control contains exactly two waters")
        world = water_chain(2, spacing=3.2)
        r = world.atoms.positions.clone()
        r[1, 0] = (3.2 - 0.98) * BOHR_PER_ANGSTROM
        # Keep the acceptor's two terminal OH channels away from the incoming H.
        # This is only a prepared initial geometry, never an electronic constraint.
        x, y = 0.39, np.sin(np.deg2rad(50.01343936639176))
        z = np.sqrt(1 - x * x - y * y)
        for i, sign in ((4, 1), (5, -1)):
            r[i] = r[3] + torch.tensor([x, sign * y, z]) * (0.9894093164864126 * BOHR_PER_ANGSTROM)
        return world.with_positions(r)
    if name in ("neutral_300k", "hot_10000k", "prepared_ion_pair"):
        world = water_chain(count, spacing=2.6)
        if name == "prepared_ion_pair":
            if count != 2:
                raise ValueError("The prepared ion-pair control contains exactly two waters")
            r = world.atoms.positions.clone()
            r[1, 0] = 1.62 * BOHR_PER_ANGSTROM
            world = world.with_positions(r)
        else:
            world = thermal_momenta(world, 300 if name == "neutral_300k" else 10000, seed)
        return world
    if name == "shared_proton":
        # H5O2+ is a prepared reactive intermediate. No connectivity is supplied.
        positions = [
            [-1.2, 0, 0],
            [-1.76, 0.79, 0],
            [-1.76, -0.79, 0],
            [1.2, 0, 0],
            [1.76, 0.79, 0],
            [1.76, -0.79, 0],
            [0.07, 0, 0],
        ]
        return make_world(
            ["O", "H", "H", "O", "H", "H", "H"], np.array(positions) * BOHR_PER_ANGSTROM, charge=1
        )
    raise ValueError(name)


def observe(world, evaluation):
    """Distances, Mulliken fragment charges and Mayer indices: read-only evidence."""
    z = world.atoms.numbers.numpy()
    r = world.atoms.positions.numpy() / BOHR_PER_ANGSTROM
    oxygen, hydrogen = np.where(z == 8)[0], np.where(z == 1)[0]
    distances = np.linalg.norm(r[hydrogen, None] - r[oxygen][None], axis=-1)
    owners = oxygen[distances.argmin(1)]
    nearest = distances.min(1)
    ambiguous = np.sort(distances, axis=1)[:, 1] - nearest < 0.15
    counts = np.array([(owners == o).sum() for o in oxygen])
    charge = (
        z
        - atomic_populations(evaluation.electronic.densities, evaluation.integrals.overlap).numpy()
    )
    fragment_charges = [float(charge[o] + charge[hydrogen[owners == o]].sum()) for o in oxygen]
    bi, bj, bv = bond_indices(evaluation.electronic.densities, evaluation.integrals.overlap)
    lookup = {(int(i), int(j)): float(v) for i, j, v in zip(bi, bj, bv)}
    bonds = np.array(
        [[lookup.get(tuple(sorted((int(h), int(o)))), 0.0) for o in oxygen] for h in hydrogen]
    )
    compact = bool(nearest.max() < 1.35 and not ambiguous.any())
    geometric_pair = bool((counts == 1).any() and (counts == 3).any())
    charge_pair = bool(
        geometric_pair
        and compact
        and all(q > 0.25 for q, n in zip(fragment_charges, counts) if n == 3)
        and all(q < -0.25 for q, n in zip(fragment_charges, counts) if n == 1)
    )
    return {
        "time_fs": world.time / ATOMIC_TIME_PER_FS,
        "oxygen_ids": oxygen.tolist(),
        "hydrogen_ids": hydrogen.tolist(),
        "owners": owners.tolist(),
        "coordination": counts.tolist(),
        "ambiguous": ambiguous.tolist(),
        "oh_distances_angstrom": distances.tolist(),
        "oh_mayer_indices": bonds.tolist(),
        "atomic_mulliken_charges": charge.tolist(),
        "fragment_mulliken_charges": fragment_charges,
        "geometric_ionic_pair": geometric_pair,
        "charge_supported_ionic_pair": charge_pair,
        "compact_neutral_coordination": bool(compact and (counts == 2).all()),
        "scf_iterations": evaluation.electronic.iterations,
        "electron_error": max(evaluation.electronic.electron_errors),
        "purity_error": max(evaluation.electronic.purity_errors),
    }


def episodes(frames, key, dt_fs):
    runs = []
    begin = None
    for i in range(len(frames) + 1):
        active = i < len(frames) and frames[i][key]
        if active and begin is None:
            begin = i
        if not active and begin is not None:
            runs.append(
                {
                    "start_fs": frames[begin]["time_fs"],
                    "end_fs": frames[i - 1]["time_fs"],
                    "sampled_duration_fs": (i - begin - 1) * dt_fs,
                    "includes_initial_frame": begin == 0,
                }
            )
            begin = None
    return runs


def benchmark(args, root):
    metadata = environment(args.threads)
    rows = []
    for count in args.waters:
        row = {"waters": count, "atoms": 3 * count}
        world = water_chain(count)
        start = perf_counter()
        try:
            before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            provider = GaussianIntegrals(world.atoms.numbers.numpy(), world.atoms.positions.numpy())
            built = perf_counter()
            solver = SCFSolver()
            solution = solver.solve(provider.operators(), world.spin_counts)
            solved = perf_counter()
            force = provider.forces(solution)
            finished = perf_counter()
            row.update(
                {
                    "status": "completed",
                    "basis_functions": int(provider.mol.nao),
                    "integral_setup_seconds": built - start,
                    "cold_scf_seconds": solved - built,
                    "force_seconds": finished - solved,
                    "cold_energy_force_seconds": finished - start,
                    "scf_iterations": solution.iterations,
                    "density_blocks_per_spin": [len(p.row) for p in solution.densities],
                    "density_block_fill_fraction": len(solution.densities[0].row)
                    / world.n_atoms**2,
                    "density_value_bytes": sum(
                        p.values.numel() * p.values.element_size() for p in solution.densities
                    ),
                    "integral_cache_bytes": provider._cache_bytes,
                    "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024,
                    "peak_rss_increase_mib": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before_rss
                    )
                    / 1024,
                    "energy_hartree": solution.energy,
                    "max_force_au": float(force.abs().max()),
                    "electron_error": max(solution.electron_errors),
                    "purity_error": max(solution.purity_errors),
                }
            )
            # Full native steps include both endpoint SCFs and forces, as in production.
            del provider
            state = replace(world, densities=solution.densities)
            engine = SynchronousEngine(dt=args.dt_fs * ATOMIC_TIME_PER_FS)
            step_times = []
            for _ in range(args.steps):
                start_step = perf_counter()
                value = engine.step(state)
                step_times.append(perf_counter() - start_step)
                state = value.world
            row["native_step_seconds"] = step_times
            row["median_native_step_seconds"] = float(np.median(step_times))
            row["projected_wall_seconds_per_ps"] = float(np.median(step_times)) * 1000 / args.dt_fs
        except (StateError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds_before_failure": perf_counter() - start,
                }
            )
        rows.append(row)
        write_json(
            root / f"benchmark_threads_{args.threads}.json",
            {
                "environment": metadata,
                "rows": rows,
                "memory_scope": "Process peak RSS includes Python, Torch and PySCF; cache and density bytes are separate.",
                "geometry": "Open hydrogen-bond chain, 2.7 Angstrom O-O; not optimized or periodic liquid.",
                "dt_fs": args.dt_fs,
                "trajectory_steps_per_case": args.steps,
            },
        )
        print(json.dumps(row), flush=True)


def scan(args, root):
    metadata = environment(args.threads)
    world = water_chain(2, spacing=2.6)
    evaluator = Evaluator()
    rows = []
    for x in (0.99, 1.10, 1.20, 1.30, 1.40, 1.50, 1.62):
        positions = world.atoms.positions.clone()
        positions[1, 0] = x * BOHR_PER_ANGSTROM
        state = world.with_positions(positions)
        start = perf_counter()
        try:
            value = evaluator(state)
            row = {
                "proton_x_angstrom": x,
                "status": "completed",
                "energy_hartree": value.energy,
                "proton_force_x_au": float(value.forces[1, 0]),
                **observe(state, value),
            }
        except (StateError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row = {
                "proton_x_angstrom": x,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        row["seconds"] = perf_counter() - start
        rows.append(row)
        print(json.dumps(row), flush=True)
        write_json(
            root / "proton_scan.json",
            {
                "environment": metadata,
                "rows": rows,
                "scope": "Fixed-nuclei diagnostic scan, not a reaction trajectory or free-energy barrier.",
            },
        )


def trajectory(args, root):
    metadata = environment(args.threads)
    state = scenario_world(args.scenario, args.seed, args.count)
    evaluator = Evaluator()
    engine = SynchronousEngine(evaluator, dt=args.dt_fs * ATOMIC_TIME_PER_FS)
    frames, positions, momenta, energy_errors, times = [], [], [], [], []
    status, error = "completed", None
    start = perf_counter()
    steps = round(args.duration_fs / args.dt_fs)
    if not np.isclose(steps * args.dt_fs, args.duration_fs):
        raise ValueError("Duration must be an integer number of time steps")
    tag = f"{args.scenario}_n_{args.count}_seed_{args.seed}_dt_{args.dt_fs}"
    path = root / tag
    path.mkdir(parents=True, exist_ok=True)
    try:
        evaluation = evaluator(state)
        state = replace(state, densities=evaluation.electronic.densities)
        kinetic = float((state.atoms.momenta.square() / (2 * state.atoms.masses[:, None])).sum())
        initial_energy = evaluation.energy + kinetic
        for step in range(steps + 1):
            frames.append(observe(state, evaluation))
            positions.append(state.atoms.positions.numpy())
            momenta.append(state.atoms.momenta.numpy())
            kinetic = float(
                (state.atoms.momenta.square() / (2 * state.atoms.masses[:, None])).sum()
            )
            energy_errors.append((evaluation.energy + kinetic - initial_energy) * EV_PER_HARTREE)
            if step % 20 == 0:
                write_json(
                    path / "progress.json",
                    {
                        "steps_completed": step,
                        "time_fs": state.time / ATOMIC_TIME_PER_FS,
                        "wall_seconds": perf_counter() - start,
                        "latest": frames[-1],
                    },
                )
                print(
                    json.dumps(
                        {
                            "scenario": args.scenario,
                            "seed": args.seed,
                            "step": step,
                            "time_fs": frames[-1]["time_fs"],
                            "coordination": frames[-1]["coordination"],
                            "fragment_charges": frames[-1]["fragment_mulliken_charges"],
                            "wall_seconds": perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
            if step == steps:
                break
            begin = perf_counter()
            value = engine.step(state)
            times.append(perf_counter() - begin)
            state, evaluation = value.world, value.evaluation
    except (StateError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    np.savez_compressed(
        path / "trajectory.npz",
        positions_bohr=np.array(positions),
        momenta_au=np.array(momenta),
        time_fs=np.array([f["time_fs"] for f in frames]),
        energy_balance_ev=np.array(energy_errors),
        numbers=state.atoms.numbers.numpy(),
    )
    save_world(state, path / "final_world.npz")
    ion_runs = episodes(frames, "charge_supported_ionic_pair", args.dt_fs)
    neutral_runs = episodes(frames, "compact_neutral_coordination", args.dt_fs)
    owner_changes = []
    if frames:
        last = [None] * len(frames[0]["owners"])
        for frame in frames:
            for j, (owner, ambiguous) in enumerate(zip(frame["owners"], frame["ambiguous"])):
                if ambiguous or min(frame["oh_distances_angstrom"][j]) > 1.35:
                    continue
                if last[j] is not None and last[j] != owner:
                    owner_changes.append(
                        {
                            "time_fs": frame["time_fs"],
                            "hydrogen_id": frame["hydrogen_ids"][j],
                            "from_oxygen": last[j],
                            "to_oxygen": owner,
                        }
                    )
                last[j] = owner
    report = {
        "environment": metadata,
        "scenario": args.scenario,
        "water_oxygen_count": int((state.atoms.numbers == 8).sum()),
        "seed": args.seed,
        "status": status,
        "error": error,
        "dt_fs": args.dt_fs,
        "requested_duration_fs": args.duration_fs,
        "steps_completed": state.step,
        "actual_duration_fs": state.time / ATOMIC_TIME_PER_FS,
        "wall_seconds": perf_counter() - start,
        "median_native_step_seconds": float(np.median(times)) if times else None,
        "max_energy_balance_error_ev": max(map(abs, energy_errors)) if energy_errors else None,
        "max_electron_error": max((f["electron_error"] for f in frames), default=None),
        "max_purity_error": max((f["purity_error"] for f in frames), default=None),
        "nonambiguous_compact_owner_changes": owner_changes,
        "charge_supported_ionic_episodes": ion_runs,
        "neutral_episodes": neutral_runs,
        "persistent_ionic_episodes_2fs": [r for r in ion_runs if r["sampled_duration_fs"] >= 2],
        "observation_scope": "Nearest-O coordination, compactness, Mayer bonds and Mulliken charges are post-hoc. Prepared initial ions/shared protons are controls, not spontaneous neutral-water ionization. Classical nuclei, HF/STO-3G; no bulk kinetics or free energy claim.",
        "frames": frames,
    }
    write_json(path / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "scan", "trajectory"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--waters", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--duration-fs", type=float, default=10)
    parser.add_argument(
        "--scenario",
        choices=(
            "neutral_300k",
            "hot_10000k",
            "prepared_ion_pair",
            "oriented_ion_pair",
            "shared_proton",
        ),
        default="neutral_300k",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--output", default="outputs/water_clusters_v04")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    {"benchmark": benchmark, "scan": scan, "trajectory": trajectory}[args.mode](args, root)


if __name__ == "__main__":
    main()
