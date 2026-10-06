"""Native synchronous World control with a real-DFT-trained water energy.

Electronic states are spectators in this explicitly geometry-only control.
Same nuclear physics is compared to the independent NVE callback integrator.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from unichem.benchmarks import _reorder
from unichem.checkpoint import load_world, save_world
from unichem.evaluate import Evaluator
from unichem.experiment import AU_TIME_FS
from unichem.factory import make_world
from unichem.grid import Grid
from unichem.models.local_energy import LocalEnergyModel
from unichem.runtime.electrons import LocalKernelRelaxer
from unichem.runtime.engine import SynchronousEngine
from unichem.runtime.potential import integrate_potential


class GeometryControl(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context):
        world = context.world
        return {
            "geometry_control": self.model.energy(
                context.layout.atomic_numbers,
                world.atoms.positions,
                world.atoms.positions.new_tensor(world.grid.lengths),
            )
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/designed_water_geometry_long/model.pt")
    parser.add_argument("--initial", default="outputs/designed_water_control/nve_dt_0.1.npz")
    parser.add_argument("--output", default="outputs/native_water_control")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("At least two steps required")
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["kind"] != "geometry" or payload["feature_version"] != "joint_patch_sp_v3":
        raise ValueError("Current trained geometry control required")
    model = LocalEnergyModel(
        hidden=payload["hidden"], reference_per_atom=payload["reference_per_atom"]
    ).double()
    model.load_state_dict(payload["state_dict"])
    with np.load(args.initial, allow_pickle=False) as data:
        initial = data["positions"][0].copy()
    world = make_world(["O", "H", "H"], initial, grid=Grid((48,) * 3, (0.6,) * 3))
    evaluator = Evaluator(functional=GeometryControl(model))
    dt = 0.1 / AU_TIME_FS
    engine = SynchronousEngine(
        evaluator,
        dt=dt,
        relaxer=LocalKernelRelaxer(max_iterations=0),
        require_electronic_convergence=True,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def callback(positions):
        e, f = model.energy_and_forces(
            torch.tensor([8, 1, 1]), positions, positions.new_tensor([28.8] * 3)
        )
        return e.detach(), f.detach(), {}

    mass = torch.tensor([s.mass for s in world.species], dtype=torch.float64)
    reference = list(
        integrate_potential(
            world.atoms.positions,
            world.atoms.momenta,
            mass,
            callback,
            atom_ids=world.atoms.ids,
            steps=args.steps,
            dt=dt,
        )
    )
    current = world
    positions, energies = [initial], []
    maximum_gap = 0.0
    for index in range(args.steps):
        result = engine.step(current)
        current = result.world
        positions.append(current.atoms.positions.numpy())
        energies.append(result.diagnostics["total_energy_hartree"])
        maximum_gap = max(
            maximum_gap, float(abs(current.atoms.positions - reference[index + 1].positions).max())
        )
        if index + 1 == args.steps // 2:
            save_world(current, out / "restart.npz")
    restart = load_world(out / "restart.npz")
    for _ in range(args.steps - args.steps // 2):
        restart = engine.step(restart).world
    order = [2, 0, 1]
    shuffled = _reorder(world, order)
    for _ in range(args.steps):
        shuffled = engine.step(shuffled).world
    replay = torch.equal(current.atoms.positions, restart.atoms.positions) and torch.equal(
        current.atoms.momenta, restart.atoms.momenta
    )
    order_error = float(abs(current.atoms.positions[order] - shuffled.atoms.positions).max())
    np.savez_compressed(
        out / "trajectory.npz",
        positions=np.array(positions),
        time_fs=np.arange(args.steps + 1) * 0.1,
        total_energy_hartree=np.array(energies),
    )
    report = {
        "checkpoint": args.checkpoint,
        "engine": "native SynchronousEngine",
        "steps": args.steps,
        "duration_fs": args.steps * 0.1,
        "native_vs_callback_max_position_error_bohr": maximum_gap,
        "checkpoint_replay_bitwise": replay,
        "atom_reordering_error_bohr": order_error,
        "nuclear_control_passed": maximum_gap < 1e-11 and order_error < 1e-11 and replay,
        "electronic_model": "Spectator state; energy depends only on shared geometric descriptors",
        "chemistry_emergence_validated": False,
        "scope": "Native short water nuclear control; not equivalence of the quantum predictor and variational electronic dynamics",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
