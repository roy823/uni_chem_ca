"""Fixed-checkpoint vacuum-grid refinement and subcell translation audit.

This is a numerical ablation of the learned energy, not retraining on the
test geometry or replacing its reference. Same box and physical coordinates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unichem.datasets import BOHR_ANGSTROM, HARTREE_EV
from unichem.experiment import load_quantum_potential
from unichem.factory import make_world
from unichem.grid import Grid
from unichem.species import HO_RESEARCH_SPECIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/designed_quantum/model.pt")
    parser.add_argument("--reference", default="outputs/water_dft_reference/reference.npz")
    parser.add_argument("--output", default="outputs/designed_grid_audit")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    _, predictor = load_quantum_potential(args.checkpoint)
    with np.load(args.reference, allow_pickle=False) as data:
        positions, reference_force = data["positions"].copy(), data["forces"].copy()
    rows = []
    for spacing in (0.6, 0.4, 0.3):
        side = round(28.8 / spacing)
        grid = Grid((side,) * 3, (spacing,) * 3)
        world = make_world(["O", "H", "H"], positions, grid=grid, registry=HO_RESEARCH_SPECIES)
        base = predictor(world, create_graph=False)
        energy = float(base.evaluation.energy.detach())
        shifts = []
        for axis in range(3):
            shifted = world.atoms.positions.clone()
            shifted[:, axis] += spacing / 2
            proposal = predictor(world.with_positions(shifted), create_graph=False)
            shifts.append(
                {
                    "axis": axis,
                    "shift_bohr": spacing / 2,
                    "energy_change_ev": (float(proposal.evaluation.energy.detach()) - energy)
                    * HARTREE_EV,
                    "force_change_rmse_ev_per_angstrom": float(
                        (proposal.forces.detach() - base.forces.detach()).square().mean().sqrt()
                    )
                    * HARTREE_EV
                    / BOHR_ANGSTROM,
                }
            )
        row = {
            "spacing_bohr": spacing,
            "grid_shape": [side] * 3,
            "energy_hartree": energy,
            "reference_force_rmse_ev_per_angstrom": float(
                (base.forces.detach() - torch.tensor(reference_force)).square().mean().sqrt()
            )
            * HARTREE_EV
            / BOHR_ANGSTROM,
            "subcell_translations": shifts,
            "largest_translation_energy_change_ev": max(abs(s["energy_change_ev"]) for s in shifts),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del base, proposal, world
    report = {
        "checkpoint": args.checkpoint,
        "reference": args.reference,
        "rows": rows,
        "training_grid_spacing_bohr": 0.6,
        "weights_frozen": True,
        "force_definition": "Total finite-electronic-map derivative",
        "scope": "Quadrature ablation; grid refinement alone is not chemical validation",
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
