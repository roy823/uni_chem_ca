"""Reproducible numerical demonstrations. No experimental/DFT labels are used."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .checkpoint import save_world
from .evaluate import Evaluator
from .factory import make_world
from .grid import Grid
from .models.coupling import ConditionalGaussianCoupling
from .models.encoder import CoarseFieldEncoder
from .quantum.chemistry import chemical_observables
from .quantum.density import LocalQuantumBackend
from .quantum.hamiltonian import kernel_stationarity
from .runtime.coupled import BathConfig, CoupledSynchronousEngine
from .runtime.electrons import LocalKernelRelaxer


def run_coupled_demo(
    *, steps: int = 3, output: str | Path = "outputs/coupled", device: str = "cpu"
) -> dict:
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    world = make_world(
        ["O", "H", "H"],
        [[8.03, 8.07, 8.11], [9.63, 8.07, 8.11], [7.63, 9.57, 8.11]],
        grid=Grid((28, 28, 28), (0.6, 0.6, 0.6)),
        latent_stride=2,
        latent_kind="auxiliary_momentum",
        device=device,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(913)
        encoder = CoarseFieldEncoder(4, hidden=12).double().to(device)
    coupling = ConditionalGaussianCoupling(encoder, stiffness=0.1, gradient_stiffness=0.02)
    evaluator = Evaluator(backend=LocalQuantumBackend(), coupling=coupling)
    # Start the auxiliary coordinate at its conditional centre, not an
    # arbitrary offset whose energy could dominate the molecular demonstration.
    with torch.no_grad():
        world = world.with_latent(encoder(evaluator(world).context).clone())
    bath = BathConfig(kT=0.001, nuclear_friction=0.002, latent_friction=0.1)
    relaxer = LocalKernelRelaxer(max_iterations=4)
    engine = CoupledSynchronousEngine(evaluator, dt=0.05, relaxer=relaxer, bath=bath, seed=913)
    save_world(world, output / "initial.npz")
    records = []
    for _ in range(steps):
        result = engine.step(world)
        world = result.world
        records.append(result.diagnostics)
    save_world(world, output / "final.npz")
    torch.save(encoder.state_dict(), output / "encoder_state.pt")
    final = evaluator(world)
    h = kernel_stationarity(world, evaluator)
    chemistry = chemical_observables(final.context)
    array = lambda x: x.detach().cpu().numpy()
    np.savez_compressed(
        output / "fields.npz",
        density=array(final.context.quantum.density).reshape(world.grid.shape),
        potential=array(final.context.fields.potential),
        electric_faces=array(final.context.fields.electric_faces),
        latent=array(world.latent.channels),
        kernel_row=array(world.electrons.kernel.row),
        kernel_col=array(world.electrons.kernel.col),
        kernel_values=array(world.electrons.kernel.values),
        spin_counts=array(world.electrons.spin_counts),
    )
    report = {
        "version": __version__,
        "scope": "UNTRAINED NUMERICAL DEMONSTRATION; NOT VALIDATED CHEMISTRY",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "fine_grid": asdict(world.grid),
        "latent_grid": asdict(world.latent_grid),
        "bath": asdict(bath),
        "seed": 913,
        "dt_au": 0.05,
        "encoder": {"source": "nuclear", "channels": 4, "hidden": 12},
        "coupling": {"stiffness": 0.1, "gradient_stiffness": 0.02},
        "relaxer": {"algorithm": "local_kernel", "max_iterations": 4},
        "quantum_backend": final.context.quantum.diagnostics,
        "kernel_stationarity_norm": float(h.norm),
        "observer_not_bond_rules": {
            "assigned_valence_charge": chemistry.assigned_valence_charge.cpu().tolist(),
            "mayer_like": chemistry.mayer_like.cpu().tolist(),
        },
        "steps": records,
        "all_electronic_solves_converged": bool(records)
        and all(x["electrons_converged"] for x in records),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
