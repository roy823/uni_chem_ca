"""Real-reference training, with distinct geometry and electronic comparisons."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .datasets import BOHR_ANGSTROM, HARTREE_EV, load_reference_groups
from .evaluate import Evaluator
from .factory import make_world
from .fields.isolated import IsolatedPoisson
from .grid import Grid
from .models.local_energy import LocalEnergyModel, LocalQuantumResidual
from .quantum.density import LocalQuantumBackend
from .species import HO_RESEARCH_SPECIES
from .training import KernelUnrolledPredictor


def frame_world(frame, spacing=0.6):
    shape = (
        frame.density_grid_shape
        if frame.density_grid_shape is not None
        else tuple(max(20, int(np.ceil(x / spacing))) for x in frame.box)
    )
    grid = Grid(shape, tuple(float(x) / n for x, n in zip(frame.box, shape)))
    symbols = ["H" if z == 1 else "O" for z in frame.atomic_numbers]
    return make_world(
        symbols,
        frame.positions,
        grid=grid,
        net_charge=frame.net_charge,
        external_field=frame.external_field,
        registry=HO_RESEARCH_SPECIES,
    )


def train_reference(
    *,
    dataset="data/ho_pbe0",
    output="outputs/ho_final_quantum",
    epochs=10,
    max_frames=32,
    kind="quantum",
    seed=203,
    hidden=48,
    electronic_steps=1,
    metric_degree=64,
    resume=None,
):
    started = time.perf_counter()
    if kind not in ("quantum", "geometry") or epochs < 1 or max_frames < 1:
        raise ValueError("Invalid reference training configuration")
    methods = set()
    for path in Path(dataset).glob("*/*/frames.npz"):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if meta["units"] != "atomic":
                raise ValueError("Reference unit mismatch")
            methods.add(meta["method"])
    if len(methods) != 1:
        raise ValueError("A training run must use one reference method and basis")
    train = load_reference_groups(dataset, split="training", max_frames=max_frames)
    validation = load_reference_groups(
        dataset, split="validation", max_frames=max(4, max_frames // 4)
    )
    if kind == "geometry" and any(
        f.net_charge or any(f.external_field) for f in train + validation
    ):
        raise ValueError("Geometry-only comparison is limited to neutral zero-field references")
    if not train or not validation:
        raise ValueError("Reference training/validation frames required")
    if kind == "quantum" and max(len(f.atomic_numbers) for f in train) > 32:
        raise ValueError(
            "Quantum reference training budget: <=32 atoms per frame; never crop labeled bulk frames"
        )
    targets = np.array([f.energy for f in train])
    if kind == "quantum":
        base = Evaluator(
            backend=LocalQuantumBackend(metric_degree=metric_degree), poisson=IsolatedPoisson()
        )
        baselines = []
        for index, frame in enumerate(train):
            baselines.append(float(base(frame_world(frame)).energy.detach()))
            if (index + 1) % 20 == 0:
                print(json.dumps({"baseline_frames": index + 1, "total": len(train)}), flush=True)
        targets -= np.array(baselines)
    design = np.array([[len(f.atomic_numbers), f.atomic_numbers.sum()] for f in train])
    offset, slope = np.linalg.lstsq(design, targets, rcond=None)[0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cls = LocalQuantumResidual if kind == "quantum" else LocalEnergyModel
        model = cls(hidden=hidden, reference_per_atom=offset).double()
        reference = model.geometry if kind == "quantum" else model
        reference.reference_slope.copy_(reference.reference_slope.new_tensor(slope))
    resumed = None
    if resume is not None:
        resumed = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            resumed.get("feature_version") != "joint_patch_sp_v3"
            or resumed["kind"] != kind
            or resumed["hidden"] != hidden
            or resumed["dataset"] != str(dataset)
        ):
            raise ValueError("Resume requires matching current architecture and dataset")
        model.load_state_dict(resumed["state_dict"])
        offset = float(reference.reference)
        slope = float(reference.reference_slope)
    if kind == "quantum":
        backend = LocalQuantumBackend(metric_degree=metric_degree)
        predictor = KernelUnrolledPredictor(
            Evaluator(backend=backend, residual=model, poisson=IsolatedPoisson()),
            electronic_steps=electronic_steps,
            kernel_initializer=model.initialize_kernel,
        )
        worlds = {id(f): frame_world(f) for f in train + validation}

    def predict(frame, graph):
        if kind == "quantum":
            p = predictor(worlds[id(frame)], create_graph=graph)
            return p.evaluation.energy, p.forces, p.evaluation.context.quantum.density
        energy, forces = model.energy_and_forces(
            torch.tensor(frame.atomic_numbers),
            torch.tensor(frame.positions, dtype=torch.float64),
            torch.tensor(frame.box, dtype=torch.float64),
            create_graph=graph,
        )
        return energy, forces, None

    def terms(frame, graph):
        energy, forces, density = predict(frame, graph)
        de = (energy - frame.energy) / len(frame.atomic_numbers)
        df = forces - torch.tensor(frame.forces, dtype=torch.float64)
        density_loss = de * 0
        if density is not None and frame.density is not None:
            if density.shape != frame.density.shape:
                raise ValueError("Reference density grid mismatch")
            density_loss = (
                (density - torch.tensor(frame.density, dtype=density.dtype)).square().sum()
            )
            density_loss *= worlds[id(frame)].grid.dv / worlds[id(frame)].electron_count
        return de, df, density_loss

    def score(frames):
        es, fs, ds = [], [], []
        for f in frames:
            de, df, drho = terms(f, False)
            es.append(float(de.detach()) ** 2)
            fs.append(float(df.detach().square().mean()))
            ds.append(float(drho.detach()))
        return {
            "energy_rmse_ev_per_atom": float(np.sqrt(np.mean(es)) * HARTREE_EV),
            "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean(fs)) * HARTREE_EV / BOHR_ANGSTROM),
            "density_l2_per_electron": float(np.mean(ds))
            if kind == "quantum" and any(f.density is not None for f in frames)
            else None,
        }

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if resumed is not None and "optimizer_state" in resumed:
        optimizer.load_state_dict(resumed["optimizer_state"])
    history = [{"epoch": 0, "train": score(train), "validation": score(validation)}]

    def selection_value(scores):
        value = scores["energy_rmse_ev_per_atom"] + scores["force_rmse_ev_per_angstrom"]
        if scores["density_l2_per_electron"] is not None:
            value += float(np.sqrt(scores["density_l2_per_electron"] / 0.01))
        return value

    print(json.dumps(history[-1]), flush=True)
    best = selection_value(history[-1]["validation"])
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    selected_epoch = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        for index in rng.permutation(len(train)):
            optimizer.zero_grad(set_to_none=True)
            de, df, drho = terms(train[index], True)
            loss = de.square() + 0.3 * df.square().mean() + 0.05 * drho
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("Nonfinite reference gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            scores = score(validation)
            history.append({"epoch": epoch, "train": score(train), "validation": scores})
            value = selection_value(scores)
            if value < best:
                best = value
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                selected_epoch = epoch
            print(json.dumps(history[-1]), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "training_frames": len(train),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # This metadata prevents geometry pretraining being mistaken for an
    # electron-density-trained or chemically validated quantum potential.
    payload = {
        "schema_version": 3,
        "kind": kind,
        "hidden": hidden,
        "state_dict": model.state_dict(),
        "reference_per_atom": float(offset),
        "reference_slope": float(slope),
        "dataset": str(dataset),
        "electronic_steps": electronic_steps,
        "metric_degree": metric_degree,
        "seed": seed,
        "synthetic_only": False,
        "chemistry_validated": False,
        "reference_method": next(iter(methods)),
        "density_supervised": kind == "quantum" and all(f.density is not None for f in train),
        "feature_version": "joint_patch_sp_v3",
        "basis": "ho_radial_sp_v3" if kind == "quantum" else None,
        "electrostatics": "isolated_free_space_fft" if kind == "quantum" else "geometry_only",
        "training_atoms_min": min(len(f.atomic_numbers) for f in train),
        "training_atoms_max": max(len(f.atomic_numbers) for f in train),
    }
    torch.save({**payload, "optimizer_state": optimizer.state_dict()}, out / "training_last.pt")
    model.load_state_dict(best_state)
    payload["state_dict"] = model.state_dict()
    torch.save(payload, out / "model.pt")
    report = {k: v for k, v in payload.items() if k != "state_dict"}
    report.update(
        {
            "training_frames": len(train),
            "validation_frames": len(validation),
            "resume_checkpoint": str(resume) if resume else None,
            "model_selection_scales": {
                "energy_ev_per_atom": 1.0,
                "force_ev_per_angstrom": 1.0,
                "density_l2_per_electron_e_per_bohr3": 0.01,
            },
            "training_ids": [[f.group, f.index] for f in train],
            "validation_ids": [[f.group, f.index] for f in validation],
            "history": history,
            "selected_model_validation": score(validation),
            "selected_epoch": selected_epoch,
            "epochs_completed": epochs,
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "real energy/force pipeline; chemical accuracy, free energies and kinetics require independent validation",
        }
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
