from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .errors import StateError
from .evaluate import EnergyEvaluation, Evaluator
from .factory import make_world
from .grid import Grid
from .models.energy import AnalyticTeacherResidual, SharedScalarResidual
from .state import World


@dataclass(frozen=True)
class Prediction:
    evaluation: EnergyEvaluation
    forces: Tensor


class KernelUnrolledPredictor:
    """Differentiable fixed-count synchronous electronic sweeps for training.

    Unlike the production/reference line search, this path keeps the graph.
    It enables density supervision to train the energy residual through its
    induced electronic relaxation. Forces are gradients of the ENTIRE unrolled
    scalar map. They are not claimed to be converged BO forces.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        electronic_steps: int = 1,
        electronic_step_size: float = 0.03,
        kernel_initializer=None,
    ):
        if electronic_steps < 0 or electronic_step_size <= 0:
            raise StateError("Invalid differentiable relaxation settings")
        self.evaluator, self.electronic_steps = evaluator, electronic_steps
        self.step_size = electronic_step_size
        self.kernel_initializer = kernel_initializer

    def __call__(self, world: World, *, create_graph: bool = True) -> Prediction:
        with torch.enable_grad():
            positions = world.atoms.positions.detach().clone().requires_grad_(True)
            values = world.electrons.kernel.values.detach().clone().requires_grad_(True)
            snapshot = world.with_positions(positions).with_kernel_values(values)
            if self.kernel_initializer is not None:
                snapshot = self.kernel_initializer(snapshot)
                values = snapshot.electrons.kernel.values
            for _ in range(self.electronic_steps):
                energy = self.evaluator(snapshot).energy + values.sum() * 0
                gradient = torch.autograd.grad(energy, values, create_graph=True)[0]
                values = (values - self.step_size * gradient).clamp(-8, 8)
                snapshot = snapshot.with_kernel_values(values)
            evaluation = self.evaluator(snapshot)
            forces = -torch.autograd.grad(
                evaluation.energy + positions.sum() * 0,
                positions,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        return Prediction(evaluation, forces)


@dataclass(frozen=True)
class TrainingExample:
    world: World
    energy: Tensor
    forces: Tensor
    density: Tensor


@dataclass(frozen=True)
class LossWeights:
    energy: float = 1.0
    forces: float = 0.2
    density: float = 0.1


DEFAULT_LOSS_WEIGHTS = LossWeights()


def supervised_loss(
    pred: Prediction, target: TrainingExample, weights: LossWeights = DEFAULT_LOSS_WEIGHTS
) -> tuple[Tensor, dict[str, Tensor]]:
    """Energy/force/density loss with explicit quadrature weighting.

    Real datasets must share units, basis/core conventions, boundary conditions,
    spin sector and state labelling. Arbitrary DFT arrays cannot simply be mixed.
    Density L2 is a volume integral, not a mesh-size-dependent unweighted sum.
    """
    if (
        pred.forces.shape != target.forces.shape
        or pred.evaluation.context.quantum.density.shape != target.density.shape
    ):
        raise StateError("Supervision shape mismatch")
    n = target.world.n_atoms
    terms = {
        "energy": ((pred.evaluation.energy - target.energy) / n).square(),
        "forces": (pred.forces - target.forces).square().mean(),
        "density": (pred.evaluation.context.quantum.density - target.density).square().sum()
        * target.world.grid.dv
        / target.world.electron_count,
    }
    loss = (
        weights.energy * terms["energy"]
        + weights.forces * terms["forces"]
        + weights.density * terms["density"]
    )
    return loss, terms


def synthetic_examples(*, count: int = 8, electronic_steps: int = 1) -> list[TrainingExample]:
    """In-memory synthetic H2-like labels; no DFT, experiment or bond labels."""
    grid = Grid((18, 18, 18), (0.6, 0.6, 0.6))
    teacher = KernelUnrolledPredictor(
        Evaluator(residual=AnalyticTeacherResidual()), electronic_steps=electronic_steps
    )
    out = []
    for index in range(count):
        distance = 1.5 + 1.4 * index / max(1, count - 1)
        shift = 0.037 * index
        world = make_world(
            ["H", "H"],
            [[5.1 - distance / 2, 5.3 + shift, 5.1], [5.1 + distance / 2, 5.3 + shift, 5.1]],
            grid=grid,
        )
        pred = teacher(world, create_graph=False)
        out.append(
            TrainingExample(
                world,
                pred.evaluation.energy.detach(),
                pred.forces.detach(),
                pred.evaluation.context.quantum.density.detach(),
            )
        )
    return out


def train_synthetic(
    *, epochs: int = 25, output: str | Path = "outputs/training", seed: int = 13
) -> dict:
    """Exercise real backpropagation through energy, force and electronic density.

    Reported errors are against a synthetic teacher ONLY; they are not chemical
    accuracy, not transferability, and not a fitted solution-chemistry model.
    """
    if epochs < 1:
        raise StateError("epochs must be >=1")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SharedScalarResidual(hidden=12).double()
    evaluator = Evaluator(residual=model)
    predictor = KernelUnrolledPredictor(evaluator, electronic_steps=1)
    examples = synthetic_examples(count=8)
    train = [examples[i] for i in (0, 2, 3, 5, 7)]
    validation = [examples[i] for i in (1, 4, 6)]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

    def score(data):
        values = []
        for example in data:
            prediction = predictor(example.world, create_graph=False)
            loss, _ = supervised_loss(prediction, example)
            values.append(float(loss.detach()))
        return sum(values) / len(values)

    history = [{"epoch": 0, "train_loss": score(train), "validation_loss": score(validation)}]
    for epoch in range(1, epochs + 1):
        for example in train:
            optimizer.zero_grad(set_to_none=True)
            prediction = predictor(example.world, create_graph=True)
            loss, _ = supervised_loss(prediction, example)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise StateError("Non-finite training gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            history.append(
                {"epoch": epoch, "train_loss": score(train), "validation_loss": score(validation)}
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_class": "SharedScalarResidual",
            "hidden": 12,
            "state_dict": model.state_dict(),
            "synthetic_only": True,
        },
        output / "synthetic_model.pt",
    )
    report = {
        "status": "synthetic_pipeline_test_only",
        "seed": seed,
        "electronic_sweeps": 1,
        "electronic_update": "local_kernel",
        "training_examples": len(train),
        "validation_examples": len(validation),
        "history": history,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
