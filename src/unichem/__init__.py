"""Quantum-informed, synchronous extended-cell research infrastructure.

An atom is an extended quantum cell; a lattice voxel is only a quadrature/field cell.
The shipped energy is a numerical demonstration, NOT a calibrated chemical potential.
"""

from .evaluate import Evaluator
from .factory import make_world
from .grid import Grid
from .runtime.engine import StepResult, SynchronousEngine
from .state import AtomState, ElectronState, LatentState, World

__all__ = [
    "AtomState",
    "ElectronState",
    "Evaluator",
    "Grid",
    "LatentState",
    "StepResult",
    "SynchronousEngine",
    "World",
    "make_world",
]
__version__ = "0.3.0"

from .models.coupling import ConditionalGaussianCoupling
from .models.encoder import CoarseFieldEncoder
from .quantum.blocks import BlockMatrix
from .quantum.density import LocalQuantumBackend
from .runtime.coupled import BathConfig, CoupledSynchronousEngine
from .runtime.electrons import LocalKernelRelaxer

__all__ += [
    "BathConfig",
    "BlockMatrix",
    "CoarseFieldEncoder",
    "ConditionalGaussianCoupling",
    "CoupledSynchronousEngine",
    "LocalKernelRelaxer",
    "LocalQuantumBackend",
]
