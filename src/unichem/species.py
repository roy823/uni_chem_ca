from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import StateError
from .quantum.angular import COMPONENTS
from .units import ELECTRON_MASS_PER_U


@dataclass(frozen=True)
class OrbitalSpec:
    """Real s/p/d Gaussian channel, not a complete hydrogenic eigenfunction.

    n is a shell label. real_component is a real solid harmonic, not a definite m_l
    for px/py. A complex spherical-harmonic backend can use a richer schema.
    width is an AMPLITUDE width in bohr: exp[-r^2 / (2*width^2)].
    """

    n: int
    ell: int
    real_component: str
    width: float

    def __post_init__(self):
        allowed = {name: values[1] for name, values in COMPONENTS.items()}
        if self.real_component not in allowed or allowed[self.real_component] != self.ell:
            raise StateError("The reference basis implements only real s, p and d channels")
        if (
            type(self.n) is not int
            or type(self.ell) is not int
            or self.n <= self.ell
            or not (0 < self.width < float("inf"))
        ):
            raise StateError("Invalid orbital shell or width")

    @property
    def axis(self) -> int:
        return COMPONENTS[self.real_component][0]


@dataclass(frozen=True)
class Species:
    symbol: str
    atomic_number: int
    core_charge: float
    mass: float
    core_smearing: float
    orbitals: tuple[OrbitalSpec, ...]

    def __post_init__(self):
        if (
            type(self.atomic_number) is not int
            or self.atomic_number < 1
            or not (0 < self.core_charge <= self.atomic_number)
        ):
            raise StateError("Invalid atomic number / effective core charge")
        if (
            not self.orbitals
            or not (0 < self.mass < math.inf)
            or not (0 < self.core_smearing < math.inf)
        ):
            raise StateError("Species needs positive mass, smearing, and a basis")


# ILLUSTRATIVE valence basis, NOT fitted atomic radii or a validated pseudopotential.
# Oxygen represents a +6 frozen-core ion and six valence electrons, NOT a +8
# point nucleus plus six electrons. The core smearing is numerical, not nuclear size.
DEMO_SPECIES: dict[str, Species] = {
    "H": Species("H", 1, 1.0, 1.008 * ELECTRON_MASS_PER_U, 0.55, (OrbitalSpec(1, 0, "s", 0.80),)),
    "O": Species(
        "O",
        8,
        6.0,
        15.999 * ELECTRON_MASS_PER_U,
        0.55,
        (
            OrbitalSpec(2, 0, "s", 1.10),
            OrbitalSpec(2, 1, "px", 0.95),
            OrbitalSpec(2, 1, "py", 0.95),
            OrbitalSpec(2, 1, "pz", 0.95),
        ),
    ),
}


# An explicitly separate research basis, not a fitted pseudopotential. Extra
# radial s channels allow radial nodes; H p channels permit polarization.
# Training records this basis identity and never imports older model weights.
HO_RESEARCH_SPECIES = {
    "H": Species(
        "H",
        1,
        1.0,
        1.008 * ELECTRON_MASS_PER_U,
        0.55,
        (
            OrbitalSpec(1, 0, "s", 0.8),
            OrbitalSpec(2, 0, "s", 1.25),
            OrbitalSpec(2, 1, "px", 1.0),
            OrbitalSpec(2, 1, "py", 1.0),
            OrbitalSpec(2, 1, "pz", 1.0),
        ),
    ),
    "O": Species(
        "O",
        8,
        6.0,
        15.999 * ELECTRON_MASS_PER_U,
        0.55,
        (
            OrbitalSpec(2, 0, "s", 0.65),
            OrbitalSpec(2, 0, "s", 1.1),
            OrbitalSpec(2, 1, "px", 0.95),
            OrbitalSpec(2, 1, "py", 0.95),
            OrbitalSpec(2, 1, "pz", 0.95),
        ),
    ),
}
