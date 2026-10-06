"""Read-only potential-surface and finite-field measurements.

Scan coordinates are analysis metadata. None of these observers participates
in a force calculation, training input, or a nuclear update.
"""

import numpy as np

from .datasets import HARTREE_EV
from .units import ELECTRON_MASS_PER_U


def vibrational_curvatures(positions, hessian, masses_u):
    """Mass-weighted Cartesian curvatures after removing rigid motion.

    Input Hessian is Hartree/bohr^2. Signed cm^-1 values represent imaginary
    curvatures with a minus sign. Frequencies require a stationary geometry.
    """
    r, h, masses = (np.asarray(a, dtype=float) for a in (positions, hessian, masses_u))
    n = len(r)
    if (
        r.shape != (n, 3)
        or h.shape != (3 * n, 3 * n)
        or masses.shape != (n,)
        or (masses <= 0).any()
    ):
        raise ValueError("Cartesian Hessian, positions and positive masses required")
    centre = (r * masses[:, None]).sum(0) / masses.sum()
    root = np.sqrt(masses)
    rigid = np.stack(
        [np.tile(axis, (n, 1)) * root[:, None] for axis in np.eye(3)]
        + [np.cross(axis, r - centre) * root[:, None] for axis in np.eye(3)],
        axis=-1,
    ).reshape(3 * n, 6)
    u, s, _ = np.linalg.svd(rigid, full_matrices=True)
    rank = int(np.count_nonzero(s > s[0] * 1e-9))
    q = u[:, rank:]
    inverse_root = 1 / np.repeat(np.sqrt(masses * ELECTRON_MASS_PER_U), 3)
    weighted = ((h + h.T) / 2) * inverse_root[:, None] * inverse_root[None, :]
    eigenvalues = np.linalg.eigvalsh(q.T @ weighted @ q)
    return np.sign(eigenvalues) * np.sqrt(abs(eigenvalues)) * 219474.6313632


def surface_statistics(delta, energy):
    """Measure a symmetric, frozen-environment proton scan, not a PMF."""
    delta, energy = np.asarray(delta, dtype=float), np.asarray(energy, dtype=float)
    order = np.argsort(delta)
    delta, energy = delta[order], energy[order]
    if (
        delta.ndim != 1
        or energy.shape != delta.shape
        or len(delta) < 5
        or not np.isfinite(energy).all()
        or not np.allclose(delta, -delta[::-1], atol=1e-9)
        or np.any(np.diff(delta) <= 0)
    ):
        raise ValueError("A finite, unique symmetric scan with at least five points is required")
    centre = int(np.argmin(abs(delta)))
    if abs(delta[centre]) > 1e-9:
        raise ValueError("The centred proton must be sampled")
    e = (energy - energy[centre]) * HARTREE_EV
    left, right = centre - 1, centre + 1
    h = delta[right]
    if not np.isclose(-delta[left], h):
        raise ValueError("Equal spacing around the centre required")
    curvature = float((e[left] + e[right]) / h**2)
    minimum = int(np.argmin(e))
    return {
        "delta_angstrom": delta.tolist(),
        "centred_energy_ev": e.tolist(),
        "centre_curvature_ev_per_angstrom2": curvature,
        "centre_shape": "single_well" if curvature > 0 else "double_well",
        "sampled_centre_barrier_ev": float(-e.min()),
        "sampled_minimum_abs_delta_angstrom": float(abs(delta[minimum])),
        "minimum_on_scan_boundary": minimum in (0, len(delta) - 1),
        "reflection_error_ev": float(abs(e - e[::-1]).max()),
    }


def field_statistics(strength, energy, dipole):
    """Symmetric finite differences of energies and dipoles in atomic units.

    The effective energy dipole of a finite electronic map need not equal its
    physical density dipole. Their difference is reported rather than hidden.
    """
    f, e, mu = (np.asarray(a, dtype=float) for a in (strength, energy, dipole))
    if (
        f.ndim != 1
        or e.shape != f.shape
        or mu.shape != f.shape
        or not all(np.isfinite(a).all() for a in (f, e, mu))
    ):
        raise ValueError("One energy and dipole per field strength required")
    zero = np.flatnonzero(np.isclose(f, 0, atol=1e-12))
    if len(zero) != 1 or len(np.unique(f)) != len(f):
        raise ValueError("Unique field strengths and one zero-field calculation required")
    measurements = []
    for h in sorted(f[f > 0]):
        minus = np.flatnonzero(np.isclose(f, -h, rtol=0, atol=1e-12))
        if len(minus) != 1:
            raise ValueError("Both field signs required")
        plus = int(np.flatnonzero(f == h)[0])
        minus, origin = int(minus[0]), int(zero[0])
        energy_dipole = -(e[plus] - e[minus]) / (2 * h)
        alpha_energy = -(e[plus] + e[minus] - 2 * e[origin]) / h**2
        alpha_density = (mu[plus] - mu[minus]) / (2 * h)
        measurements.append(
            {
                "field_magnitude_au": float(h),
                "dipole_from_energy_au": float(energy_dipole),
                "zero_field_density_dipole_au": float(mu[origin]),
                "dipole_consistency_error_au": float(abs(energy_dipole - mu[origin])),
                "polarizability_from_energy_bohr3": float(alpha_energy),
                "polarizability_from_density_bohr3": float(alpha_density),
                "polarizability_consistency_error_bohr3": float(abs(alpha_energy - alpha_density)),
            }
        )
    return measurements
