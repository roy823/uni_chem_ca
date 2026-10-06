"""Standalone scientific figures from measured reports, no fitted plotting data."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unichem.datasets import BOHR_ANGSTROM, HARTREE_EV, load_reference_groups


def main():
    out = Path("outputs/v03_validation/figures")
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads(Path("outputs/two_water/report.json").read_text(encoding="utf8"))
    numerics = json.loads(Path("outputs/v03_validation/numerics.json").read_text(encoding="utf8"))
    reference = [
        f
        for f in load_reference_groups("data/ho_pbe0", split="validation")
        if f.group == "neutral_pair_scan"
    ]
    x = np.array(
        [np.linalg.norm(f.positions[2] - f.positions[0]) * BOHR_ANGSTROM for f in reference]
    )
    prediction = np.array([f["energy_hartree"] for f in result["reference_comparison"]])
    exact = np.array([f.energy for f in reference])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, (exact - exact[0]) * HARTREE_EV, "o-", label="PBE0/aug-cc-pVDZ reference")
    ax.plot(x, (prediction - prediction[0]) * HARTREE_EV, "s--", label="Local quantum ML model")
    ax.set(
        xlabel="Reference displaced O-H distance (angstrom)",
        ylabel="Relative potential energy (eV)",
        title="Independent water-pair scan: reference accuracy FAILED",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "independent_pes.png", dpi=180)
    plt.close(fig)
    chain = numerics["controlled_banded_operator_scaling"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(
        [v["atoms"] for v in chain],
        [v["density_blocks"] for v in chain],
        "o-",
        label="Stored local blocks",
    )
    nodes = np.array([v["atoms"] for v in chain])
    ax.loglog(
        nodes, nodes * chain[0]["density_blocks"] / chain[0]["atoms"], ":", label="Linear guide"
    )
    ax.set(
        xlabel="Number of blocks/sites",
        ylabel="Stored density-kernel blocks",
        title="Controlled banded graph; not liquid-water scaling",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "controlled_scaling.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
    for trajectory, observations in zip(result["trajectories"], result["read_only_observations"]):
        time = np.array([v["time_au"] for v in trajectory["diagnostics"]]) * 0.024188843265857
        balance = (
            np.array([v["balance_residual_hartree"] for v in trajectory["diagnostics"]])
            * HARTREE_EV
        )
        axes[0].plot(time, balance, label=f"seed {trajectory['seed']}")
        charges = np.array([v["assigned_fragment_charge_e"] for v in observations["observations"]])
        axes[1].plot(time, charges[:, 0], label=f"seed {trajectory['seed']}, fragment 0")
    axes[0].set(
        ylabel="Energy - bath heat drift (eV)", title="Unvalidated 2-water exploratory trajectories"
    )
    axes[1].set(xlabel="Time (fs)", ylabel="Partitioned fragment charge (e)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectory_diagnostics.png", dpi=180)
    plt.close(fig)
    print("Measured figures written to", out)


if __name__ == "__main__":
    main()
