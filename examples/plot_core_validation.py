"""Publication-style artifacts for quantum core accuracy and sparse evolution."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


def main():
    root = Path("outputs/v04_validation")
    new = json.loads((root / "matrix/report.json").read_text())
    old = json.loads(Path("outputs/quantum_baseline/oracle/report.json").read_text())
    motion = json.loads((root / "molecular/report.json").read_text())
    scaling = json.loads((root / "scaling/report.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    cases = [r for r in new["frames"] if r["name"].startswith("one_electron_pair")]
    previous = [r for r in old["frames"] if r["reference"]["name"].startswith("one_electron_pair")]
    distances = [r["reference"]["distance_angstrom"] for r in previous]
    axes[0].semilogy(
        distances,
        [abs(r["best"]["energy_excess_ev"]) for r in previous],
        "o-",
        color="#d55e00",
        label="v0.3 soft occupation",
    )
    axes[0].semilogy(
        distances,
        [max(r["energy_error_hartree"] * 27.211386245988, 1e-14) for r in cases],
        "o-",
        color="#0072b2",
        label="v0.4 zero-temperature core",
    )
    axes[0].set_xlabel("H-H separation (Å)")
    axes[0].set_ylabel("Same-operator energy error (eV)")
    axes[0].set_title("H₂⁺ ground state; zeros displayed at 1e-14")
    axes[0].legend(fontsize=8)
    runs = sorted(motion["runs"], key=lambda r: r["dt_fs"])
    dt = np.array([r["dt_fs"] for r in runs])
    errors = np.array([r["max_energy_balance_error_ev"] for r in runs])
    axes[1].loglog(dt, errors, "o-", color="#0072b2", label="Native quantum NVE")
    axes[1].loglog(dt, errors[-1] * (dt / dt[-1]) ** 2, "k--", label="Δt² reference")
    axes[1].set_xlabel("Time step (fs)")
    axes[1].set_ylabel("Maximum energy balance error (eV)")
    axes[1].set_title("Untrained water; HF/STO-3G, 10 fs")
    axes[1].legend(fontsize=8)
    s = scaling["cases"]
    n = np.array([r["sites"] for r in s])
    blocks = np.array([r["density_blocks"] for r in s])
    axes[2].loglog(n, blocks, "o-", color="#0072b2", label="Stored density blocks")
    axes[2].loglog(n, n * n, "k--", label="Dense n² storage")
    axes[2].set_xlabel("Sites")
    axes[2].set_ylabel("Matrix blocks")
    axes[2].set_title("Gapped local operator; not molecular scaling")
    axes[2].set_xticks(n, [str(int(x)) for x in n])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Core architecture replacement: physical operators, pure densities, consistent forces"
    )
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "core_optimization.png", dpi=180)
    fig.savefig(out / "core_optimization.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for dt, color in zip((0.2, 0.1, 0.05), ("#0072b2", "#d55e00", "#009e73")):
        with np.load(root / "molecular" / f"nve_dt_{dt}.npz") as data:
            r = data["positions"]
            v = r[:, 1:] - r[:, :1]
            lengths = np.linalg.norm(v, axis=-1) * 0.529177210903
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        (v[:, 0] * v[:, 1]).sum(-1)
                        / (np.linalg.norm(v[:, 0], axis=-1) * np.linalg.norm(v[:, 1], axis=-1)),
                        -1,
                        1,
                    )
                )
            )
            axes[0].plot(data["time_fs"], lengths[:, 0], color=color, label=f"{dt} fs")
            axes[1].plot(data["time_fs"], angle, color=color)
            axes[2].plot(data["time_fs"], data["energy_balance_ev"], color=color)
    for ax in axes:
        ax.set_xlabel("Time (fs)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("O-H distance (Å)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("H-O-H angle (degrees)")
    axes[2].set_ylabel("Energy balance error (eV)")
    fig.suptitle("Native quantum-cell water trajectory; no learned energy or bond rule")
    fig.savefig(out / "native_water.png", dpi=180)
    fig.savefig(out / "native_water.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
