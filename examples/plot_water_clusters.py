"""Standalone scientific figures from measured water-cluster experiments."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/water_clusters_v04")
COLORS = ("#0072b2", "#d55e00")


def read_case(name, dt=0.1):
    folder = ROOT / f"{name}_n_2_seed_11_dt_{dt}"
    report = json.loads((folder / "report.json").read_text())
    data = np.load(folder / "trajectory.npz")
    return report, data


def main():
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    one = json.loads((ROOT / "benchmark_threads_1.json").read_text())
    four = json.loads((ROOT / "benchmark_threads_4.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    n = np.arange(1, 5)
    for metadata, offset, color in ((one, -0.15, COLORS[0]), (four, 0.15, COLORS[1])):
        rows = [r for r in metadata["rows"] if r["status"] == "completed"]
        values = [r["median_native_step_seconds"] for r in rows]
        axes[0].bar(
            n + offset,
            values,
            0.28,
            color=color,
            label=f"{metadata['environment']['threads']} CPU threads",
        )
        for x, row in zip(n + offset, rows):
            axes[0].scatter(
                [x] * len(row["native_step_seconds"]),
                row["native_step_seconds"],
                s=13,
                color="black",
                zorder=3,
            )
    axes[0].set_ylabel("Wall seconds / native step")
    axes[0].set_title("Full HF step, dt = 0.1 fs; 2 samples")
    axes[0].legend(fontsize=8)
    rows = one["rows"][:4]
    setup = np.array([r["integral_setup_seconds"] for r in rows])
    scf = np.array([r["cold_scf_seconds"] for r in rows])
    force = np.array([r["force_seconds"] for r in rows])
    axes[1].bar(n, setup, color="#009e73", label="Integral setup")
    axes[1].bar(n, scf, bottom=setup, color=COLORS[0], label="Cold SCF")
    axes[1].bar(n, force, bottom=setup + scf, color=COLORS[1], label="Analytic forces")
    axes[1].set_ylabel("Wall seconds / first energy + force")
    axes[1].set_title("Measured components, 1 thread")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        n,
        [r["density_blocks_per_spin"][0] for r in rows],
        "o-",
        color=COLORS[0],
        label="Stored blocks / spin",
    )
    axes[2].plot(n, (3 * n) ** 2, "k--", label="All atom pairs")
    axes[2].set_ylabel("Atom-pair density blocks")
    axes[2].set_title("These small water clusters are fully filled")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(n)
        ax.set_xlabel("Water molecules")
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Internal CA measurements; Ryzen 9 7945HX / WSL2; no other-solver comparison")
    fig.savefig(out / "performance.png", dpi=180)
    fig.savefig(out / "performance.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, scenario in enumerate(("shared_proton", "prepared_ion_pair", "oriented_ion_pair")):
        report, data = read_case(scenario)
        h_id = 6 if scenario == "shared_proton" else 1
        h = report["frames"][0]["hydrogen_ids"].index(h_id)
        time = data["time_fs"]
        distance = np.array([f["oh_distances_angstrom"][h] for f in report["frames"]])
        bonds = np.array([f["oh_mayer_indices"][h] for f in report["frames"]])
        for o, color in enumerate(COLORS):
            axes[0, col].plot(
                time, distance[:, o], color=color, label=f"O {report['frames'][0]['oxygen_ids'][o]}"
            )
            axes[1, col].plot(time, bonds[:, o], color=color)
        axes[0, col].set_title(
            {
                "shared_proton": "Prepared H5O2+: shared-proton motion",
                "prepared_ion_pair": "Geometric ion candidate control",
                "oriented_ion_pair": "Electron-supported initial ion pair",
            }[scenario]
        )
        axes[0, col].set_ylabel(f"H {h_id} to oxygen distance (Å)")
        axes[1, col].set_ylabel("Mayer OH bond index")
        axes[0, col].legend(fontsize=8)
        for row in range(2):
            axes[row, col].set_xlabel("Time (fs)")
            axes[row, col].grid(alpha=0.15)
        data.close()
    fig.suptitle(
        "Uniform HF/STO-3G nuclear evolution; prepared controls are not spontaneous autoionization"
    )
    fig.savefig(out / "proton_motion.png", dpi=180)
    fig.savefig(out / "proton_motion.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
