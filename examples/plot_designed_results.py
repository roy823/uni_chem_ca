"""Standalone research plots of the complete scan and short NVE pilot."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", default="outputs/designed_screen/report.json")
    parser.add_argument("--water", default="outputs/designed_water")
    parser.add_argument("--output", default="outputs/designed_figures")
    args = parser.parse_args()
    report = json.loads(Path(args.screen).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for ax, surface in zip(axes.ravel(), report["zundel_surfaces"]):
        ref = surface["reference"]
        x = ref["delta_angstrom"]
        ax.plot(x, ref["centred_energy_ev"], "k.-", label="PBE0 reference")
        if "model" in surface:
            ax.plot(
                x,
                surface["model"]["centred_energy_ev"],
                "o-",
                color="#d55e00",
                markersize=3,
                label="Fresh quantum model",
            )
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_title(f"R(O-O) = {surface['oo_angstrom']:.1f} Å | {surface['split']}")
        ax.set_xlabel("Proton distance difference δ (Å)")
        ax.set_ylabel("E(δ) - E(0) (eV)")
        ax.grid(alpha=0.15)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Frozen-environment proton potential: all nine surfaces; not a free energy")
    fig.savefig(out / "zundel_surfaces.png", dpi=180)
    fig.savefig(out / "zundel_surfaces.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    fields = report["water_field_response"]
    x = np.arange(3)
    ref = [r["reference"][0]["polarizability_from_energy_bohr3"] for r in fields]
    ax.bar(x - 0.22, ref, 0.22, color="0.3", label="Reference (energy derivative)")
    if "model" in fields[0]:
        ax.bar(
            x,
            [r["model"][0]["polarizability_from_energy_bohr3"] for r in fields],
            0.22,
            color="#d55e00",
            label="Model (energy derivative)",
        )
        ax.bar(
            x + 0.22,
            [r["model"][0]["polarizability_from_density_bohr3"] for r in fields],
            0.22,
            color="#0072b2",
            label="Model (density response)",
        )
    ax.set_xticks(x, ["x", "y", "z"])
    ax.set_ylabel("Polarizability (bohr³)")
    ax.set_title("Water field response at |F| = 0.001 au")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.axhline(0, color="0.5", lw=0.7)
    fig.savefig(out / "water_polarizability.png", dpi=180)
    fig.savefig(out / "water_polarizability.pdf")
    plt.close(fig)
    if (Path(args.water) / "report.json").exists():
        water_report = json.loads((Path(args.water) / "report.json").read_text())
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
        duration = 0.0
        for dt, color in zip((0.2, 0.1, 0.05), ("#0072b2", "#d55e00", "#009e73")):
            with np.load(Path(args.water) / f"nve_dt_{dt}.npz", allow_pickle=False) as data:
                t = data["time_fs"]
                duration = max(duration, float(t[-1]))
                axes[0].plot(t, data["geometry"][:, 0], color=color, label=f"dt = {dt} fs")
                axes[0].plot(t, data["geometry"][:, 1], color=color, ls="--", alpha=0.7)
                axes[1].plot(t, data["geometry"][:, 2], color=color)
                axes[2].plot(t, data["balance_residual_ev"], color=color)
        for ax in axes:
            ax.set_xlabel("Time (fs)")
            ax.set_xlim(0, min(50, duration))
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("O-H distance (Å)")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("H-O-H angle (degrees)")
        axes[2].set_ylabel("NVE energy balance residual (eV)")
        fig.suptitle(
            f"Geometry-only water NVE control: first {min(50, duration):g} fs of {duration:g} fs"
            if water_report["kind"] == "geometry"
            else "Exploratory unvalidated water pilot; full finite-map forces"
        )
        fig.savefig(out / "water_nve_pilot.png", dpi=180)
        fig.savefig(out / "water_nve_pilot.pdf")
        plt.close(fig)
    print(json.dumps({"figures": sorted(str(p) for p in out.iterdir())}))


if __name__ == "__main__":
    main()
