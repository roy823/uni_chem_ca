"""Scientific scaling artifacts for the bounded CA execution experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("outputs/local_ca_capacity")
    report = json.loads((root / "report_cuda.json").read_text())
    rows = report["rows"]
    cells = np.array([r["cells"] for r in rows])
    time = np.array([r["median_ca_step_seconds"] for r in rows])
    microseconds = time * 1e6 / cells
    lower = np.array([min(r["ca_step_seconds"]) for r in rows])
    upper = np.array([max(r["ca_step_seconds"]) for r in rows])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    axes[0].errorbar(
        cells,
        time,
        yerr=[time - lower, upper - time],
        fmt="o-",
        capsize=4,
        color="#0072b2",
        label="Measured median; min-max interval",
    )
    axes[0].plot(cells, time[-1] * cells / cells[-1], "k--", label="N reference, fixed hardware")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Atomic cells")
    axes[0].set_ylabel("Seconds / CA step")
    axes[0].set_title("Total latency on one GPU")
    axes[0].legend(fontsize=8)
    axes[1].plot(cells, microseconds, "o-", color="#009e73")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Atomic cells")
    axes[1].set_ylabel("Microseconds / cell / step")
    axes[1].set_title("Bounded local work; finite hardware throughput")
    for x, y in zip(cells, microseconds):
        axes[1].annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Local CA only · RTX 4060 Laptop GPU · Untrained fixture fields, not chemistry")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "ca_scaling.png", dpi=180)
    fig.savefig(output / "ca_scaling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
