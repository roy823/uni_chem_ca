"""The same H cloud on successively finer lattices; no atomic-size rescaling."""

import json
from pathlib import Path

import torch

from unichem import Evaluator, Grid, make_world


def main():
    torch.set_num_threads(1)
    rows = []
    for size, h in [(20, 0.6), (30, 0.4), (40, 0.3)]:
        world = make_world(["H"], [[6.137, 6.234, 6.351]], grid=Grid((size,) * 3, (h,) * 3))
        c = Evaluator()(world).context
        rows.append(
            {
                "grid_shape": [size] * 3,
                "spacing_bohr": h,
                "support_voxels": int(c.support.counts_per_atom()[0]),
                "support_volume_bohr3": int(c.support.counts_per_atom()[0]) * world.grid.dv,
                "electron_rms_radius_bohr": float(c.clouds.rms_radius[0]),
                "electron_integral": float(c.quantum.density.sum() * world.grid.dv),
            }
        )
    report = {
        "system": "single H, illustrative Gaussian basis, same physical box and cloud",
        "results": rows,
    }
    path = Path("outputs/resolution")
    path.mkdir(parents=True, exist_ok=True)
    (path / "resolution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
