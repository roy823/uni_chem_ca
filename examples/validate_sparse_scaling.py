"""Gapped-chain numerical scaling of the quantum projector, not liquid water."""

import json
import time
from pathlib import Path

import torch

from unichem.quantum.blocks import BlockMatrix
from unichem.quantum.purification import ground_projector, orthogonalize


def main():
    torch.set_num_threads(2)
    rows = []
    for n in (64, 128, 256, 512):
        ids = torch.arange(n)
        edges = torch.arange(n - 1)
        row = torch.cat((ids, edges, edges + 1))
        col = torch.cat((ids, edges + 1, edges))
        bonds = torch.where(edges % 2 == 0, -torch.ones(n - 1), -0.2 * torch.ones(n - 1)).double()
        values = torch.cat((torch.zeros(n, dtype=torch.float64), bonds, bonds)).reshape(-1, 1, 1)
        h = BlockMatrix.assemble(row, col, values, n)
        s = BlockMatrix.diagonal(torch.ones(n, 1, 1, dtype=torch.float64))
        start = time.perf_counter()
        projected = ground_projector(
            h, orthogonalize(s, torch.ones(n, 1, dtype=torch.bool)), n // 2
        )
        result = {
            "sites": n,
            "input_blocks": len(h.row),
            "density_blocks": len(projected.density.row),
            "seconds": time.perf_counter() - start,
            "number_error": projected.number_error,
            "purity_error": projected.purity_error,
            "measured_compression_bound_sum": projected.compression_bound_sum,
        }
        if n <= 128:
            _, coefficients = torch.linalg.eigh(h.dense_for_validation(max_atoms=128))
            reference = coefficients[:, : n // 2] @ coefficients[:, : n // 2].T
            result["dense_reference_error"] = float(
                abs(projected.density.dense_for_validation(max_atoms=128) - reference).max()
            )
        rows.append(result)
        print(json.dumps(result), flush=True)
    root = Path("outputs/v04_validation/scaling")
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(
        json.dumps(
            {
                "cases": rows,
                "scope": "Manufactured gapped nearest-neighbour operator; not molecular-integral scaling, dense water or metallic locality validation",
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
