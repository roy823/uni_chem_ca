"""Independent dense small-system oracles for the local quantum map.

Production uses block-sparse matrices. Dense arrays here are explicitly
limited validation instruments; they do not replace the production backend.
"""

import numpy as np
import torch

from ..quantum.blocks import BlockMatrix


def padded_indices(counts):
    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 1 or len(counts) > 8 or counts.sum() > 16 or (counts < 1).any():
        raise ValueError("Dense audit limited to 8 atoms and 16 active AOs")
    width = int(counts.max())
    indices = np.concatenate([i * width + np.arange(count) for i, count in enumerate(counts)])
    active = np.arange(width)[None, :] < counts[:, None]
    return indices, active


def block_operator(matrix, counts):
    indices, active = padded_indices(counts)
    n, width = active.shape
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.shape != (len(indices), len(indices)):
        raise ValueError("AO matrix and per-atom AO counts disagree")
    padded = matrix.new_zeros((n * width, n * width))
    query = torch.tensor(indices)
    padded[query[:, None], query[None, :]] = matrix
    row = torch.arange(n).repeat_interleave(n)
    col = torch.arange(n).repeat(n)
    values = padded.reshape(n, width, n, width).permute(0, 2, 1, 3).reshape(n * n, width, width)
    return BlockMatrix.assemble(row, col, values, n)


def spin_density_diagnostics(kernels, overlap, counts):
    eigenvalues, vectors = np.linalg.eigh(overlap)
    root = (vectors * np.sqrt(np.clip(eigenvalues, 0, None))) @ vectors.T
    rows = []
    for kernel, count in zip(kernels, counts):
        physical = root @ kernel @ root
        occupations = np.linalg.eigvalsh((physical + physical.T) / 2)
        rows.append(
            {
                "electron_number_error": float(abs(np.trace(kernel @ overlap) - count)),
                "minimum_occupation": float(occupations.min()),
                "maximum_occupation": float(occupations.max()),
                "idempotency_defect_frobenius": float(
                    np.linalg.norm(physical @ physical - physical)
                ),
                "occupations": occupations.tolist(),
            }
        )
    return rows
