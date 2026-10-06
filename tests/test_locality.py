import pytest
import torch

from unichem import BlockMatrix
from unichem.quantum.purification import ground_projector, orthogonalize


def test_block_compression_has_an_actual_operator_error_certificate():
    ids = torch.tensor([0, 0, 1, 1])
    cols = torch.tensor([0, 1, 0, 1])
    values = torch.tensor([1.0, 1e-7, 1e-7, 2.0], dtype=torch.float64).reshape(4, 1, 1)
    matrix = BlockMatrix.assemble(ids, cols, values, 2)
    screened, bound = matrix.compress(1e-6)
    assert len(screened.row) == 2
    difference = matrix.dense_for_validation() - screened.dense_for_validation()
    assert float(torch.linalg.matrix_norm(difference, ord=2)) <= bound + 1e-15
    assert bound <= 1e-6
    assert torch.equal(
        torch.diag(matrix.dense_for_validation()), torch.diag(screened.dense_for_validation())
    )


def test_gapped_local_operator_stays_sparse_and_matches_reference():
    n = 64
    ids = torch.arange(n)
    edges = torch.arange(n - 1)
    row = torch.cat((ids, edges, edges + 1))
    col = torch.cat((ids, edges + 1, edges))
    bonds = torch.where(edges % 2 == 0, -torch.ones(n - 1), -0.2 * torch.ones(n - 1)).double()
    values = torch.cat((torch.zeros(n, dtype=torch.float64), bonds, bonds)).reshape(-1, 1, 1)
    h = BlockMatrix.assemble(row, col, values, n)
    s = BlockMatrix.diagonal(torch.ones(n, 1, 1, dtype=torch.float64))
    result = ground_projector(h, orthogonalize(s, torch.ones(n, 1, dtype=torch.bool)), n // 2)
    assert len(result.density.row) < n * n * 0.6
    _, c = torch.linalg.eigh(h.dense_for_validation(max_atoms=n))
    reference = c[:, : n // 2] @ c[:, : n // 2].T
    assert result.density.dense_for_validation(max_atoms=n) == pytest.approx(reference, abs=1e-10)
