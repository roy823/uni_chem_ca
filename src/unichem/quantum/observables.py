"""Electronic measurements are read-only; no bond or reaction feedback."""


def atomic_populations(densities, overlap):
    ps = densities[0].add(densities[1]).matmul(overlap)
    return ps.diagonal_values().diagonal(dim1=-2, dim2=-1).sum(-1)


def bond_indices(densities, overlap):
    total = densities[0].add(densities[1]).matmul(overlap)
    keep = total.row < total.col
    i, j = total.row[keep], total.col[keep]
    result = total.values.new_zeros(len(i))
    for p in densities:
        ps = p.matmul(overlap)
        forward = ps.query(i, j)
        reverse = ps.query(j, i).transpose(-1, -2)
        result = result + 2 * (forward * reverse).sum((-1, -2))
    return i, j, result
