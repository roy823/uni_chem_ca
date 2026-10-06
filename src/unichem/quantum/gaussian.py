"""All-electron Gaussian HF operators and analytic forces in atomic blocks.

PySCF/libcint supplies integrals only. This provider never calls its SCF,
orbital diagonalization or global density-matrix routines. Quartet integrals
are bounded cached atom blocks; electronic states belong to our sparse core.
"""

from collections import OrderedDict
from itertools import product

import numpy as np
import torch

from ..errors import CapacityError, StateError
from .blocks import BlockMatrix
from .operators import Operators


class GaussianIntegrals:
    def __init__(
        self,
        numbers,
        positions,
        *,
        basis="sto-3g",
        charge=0,
        spin=0,
        block_budget=100_000,
        quartet_budget=50_000,
        external_field=(0.0, 0.0, 0.0),
    ):
        from pyscf import gto

        self.positions = np.array(positions, dtype=float, copy=True)
        self.external_field = np.asarray(external_field, dtype=float)
        self.n = len(numbers)
        if self.n**4 > quartet_budget:
            raise CapacityError(
                "Exact Gaussian quartet budget exceeded; a screened long-range provider is required"
            )
        self.mol = gto.M(
            atom=list(zip([int(z) for z in numbers], self.positions.tolist())),
            unit="Bohr",
            basis=basis,
            charge=int(charge),
            spin=int(spin),
            verbose=0,
        )
        self.slices = self.mol.aoslice_by_atom()
        self.counts = self.slices[:, 3] - self.slices[:, 2]
        self.width = int(self.counts.max())
        self.active = torch.arange(self.width)[None, :] < torch.tensor(self.counts)[:, None]
        self.budget = block_budget
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = 128 * 1024**2
        self.overlap = self.matrix("int1e_ovlp")
        self.core = self.matrix("int1e_kin").add(self.matrix("int1e_nuc")).symmetric()
        if np.any(self.external_field):
            rows = []
            cols = []
            blocks = []
            for i, j in product(range(self.n), repeat=2):
                value = np.einsum(
                    "x,xab->ab", self.external_field, self.integral("int1e_r", (i, j), comp=3)
                )
                block = torch.zeros((self.width, self.width), dtype=torch.float64)
                block[: self.counts[i], : self.counts[j]] = torch.tensor(value)
                rows.append(i)
                cols.append(j)
                blocks.append(block)
            self.core = self.core.add(
                BlockMatrix.assemble(
                    torch.tensor(rows), torch.tensor(cols), torch.stack(blocks), self.n, self.budget
                )
            ).symmetric()

    def integral(self, name, atoms, *, comp=1):
        key = (name, tuple(atoms), comp)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        shape = tuple(int(self.counts[i]) for i in atoms)
        expected = int(np.prod(shape)) * comp * 8
        if expected > self._cache_limit:
            raise CapacityError("A single AO quartet exceeds the integral memory budget")
        shells = tuple(int(x) for atom in atoms for x in self.slices[atom, :2])
        value = self.mol.intor(name, comp=comp, shls_slice=shells, aosym="s1")
        value = np.asarray(value).reshape(((comp,) if comp > 1 else ()) + shape)
        while self._cache and self._cache_bytes + value.nbytes > self._cache_limit:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
        self._cache[key] = value
        self._cache_bytes += value.nbytes
        return value

    def matrix(self, name):
        rows = []
        cols = []
        blocks = []
        for i, j in product(range(self.n), repeat=2):
            value = self.integral(name, (i, j))
            if not np.any(value):
                continue
            block = torch.zeros((self.width, self.width), dtype=torch.float64)
            block[: self.counts[i], : self.counts[j]] = torch.tensor(value)
            rows.append(i)
            cols.append(j)
            blocks.append(block)
        values = (
            torch.stack(blocks)
            if blocks
            else torch.empty((0, self.width, self.width), dtype=torch.float64)
        )
        return BlockMatrix.assemble(
            torch.tensor(rows, dtype=torch.int64),
            torch.tensor(cols, dtype=torch.int64),
            values,
            self.n,
            self.budget,
        )

    def _density_blocks(self, density):
        return {
            (int(i), int(j)): b[: self.counts[int(i)], : self.counts[int(j)]].detach().cpu().numpy()
            for i, j, b in zip(density.row, density.col, density.values)
        }

    def coulomb_exchange(self, densities):
        ps = [self._density_blocks(p) for p in densities]
        total = self._density_blocks(densities[0].add(densities[1]))
        rows = []
        cols = []
        jb = []
        kb = [[], []]
        for i, j in product(range(self.n), repeat=2):
            shape = (int(self.counts[i]), int(self.counts[j]))
            coulomb = np.zeros(shape)
            exchange = [np.zeros(shape), np.zeros(shape)]
            for (k, l), p in total.items():
                if np.any(p):
                    coulomb += np.einsum("abcd,cd->ab", self.integral("int2e", (i, j, k, l)), p)
            for spin in range(2):
                for (k, l), p in ps[spin].items():
                    if np.any(p):
                        exchange[spin] += np.einsum(
                            "acbd,cd->ab", self.integral("int2e", (i, k, j, l)), p
                        )
            rows.append(i)
            cols.append(j)
            for target, value in zip((jb, *kb), (coulomb, *exchange)):
                padded = torch.zeros((self.width, self.width), dtype=torch.float64)
                padded[: self.counts[i], : self.counts[j]] = torch.tensor(value)
                target.append(padded)

        def assemble(values):
            return BlockMatrix.assemble(
                torch.tensor(rows), torch.tensor(cols), torch.stack(values), self.n, self.budget
            )

        return assemble(jb), tuple(assemble(values) for values in kb)

    def operators(self):
        return Operators(
            self.overlap,
            self.core,
            self.active,
            float(self.mol.energy_nuc())
            - float(
                np.einsum("i,ix,x->", self.mol.atom_charges(), self.positions, self.external_field)
            ),
            self.coulomb_exchange,
        )

    def forces(self, solution):
        """HF variational gradients, including energy-weighted-density Pulay."""
        if not solution.converged:
            raise StateError("Nuclear forces require converged electronic states")
        densities = solution.densities
        ps = [self._density_blocks(p) for p in densities]
        total = self._density_blocks(densities[0].add(densities[1]))
        weighted = densities[0].matmul(solution.focks[0]).matmul(densities[0])
        weighted = weighted.add(densities[1].matmul(solution.focks[1]).matmul(densities[1]))
        weights = self._density_blocks(weighted)
        zeros = lambda i, j: np.zeros((int(self.counts[i]), int(self.counts[j])))
        get = lambda blocks, i, j: blocks.get((i, j), zeros(i, j))
        gradient = np.zeros((self.n, 3))
        for atom in range(self.n):
            # The nucleus-dependent attraction operator is differentiated at
            # its own centre; bra/ket translations are separate contributions.
            with self.mol.with_rinv_at_nucleus(atom):
                for i, j in product(range(self.n), repeat=2):
                    h1 = -self.mol.atom_charge(atom) * (
                        self.mol.intor(
                            "int1e_iprinv",
                            comp=3,
                            shls_slice=tuple(int(x) for a in (i, j) for x in self.slices[a, :2]),
                        )
                        + self.mol.intor(
                            "int1e_iprinv",
                            comp=3,
                            shls_slice=tuple(int(x) for a in (j, i) for x in self.slices[a, :2]),
                        ).transpose(0, 2, 1)
                    )
                    s1 = np.zeros_like(h1)
                    if i == atom:
                        h1 -= self.integral("int1e_ipkin", (i, j), comp=3) + self.integral(
                            "int1e_ipnuc", (i, j), comp=3
                        )
                        s1 -= self.integral("int1e_ipovlp", (i, j), comp=3)
                        if np.any(self.external_field):
                            irp = self.integral("int1e_irp", (j, i), comp=9).reshape(
                                3, 3, self.counts[j], self.counts[i]
                            )
                            h1 -= np.einsum("yxba,y->xab", irp, self.external_field)
                    if j == atom:
                        h1 -= self.integral("int1e_ipkin", (j, i), comp=3).transpose(
                            0, 2, 1
                        ) + self.integral("int1e_ipnuc", (j, i), comp=3).transpose(0, 2, 1)
                        s1 -= self.integral("int1e_ipovlp", (j, i), comp=3).transpose(0, 2, 1)
                        if np.any(self.external_field):
                            irp = self.integral("int1e_irp", (i, j), comp=9).reshape(
                                3, 3, self.counts[i], self.counts[j]
                            )
                            h1 -= np.einsum("yxab,y->xab", irp, self.external_field)
                    gradient[atom] += np.einsum("xab,ab->x", h1, get(total, i, j)) - np.einsum(
                        "xab,ab->x", s1, get(weights, i, j)
                    )
        for i, j, k, l in product(range(self.n), repeat=4):
            coefficient = 0.5 * np.einsum("ab,cd->abcd", get(total, i, j), get(total, k, l))
            for spin in range(2):
                coefficient -= 0.5 * np.einsum(
                    "ac,bd->abcd", get(ps[spin], i, k), get(ps[spin], j, l)
                )
            if not np.any(coefficient):
                continue
            for atom, atoms, axes in (
                (i, (i, j, k, l), (0, 1, 2, 3, 4)),
                (j, (j, i, k, l), (0, 2, 1, 3, 4)),
                (k, (k, l, i, j), (0, 3, 4, 1, 2)),
                (l, (l, k, i, j), (0, 3, 4, 2, 1)),
            ):
                derivative = self.integral("int2e_ip1", atoms, comp=3).transpose(axes)
                gradient[atom] -= np.einsum("xabcd,abcd->x", derivative, coefficient)
        z = self.mol.atom_charges()
        gradient -= z[:, None] * self.external_field
        for i in range(self.n):
            for j in range(i):
                delta = self.positions[i] - self.positions[j]
                force = z[i] * z[j] * delta / np.linalg.norm(delta) ** 3
                gradient[i] -= force
                gradient[j] += force
        return torch.tensor(-gradient, dtype=torch.float64)
