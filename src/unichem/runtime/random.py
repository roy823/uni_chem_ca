from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..errors import StateError

_MASK32 = (1 << 32) - 1


def _mulhilo32(multiplier: int, value: Tensor) -> tuple[Tensor, Tensor]:
    """32x32->64 product using 16-bit limbs; never overflow signed int64."""
    al, ah = multiplier & 65535, multiplier >> 16
    bl, bh = value & 65535, value >> 16
    p0, p1, p2, p3 = al * bl, ah * bl, al * bh, ah * bh
    middle = (p0 >> 16) + (p1 & 65535) + (p2 & 65535)
    low = ((middle & 65535) << 16) | (p0 & 65535)
    high = (p3 + (p1 >> 16) + (p2 >> 16) + (middle >> 16)) & _MASK32
    return high, low


def philox4x32(counter: Tensor, key: tuple[int, int], *, rounds: int = 10) -> Tensor:
    """Independent vectorized Philox-4x32 rounds (Salmon et al., SC11).

    Integer outputs follow the published algorithm; standard known-answer
    vectors are tested. This implementation has not itself run TestU01.
    No mutable RNG state, global torch seed changes, or sequential cell loop.
    """
    if counter.dtype != torch.int64 or counter.shape[-1] != 4 or rounds < 1:
        raise StateError("Philox requires (...,4) int64 counters and positive rounds")
    c0, c1, c2, c3 = (counter[..., k] & _MASK32 for k in range(4))
    k0, k1 = key[0] & _MASK32, key[1] & _MASK32
    for _ in range(rounds):
        hi0, lo0 = _mulhilo32(0xD2511F53, c0)
        hi1, lo1 = _mulhilo32(0xCD9E8D57, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = (k0 + 0x9E3779B9) & _MASK32, (k1 + 0xBB67AE85) & _MASK32
    return torch.stack((c0, c1, c2, c3), -1)


@dataclass(frozen=True)
class CounterNoise:
    """Noise keyed by (seed, step, phase, persistent entity id, component).

    Reordering/chunking entities cannot change their draws. Checkpoint replay
    needs the same seed, phase names, world step, ids, integrator and weights.
    Floating transcendentals/reductions are not promised bitwise CPU/GPU equal.
    """

    seed: int = 0

    def normal(
        self,
        ids: Tensor,
        components: int,
        *,
        step: int,
        phase: str,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        if ids.ndim != 1 or ids.dtype != torch.int64 or components < 1:
            raise StateError("Noise expects int64 entity ids and a positive component count")
        if not 0 <= step < 2**32 or not 0 <= self.seed < 2**64 or components >= 2**32:
            raise StateError("Counter range exceeded; change stream namespace explicitly")
        if dtype not in (torch.float32, torch.float64):
            raise StateError("Noise output supports float32 or float64")
        stream = int.from_bytes(hashlib.blake2b(phase.encode(), digest_size=8).digest(), "little")
        key64 = self.seed ^ stream
        index = torch.arange(components, device=ids.device, dtype=torch.int64)
        c = torch.stack(
            (
                ids[:, None].expand(-1, components) & _MASK32,
                (ids[:, None].expand(-1, components) >> 32) & _MASK32,
                index[None, :].expand(len(ids), -1),
                torch.full((len(ids), components), step, device=ids.device, dtype=torch.int64),
            ),
            -1,
        )
        words = philox4x32(c, (key64 & _MASK32, key64 >> 32))
        # Compute uniform variates in float64 before casting to avoid float32
        # rounding an endpoint to exactly 0 or 1 in Box-Muller.
        u = (words[..., :2].to(torch.float64) + 0.5) / 2**32
        return (torch.sqrt(-2 * torch.log(u[..., 0])) * torch.cos(2 * math.pi * u[..., 1])).to(
            dtype
        )
