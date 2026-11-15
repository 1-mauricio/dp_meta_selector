"""Random sources for reproducible public benchmarks and private deployments."""

from __future__ import annotations

import secrets

import numpy as np
from scipy.special import ndtri


class SystemRandomGenerator:
    """Vectorized inverse-transform sampler backed by OS cryptographic entropy.

    Finite-precision sampling still needs a mechanism-specific audit before a
    production privacy claim. The public OpenML benchmark instead injects
    explicit deterministic seeds for reproducibility.
    """

    @staticmethod
    def random(size=None):
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        count = int(np.prod(shape)) if shape else 1
        words = np.frombuffer(secrets.token_bytes(count * 8), dtype=np.uint64)
        # Midpoints of 53-bit bins avoid exact 0 and 1 for inverse transforms.
        values = ((words >> 11).astype(np.float64) + 0.5) / float(2**53)
        if not shape:
            return float(values[0])
        return values.reshape(shape)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if scale < 0:
            raise ValueError("scale precisa ser não negativo")
        return loc + scale * ndtri(self.random(size))

    def laplace(self, loc=0.0, scale=1.0, size=None):
        if scale < 0:
            raise ValueError("scale precisa ser não negativo")
        uniform = self.random(size) - 0.5
        noise = -scale * np.sign(uniform) * np.log1p(-2.0 * np.abs(uniform))
        return loc + noise

    def geometric(self, p, size=None):
        if not 0 < p <= 1:
            raise ValueError("p precisa estar em (0, 1]")
        if p == 1:
            return np.ones(size, dtype=int) if size is not None else 1
        values = np.floor(np.log1p(-self.random(size)) / np.log1p(-p)).astype(int) + 1
        return values


def make_rng(seed: int | None):
    """Use a reproducible PRNG for tests/benchmarks or OS entropy when None."""
    return np.random.default_rng(seed) if seed is not None else SystemRandomGenerator()
