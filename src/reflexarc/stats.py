"""Interval estimates for small-n success counts.

Every cell in this project is 20-40 binary rollouts. At that n the normal
approximation to a binomial proportion is wrong in the direction that matters
most here -- it produces intervals that exclude 0 and 1 even when the count is
0/20 or 20/20, which is exactly where these policies live.
"""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """-> (point estimate, lower, upper) at the given z (default 95%)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]].

    Written out rather than imported from scipy because the whole dependency
    is otherwise unused, and because the tables here are tiny.
    """
    from math import comb

    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed * (1 + 1e-9)))
