"""Kaplan-Meier and log-rank, for outcomes that are a load rather than a bit.

The experiments before this one asked "at load L, did the object drop?", which
is one bit per rollout and needs L calibrated into a narrow band or it reads as
all-drop or all-hold. This asks "at what load did it drop?", which is a number,
and needs no calibration because each episode sweeps the range.

Episodes that never drop are the reason a plain mean will not do. They are
**right-censored** -- the breaking load is known only to exceed the ramp
maximum -- and there are three wrong ways to handle them: discard them (drops
the best-performing episodes and biases every arm toward its failures), score
them at the maximum (biases the other way, and understates any arm that would
have held much longer), or count them as failures (throws away the distinction
the experiment exists to measure). Survival analysis is the tool that uses them
without choosing one of those.

No scipy: the whole dependency would be pulled in for two short functions and a
chi-square tail that has a closed form at one degree of freedom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KM:
    """A Kaplan-Meier curve and the quantities read off it."""

    times: list[float]
    survival: list[float]
    n: int
    n_events: int

    def median(self) -> float | None:
        """Load at which survival first drops to 0.5, or None if it never does.

        None is a real answer, not a missing value: it means more than half the
        episodes never lost the object inside the ramp.
        """
        for t, s in zip(self.times, self.survival):
            if s <= 0.5:
                return t
        return None

    def quantile(self, q: float) -> float | None:
        for t, s in zip(self.times, self.survival):
            if s <= 1.0 - q:
                return t
        return None


def kaplan_meier(loads: list[float], observed: list[bool]) -> KM:
    """`loads[i]` is the breaking load, or the censoring load if not observed."""
    pairs = sorted(zip(loads, observed), key=lambda p: (p[0], not p[1]))
    n_at_risk = len(pairs)
    s = 1.0
    times: list[float] = []
    surv: list[float] = []
    i = 0
    n_events = 0
    while i < len(pairs):
        t = pairs[i][0]
        # Everything at this exact load: events and censorings together.
        d = sum(1 for p in pairs[i:] if p[0] == t and p[1])
        c = sum(1 for p in pairs[i:] if p[0] == t and not p[1])
        if d > 0 and n_at_risk > 0:
            s *= 1.0 - d / n_at_risk
            n_events += d
            times.append(t)
            surv.append(s)
        n_at_risk -= d + c
        i += d + c
    return KM(times, surv, len(pairs), n_events)


def _chi2_sf_1df(x: float) -> float:
    """P(X > x) for chi-square with 1 degree of freedom. Closed form."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def logrank(loads_a: list[float], obs_a: list[bool],
            loads_b: list[float], obs_b: list[bool]) -> tuple[float, float]:
    """Two-sided log-rank test between two arms. -> (chi2, p)."""
    events = sorted({l for l, o in zip(loads_a, obs_a) if o}
                    | {l for l, o in zip(loads_b, obs_b) if o})
    if not events:
        return 0.0, 1.0

    o_minus_e = 0.0
    var = 0.0
    for t in events:
        n_a = sum(1 for l in loads_a if l >= t)
        n_b = sum(1 for l in loads_b if l >= t)
        n = n_a + n_b
        if n < 2:
            continue
        d_a = sum(1 for l, o in zip(loads_a, obs_a) if o and l == t)
        d_b = sum(1 for l, o in zip(loads_b, obs_b) if o and l == t)
        d = d_a + d_b
        if d == 0:
            continue
        e_a = d * n_a / n
        o_minus_e += d_a - e_a
        # Hypergeometric variance, with the finite-population correction that
        # matters at these sample sizes.
        var += (d * (n_a / n) * (n_b / n) * (n - d)) / (n - 1)
    if var <= 0:
        return 0.0, 1.0
    chi2 = (o_minus_e ** 2) / var
    return chi2, _chi2_sf_1df(chi2)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, order preserved in the output."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return {k: adjusted[k] for k in pvalues}
