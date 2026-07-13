"""Seeded randomness that survives leaving the process.

`np.random.default_rng(abs(hash((name, seed))) % 2**32)` is the obvious way to
derive an independent stream per named channel, and it is wrong. Python salts
`str.__hash__` with a per-process seed (PEP 456, on by default since 3.3), so
the same `(name, seed)` produces a different stream in every interpreter.

The failure mode is quiet and specific. Inside one process everything is
consistent, so a determinism test that builds two environments and compares
them passes. Across processes -- which is to say, across a resumed sweep, a
re-run, or anyone else reproducing the work -- the stream changes, and a cell
keyed `(axis, magnitude, seed)` silently samples a different perturbation than
it did yesterday.

Observed here: two runs of an identical impulse cell (8 N, mass x50, seeds
0-11) returned 7/12 and 9/12 success, because the tug went in a different
direction each time.

`blake2b` is stable across processes, versions and machines.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_seed(*parts: object) -> int:
    """A 32-bit seed derived from the arguments, identical in every process."""
    payload = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def stable_rng(*parts: object) -> np.random.Generator:
    """A generator keyed by name and seed, reproducible across processes."""
    return np.random.default_rng(stable_seed(*parts))
