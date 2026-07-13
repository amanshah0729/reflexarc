"""Reproducibility of seeded randomness, checked across process boundaries.

This file exists because the bug it guards cannot be caught in-process. Python
salts `str.__hash__` per interpreter, so `hash(("impulse", 0))` is stable for
the lifetime of one process and different in the next. Any determinism test
that builds two objects and compares them therefore passes while the property
it claims to check is false.

The only test that catches it runs a second interpreter.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from reflexarc.disturb import Impulse
from reflexarc.reflex import yoke
from reflexarc.rng import stable_rng, stable_seed


def _in_subprocess(expr: str) -> str:
    """Evaluate `expr` in a fresh interpreter, with hash salting left ON."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src')\n"
         "from reflexarc.rng import stable_seed\n"
         "from reflexarc.disturb import Impulse\n"
         "from reflexarc.reflex import yoke\n"
         f"print({expr})"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


class TestStableSeed:
    def test_deterministic_in_process(self):
        assert stable_seed("impulse", 3) == stable_seed("impulse", 3)

    def test_distinct_streams_per_name(self):
        assert stable_seed("impulse", 3) != stable_seed("yoke", 3)
        assert stable_seed("impulse", 3) != stable_seed("impulse", 4)

    def test_survives_a_new_interpreter(self):
        assert _in_subprocess('stable_seed("impulse", 3)') == str(stable_seed("impulse", 3))

    def test_builtin_hash_does_not(self):
        # Documents the bug rather than the fix: if this ever starts passing,
        # CPython changed and the guard above is no longer load-bearing.
        got = subprocess.run(
            [sys.executable, "-c", 'print(hash(("impulse", 3)))'],
            capture_output=True, text=True, check=True).stdout.strip()
        assert got != str(hash(("impulse", 3)))

    def test_generator_stream_is_stable(self):
        a = stable_rng("x", 1).normal(size=5)
        b = stable_rng("x", 1).normal(size=5)
        assert np.allclose(a, b)


class TestPerturbationsAreReproducible:
    def test_impulse_direction_survives_a_new_interpreter(self):
        # The concrete failure: two runs of the same 8 N cell tugged the object
        # in different directions and returned 7/12 and 9/12 success.
        here = np.round(Impulse(magnitude=8.0, seed=5).unit(), 10).tolist()
        there = _in_subprocess("[round(x, 10) for x in "
                               "Impulse(magnitude=8.0, seed=5).unit().tolist()]")
        assert str(here) == there

    def test_yoked_schedule_survives_a_new_interpreter(self):
        here = sorted(yoke([1, 2, 3, 4], (50, 100), 7))
        there = _in_subprocess("sorted(yoke([1, 2, 3, 4], (50, 100), 7))")
        assert str(here) == there
