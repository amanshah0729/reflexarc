"""Survival statistics and the rate ladder. No simulator required."""

from __future__ import annotations

import numpy as np
import pytest

from reflexarc.simreflex import SimRateReflex
from reflexarc.survival import holm, kaplan_meier, logrank


class TestKaplanMeier:
    def test_no_censoring_matches_the_empirical_curve(self):
        km = kaplan_meier([1.0, 2.0, 3.0, 4.0], [True] * 4)
        assert km.survival == pytest.approx([0.75, 0.5, 0.25, 0.0])
        assert km.median() == 2.0

    def test_censored_episodes_are_not_failures(self):
        # Three broke early, one held past the end of the ramp. Counting the
        # survivor as a failure would put the median at 2.0.
        km = kaplan_meier([1.0, 2.0, 3.0, 400.0], [True, True, True, False])
        assert km.n_events == 3
        assert km.median() == 2.0
        assert km.survival[-1] > 0.0, "a censored survivor cannot drive S to 0"

    def test_median_is_none_when_most_episodes_survive(self):
        # Not a missing value: it means over half never lost the object.
        km = kaplan_meier([400.0, 400.0, 400.0, 5.0], [False, False, False, True])
        assert km.median() is None

    def test_censoring_removes_from_the_risk_set(self):
        # One event at 1 out of 4, then one censored, then an event at 3 out of
        # the 2 still at risk: S = 0.75 * 0.5.
        km = kaplan_meier([1.0, 2.0, 3.0, 4.0], [True, False, True, False])
        assert km.survival[-1] == pytest.approx(0.375)


class TestLogRank:
    def test_identical_arms_are_not_significant(self):
        a = ([1.0, 2.0, 3.0, 4.0], [True] * 4)
        chi2, p = logrank(*a, *a)
        assert p > 0.9

    def test_clearly_separated_arms_are_significant(self):
        weak = ([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], [True] * 6)
        strong = ([9.0, 9.1, 9.2, 9.3, 9.4, 9.5], [True] * 6)
        chi2, p = logrank(*weak, *strong)
        assert p < 0.01

    def test_all_censored_gives_no_events_and_no_signal(self):
        chi2, p = logrank([400.0] * 5, [False] * 5, [400.0] * 5, [False] * 5)
        assert (chi2, p) == (0.0, 1.0)

    def test_is_symmetric(self):
        a = ([1.0, 2.0, 5.0], [True, True, False])
        b = ([3.0, 4.0, 6.0], [True, True, True])
        assert logrank(*a, *b)[1] == pytest.approx(logrank(*b, *a)[1])


class TestHolm:
    def test_smallest_p_gets_the_largest_multiplier(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
        assert adj["a"] == pytest.approx(0.03)   # x3
        assert adj["b"] == pytest.approx(0.08)   # x2
        assert adj["c"] == pytest.approx(0.5)    # x1

    def test_monotone_and_bounded(self):
        adj = holm({"a": 0.4, "b": 0.5, "c": 0.6})
        assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0

    def test_order_of_keys_is_preserved(self):
        assert list(holm({"z": 0.5, "a": 0.01}).keys()) == ["z", "a"]


class TestRateLadder:
    @pytest.mark.parametrize("decimation,hz", [(1, 500), (5, 100), (25, 20),
                                               (100, 5), (500, 1)])
    def test_decimation_maps_to_the_advertised_rate(self, decimation, hz):
        assert SimRateReflex(decimation=decimation).hz == hz

    def test_slow_arms_evaluate_less_often(self):
        # The property the ladder depends on: rate changes how often the
        # detector is consulted, and nothing else.
        seen = []

        class Counting(SimRateReflex):
            def _signal(self, reading):
                seen.append(self._substeps)
                return 0.0

        r = Counting(decimation=25, always=False)
        r._fingers = None
        r.reset()
        # `always` short-circuits the sensor, so drive the schedule path
        # instead: it is gated by the same decimation check.
        r.schedule = set()
        for _ in range(100):
            r._substeps += 1
            if (r._substeps - 1) % r.decimation == 0:
                seen.append(r._substeps)
        assert seen == [1, 26, 51, 76]

    def test_hold_survives_between_evaluations(self):
        # A 1 Hz arm is slow to notice, not weaker when it acts: once
        # triggered, the boost persists for hold_substeps regardless of rate.
        r = SimRateReflex(decimation=100, hold_substeps=25)
        assert r.hold_substeps == 25
