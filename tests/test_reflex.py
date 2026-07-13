"""Unit tests for the sensor model and the reflex. No simulator required."""

from __future__ import annotations

import numpy as np
import pytest

from reflexarc.reflex import ScheduledReflex, SlipReflex, yoke
from reflexarc.sense import TactileReading


def held(fn_l=2.0, fn_r=2.0, ft_l=0.5, ft_r=0.5, mu=1.0, pad=0.0) -> TactileReading:
    return TactileReading(
        fn_left=fn_l, fn_right=fn_r, ft_left=ft_l, ft_right=ft_r,
        n_contact_left=2, n_contact_right=2, mu=mu, fn_pad=pad,
    )


class TestTactileReading:
    def test_grip_force_is_the_weaker_pad(self):
        # A two-finger grasp can only squeeze as hard as its weaker contact;
        # summing would hide a finger that has slipped off entirely.
        assert held(fn_l=4.0, fn_r=1.0).fn == 1.0

    def test_cone_ratio(self):
        r = held(fn_l=2.0, fn_r=2.0, ft_l=1.0, ft_r=1.0, mu=1.0)
        assert r.cone_ratio == pytest.approx(0.5)

    def test_friction_coefficient_scales_the_cone(self):
        assert held(mu=2.0).cone_ratio == pytest.approx(held(mu=1.0).cone_ratio / 2)

    def test_no_contact_is_maximally_unstable_not_perfectly_stable(self):
        # The naive guard returns 0.0 when there is no normal force, which
        # reads as an ideal grasp. An object held by nothing is the opposite.
        empty = TactileReading()
        assert empty.cone_ratio == float("inf")
        assert not empty.in_contact

    def test_pad_fraction(self):
        assert held(fn_l=2.0, fn_r=2.0, pad=1.0).pad_fraction == pytest.approx(0.25)
        assert TactileReading().pad_fraction == 0.0


class TestSlipReflex:
    def test_fires_above_threshold_and_arrests_the_arm(self):
        r = SlipReflex(channel="cone_ratio", threshold=0.4, arrest_gain=0.1)
        r.reset()
        action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
        out = r(action, held(ft_l=1.0, ft_r=1.0), step=0)  # cone 0.5 > 0.4
        assert out.fired
        assert np.allclose(out.action[:6], 0.1)
        assert out.action[-1] == 1.0  # forced closed

    def test_does_not_fire_below_threshold(self):
        r = SlipReflex(channel="cone_ratio", threshold=0.9)
        r.reset()
        action = np.ones(7)
        out = r(action, held(ft_l=0.1, ft_r=0.1), step=0)
        assert not out.fired
        assert np.allclose(out.action, action)

    def test_empty_hand_does_not_trigger(self):
        # cone_ratio is inf with no contact, so without the contact guard the
        # reflex fires through the entire approach phase.
        r = SlipReflex(channel="cone_ratio", threshold=1.0)
        r.reset()
        out = r(np.ones(7), TactileReading(), step=0)
        assert not out.fired

    def test_hold_keeps_it_active_after_the_trigger_clears(self):
        r = SlipReflex(channel="cone_ratio", threshold=0.4, hold_steps=3)
        r.reset()
        r(np.ones(7), held(ft_l=1.0, ft_r=1.0), step=0)
        calm = held(ft_l=0.01, ft_r=0.01)
        assert [r(np.ones(7), calm, s).fired for s in (1, 2, 3)] == [True, True, False]

    def test_interrupt_arm_does_not_touch_the_action(self):
        r = SlipReflex(channel="cone_ratio", threshold=0.4, act=False, interrupt=True)
        r.reset()
        action = np.ones(7)
        out = r(action, held(ft_l=1.0, ft_r=1.0), step=0)
        assert out.fired and out.interrupt
        assert np.allclose(out.action, action)

    @pytest.mark.parametrize("channel", ["cone_ratio", "fn", "ft", "pad_fraction",
                                         "d_fn", "d_cone"])
    def test_every_documented_channel_is_reachable(self, channel):
        # A channel named in the analysis but missing from the controller's
        # lookup raises KeyError only once a rollout reaches the reflex arm,
        # i.e. a third of the way into a long sweep.
        r = SlipReflex(channel=channel, threshold=0.0)
        r.reset()
        r(np.ones(7), held(), step=0)

    def test_derivative_channel_needs_two_steps(self):
        r = SlipReflex(channel="d_fn", threshold=-1.0, comparison="lt", hold_steps=1)
        r.reset()
        assert not r(np.ones(7), held(fn_l=4.0, fn_r=4.0), step=0).fired
        # grip force collapses by 3 N in one step
        assert r(np.ones(7), held(fn_l=1.0, fn_r=1.0), step=1).fired

    def test_reset_clears_derivative_history(self):
        r = SlipReflex(channel="d_fn", threshold=-1.0, comparison="lt")
        r.reset()
        r(np.ones(7), held(fn_l=4.0, fn_r=4.0), step=0)
        r.reset()
        assert not r(np.ones(7), held(fn_l=1.0, fn_r=1.0), step=0).fired


class TestYokedControl:
    def test_matches_the_budget_it_is_given(self):
        steps = yoke([1, 2, 3, 4], window=(50, 100), seed=0)
        assert len(steps) == 4
        assert all(50 <= s < 100 for s in steps)

    def test_is_deterministic_in_the_seed(self):
        assert yoke([1, 2, 3], (10, 40), 7) == yoke([1, 2, 3], (10, 40), 7)
        assert yoke([1, 2, 3], (10, 40), 7) != yoke([1, 2, 3], (10, 40), 8)

    def test_never_exceeds_the_window(self):
        assert yoke(list(range(50)), window=(0, 5), seed=0) == {0, 1, 2, 3, 4}
        assert yoke([1], window=(3, 3), seed=0) == set()

    def test_scheduled_reflex_fires_only_on_its_steps(self):
        r = ScheduledReflex(schedule={0: {5, 7}}, arrest_gain=0.5)
        r.bind(0)
        assert not r(np.ones(7), held(), step=4).fired
        out = r(np.ones(7), held(), step=5)
        assert out.fired and np.allclose(out.action[:6], 0.5)

    def test_scheduled_reflex_ignores_the_sensor(self):
        # The whole point of the control: identical behaviour whatever it feels.
        r = ScheduledReflex(schedule={0: {5}})
        r.bind(0)
        assert r(np.ones(7), TactileReading(), step=5).fired
