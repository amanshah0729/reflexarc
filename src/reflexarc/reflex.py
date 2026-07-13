"""The fast loop: a controller that sits below the policy and re-decides every step.

What a reflex is allowed to do here is constrained by LIBERO's action space,
and the constraint is worth stating because it rules out the textbook response.
`PandaGripper.format_action` maps the gripper command to a binary open/close,
so "detect slip, increase grip force" -- the canonical tactile reflex, and the
thing real slip controllers do -- is not expressible. The gripper is either
closed or it is not.

That leaves three channels, all of which act on the arm rather than the hand:

  arrest    scale down the commanded end-effector delta. Slip during transport
            is largely inertial loading, so decelerating reduces the tangential
            force at the contact directly. This is the primary channel.
  close     force the gripper command closed, which matters only when the
            policy was mid-release.
  interrupt discard the action chunk and force the policy to re-infer now.
            This channel spends no authority of its own; it just buys the
            policy an earlier look at the world.

`interrupt` is what separates "a reflex helped" from "reacting sooner helped",
because it makes the fast loop a trigger for cognition rather than a controller
in its own right. Running it as a separate arm is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reflexarc.rng import stable_rng
from reflexarc.runner import ReflexOutput
from reflexarc.sense import TactileReading


@dataclass
class SlipReflex:
    """Fires when a chosen fingertip signal crosses a threshold.

    `act` and `interrupt` are independent so the same detector can drive an
    arm that only slows the robot, an arm that only re-plans, or both.
    """

    channel: str = "cone_ratio"   # cone_ratio | fn | ft | d_fn | d_cone
    threshold: float = 1.0
    comparison: str = "gt"        # gt | lt
    arrest_gain: float = 0.2      # multiplier on the commanded EE delta
    close_gripper: bool = True
    hold_steps: int = 5           # stay active this long after a trigger
    act: bool = True              # modify the action
    interrupt: bool = False       # force the policy to re-infer
    require_contact: bool = True  # ignore the signal when nothing is held
    always: bool = False          # ignore the sensor entirely and stay active
    # If set, trigger on these steps instead of on the sensor. The timing-
    # matched control for a positive result: same intervention, same number of
    # activations, times chosen without looking. `always` fails a positive
    # result for two reasons at once -- wrong timing and continuous
    # application -- and cannot separate them.
    schedule: set[int] | None = None

    _hold: int = field(default=0, init=False, repr=False)
    _prev: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def reset(self) -> None:
        self._hold = 0
        self._prev = {}

    def describe(self) -> str:
        what = []
        if self.act:
            what.append(f"arrest x{self.arrest_gain:g}")
        if self.interrupt:
            what.append("replan")
        return f"{self.channel} {self.comparison} {self.threshold:g} -> {'+'.join(what)}"

    def signal(self, reading: TactileReading) -> float:
        """Current value of the chosen channel. Derivatives use the last step.

        Only physically-measurable channels are reachable here; the oracle
        fields on the reading are deliberately not exposed.
        """
        cone = min(reading.cone_ratio, 1e3)
        now = {"cone_ratio": cone, "fn": reading.fn, "ft": reading.ft,
               "pad_fraction": reading.pad_fraction}
        if self.channel in ("d_fn", "d_cone"):
            base = "fn" if self.channel == "d_fn" else "cone_ratio"
            prev = self._prev.get(base, now[base])
            value = now[base] - prev
        else:
            value = now[self.channel]
        self._prev = now
        return float(value)

    def triggered(self, reading: TactileReading) -> bool:
        if self.always:
            # The control for any positive result: same intervention, same
            # authority, no sensor. If this matches the sensed arm, the signal
            # contributed nothing and the finding is about the actuator.
            self.signal(reading)
            return True
        if self.require_contact and not reading.in_contact:
            # An empty hand has no slip to detect. Without this the cone ratio
            # reads inf whenever the gripper is open and the reflex fires for
            # the whole approach.
            self.signal(reading)
            return False
        s = self.signal(reading)
        return s > self.threshold if self.comparison == "gt" else s < self.threshold

    def _apply(self, action: np.ndarray) -> np.ndarray:
        out = np.array(action, dtype=float, copy=True)
        out[:6] *= self.arrest_gain
        if self.close_gripper:
            out[-1] = 1.0
        return out

    def __call__(self, action: np.ndarray, reading: TactileReading,
                 step: int) -> ReflexOutput:
        if self.schedule is not None:
            self.signal(reading)   # keep derivative history consistent
            trig = step in self.schedule
        else:
            trig = self.triggered(reading)
        if trig:
            self._hold = self.hold_steps
        active = self._hold > 0
        if self._hold > 0:
            self._hold -= 1
        if not active:
            return ReflexOutput(action, fired=False, interrupt=False)
        return ReflexOutput(
            self._apply(action) if self.act else action,
            fired=True,
            interrupt=self.interrupt,
        )


@dataclass
class SqueezeReflex(SlipReflex):
    """The response the action space cannot express: squeeze harder.

    `PandaGripper.format_action` collapses the gripper command to binary, so a
    policy or a reflex can only ask for "closed", never for "closed harder".
    The grip force that results is set by the finger position servo: measured
    on this task it is 5-8 N against an actuator limit of 20 N, so the hand is
    not saturated -- the authority exists in the hardware and is unreachable
    through the interface.

    This arm reaches past the interface and raises the servo gain directly
    while the reflex is active. It is deliberately not a fair policy: it is the
    control that separates "the reflex had no useful signal" from "the reflex
    had a signal and no way to act on it". If squeezing recovers drops that
    arm-arrest cannot, the limiting factor is the action space rather than the
    sensing, and that is a claim about robot interfaces rather than about
    policies.

    MuJoCo position actuators keep kp in `gainprm[0]` and `-kp` in
    `biasprm[1]`; changing one without the other does not stiffen the servo, it
    unbalances it into a constant-force actuator.
    """

    force_gain: float = 4.0

    _ids: tuple[int, ...] = field(default_factory=tuple, init=False, repr=False)
    _base: dict[int, tuple] = field(default_factory=dict, init=False, repr=False)
    _boosted: bool = field(default=False, init=False, repr=False)

    def describe(self) -> str:
        return f"{super().describe()} + grip x{self.force_gain:g}"

    def _resolve(self, sim) -> None:
        import mujoco

        from reflexarc.sense import _unwrap

        model, _ = _unwrap(sim)
        ids = []
        for a in range(int(model.nu)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            if "gripper" in name and "finger" in name:
                ids.append(a)
        if not ids:
            raise RuntimeError("no gripper actuators found; squeeze arm cannot run")
        self._ids = tuple(ids)
        self._base = {
            a: (float(model.actuator_gainprm[a][0]),
                float(model.actuator_biasprm[a][1]),
                np.array(model.actuator_forcerange[a], copy=True))
            for a in ids
        }

    def apply_sim(self, sim, active: bool) -> None:
        from reflexarc.sense import _unwrap

        if not self._ids:
            self._resolve(sim)
        if active == self._boosted:
            return
        model, _ = _unwrap(sim)
        for a in self._ids:
            kp, bias, frange = self._base[a]
            k = self.force_gain if active else 1.0
            model.actuator_gainprm[a][0] = kp * k
            model.actuator_biasprm[a][1] = bias * k
            model.actuator_forcerange[a][:] = frange * k
        self._boosted = active

    def reset(self) -> None:
        super().reset()
        self._boosted = False


@dataclass
class ScheduledReflex:
    """Yoked control: the same intervention, at times chosen without the sensor.

    The obvious control for "the reflex helped" is an arm that always slows
    down, but that changes two things at once -- whether to intervene and how
    much total intervention there is -- and a robot that crawls through every
    episode fails on the step limit for reasons unrelated to grasping.

    This arm instead replays a *matched budget*: for each seed it intervenes
    for the same number of steps the tactile reflex used on that seed, placed
    at random inside the same window. If it recovers as many drops as the
    tactile arm, the benefit was slowing down, not sensing.
    """

    schedule: dict[int, set[int]] = field(default_factory=dict)  # seed -> steps
    arrest_gain: float = 0.2
    close_gripper: bool = True
    seed: int = 0

    _steps: set[int] = field(default_factory=set, init=False, repr=False)

    def bind(self, seed: int) -> None:
        self.seed = seed
        self._steps = set(self.schedule.get(seed, set()))

    def reset(self) -> None:
        self._steps = set(self.schedule.get(self.seed, set()))

    def describe(self) -> str:
        return f"yoked arrest x{self.arrest_gain:g}"

    def __call__(self, action: np.ndarray, reading: TactileReading,
                 step: int) -> ReflexOutput:
        if step not in self._steps:
            return ReflexOutput(action, fired=False, interrupt=False)
        out = np.array(action, dtype=float, copy=True)
        out[:6] *= self.arrest_gain
        if self.close_gripper:
            out[-1] = 1.0
        return ReflexOutput(out, fired=True, interrupt=False)


def yoke(active_steps: list[int], window: tuple[int, int], seed: int) -> set[int]:
    """Place `len(active_steps)` interventions at random inside `window`."""
    lo, hi = window
    n = min(len(active_steps), max(hi - lo, 0))
    if n == 0:
        return set()
    rng = stable_rng("yoke", seed)
    return set(int(x) for x in rng.choice(np.arange(lo, hi), size=n, replace=False))
