"""A reflex that actually runs faster than the policy.

Everything in `reflex.py` re-decides once per control step. That is 20 Hz --
the same rate the policy's actions are consumed at. Its only advantage over the
policy is re-deciding *within* a chunk, which is a factor of `n_action_steps`,
not a factor of anything physical. The biological claim being tested here is
stronger than that: a spinal reflex closes its loop far below the rate at which
decisions are made.

robosuite makes the stronger version reachable. `MujocoEnv.step` runs

    for i in range(int(control_timestep / model_timestep)):
        self.sim.forward()
        self._pre_action(action, policy_step)
        self.sim.step()

which on LIBERO is 25 substeps of 2 ms per 50 ms control step. `sim.forward()`
has already run when `_pre_action` is called, so contact forces are current.
Wrapping it gives a controller at **500 Hz against a policy at 20 Hz**, and
against a decision that is frozen for a full second at the default chunk
length.

The response channel is the finger servo gain rather than the action vector,
because F4 established that the action vector has no grip-force channel at all:
`PandaGripper.format_action` collapses the gripper command to binary. A reflex
at any rate that can only push on the arm has nothing useful to say about a
grasp.

This deliberately reaches past the action space. It is a measurement of what a
fast loop *could* do given an interface that exposed grip force, not a proposal
for a policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reflexarc.sense import FingerGeoms, _unwrap, read


@dataclass
class SimRateReflex:
    """Grip-force reflex closed inside the physics loop.

    `hold_substeps` is in 2 ms substeps, so the default of 25 is one control
    step -- long enough that a single detection survives to the next policy
    action, short enough that it is not a permanent change to the gripper.
    """

    channel: str = "cone_ratio"
    threshold: float = 0.9
    comparison: str = "gt"
    force_gain: float = 6.0
    hold_substeps: int = 25
    require_contact: bool = True
    always: bool = False
    # Evaluate once every `decimation` physics substeps. The substep is 2 ms,
    # so 1 -> 500 Hz, 5 -> 100 Hz, 25 -> 20 Hz (the policy's own action rate),
    # 100 -> 5 Hz, 500 -> 1 Hz. One implementation across the whole ladder, so
    # a rate comparison varies rate and nothing else -- comparing this class at
    # 500 Hz against `SlipReflex` at 20 Hz would confound rate with two
    # different controllers.
    decimation: int = 1
    # Fire on these substeps instead of on the sensor: the timing-matched
    # control, with a budget copied from a sensed run.
    schedule: set[int] | None = None
    # Fire on ground truth instead of on the fingertip signal: an upper bound
    # on what any detector could deliver. `oracle_load` is set per step by the
    # runner from the disturbance actually in force.
    oracle_threshold: float | None = None

    # --- runtime
    _fired: int = field(default=0, init=False, repr=False)
    _triggers: int = field(default=0, init=False, repr=False)
    _substeps: int = field(default=0, init=False, repr=False)
    _first_at: int | None = field(default=None, init=False, repr=False)
    _hold: int = field(default=0, init=False, repr=False)
    _boosted: bool = field(default=False, init=False, repr=False)
    oracle_load: float = field(default=0.0, init=False, repr=False)
    _ids: tuple[int, ...] = field(default_factory=tuple, init=False, repr=False)
    _base: dict = field(default_factory=dict, init=False, repr=False)
    _orig = None
    _env = None

    @property
    def hz(self) -> float:
        return 500.0 / max(self.decimation, 1)

    def describe(self) -> str:
        if self.schedule is not None:
            rate = "yoked"
        elif self.oracle_threshold is not None:
            rate = f"ORACLE load > {self.oracle_threshold:g}"
        elif self.always:
            rate = "always"
        else:
            rate = f"{self.channel} {self.comparison} {self.threshold:g}"
        return f"{self.hz:g}Hz [{rate}] -> grip x{self.force_gain:g}"

    # -- statistics ---------------------------------------------------------

    @property
    def fired_substeps(self) -> int:
        return self._fired

    @property
    def triggers(self) -> int:
        """Distinct detections, not substeps held."""
        return self._triggers

    @property
    def first_substep(self) -> int | None:
        return self._first_at

    @property
    def substeps_seen(self) -> int:
        return self._substeps

    # -- actuator -----------------------------------------------------------

    def _resolve(self, sim) -> None:
        import mujoco

        model, _ = _unwrap(sim)
        ids = []
        for a in range(int(model.nu)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            if "gripper" in name and "finger" in name:
                ids.append(a)
        if not ids:
            raise RuntimeError("no gripper actuators; sim-rate reflex cannot run")
        self._ids = tuple(ids)
        self._base = {
            a: (float(model.actuator_gainprm[a][0]),
                float(model.actuator_biasprm[a][1]),
                np.array(model.actuator_forcerange[a], copy=True))
            for a in ids
        }

    def _set_boost(self, sim, active: bool) -> None:
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

    # -- lifecycle ----------------------------------------------------------

    def attach(self, robosuite_env, fingers: FingerGeoms) -> None:
        """Wrap `_pre_action` so the reflex runs on every physics substep."""
        self.reset()
        # An episode that raised mid-rollout never reached its detach, and a
        # leaked wrapper would keep firing through every later trial in the
        # sweep -- silently, since it only changes actuator gains. Restoring
        # any previous wrapper first makes that self-healing, and wrapping a
        # wrapper (which is what would otherwise happen) unreachable.
        prev = getattr(robosuite_env, "_reflexarc_orig", None)
        if prev is not None:
            robosuite_env._pre_action = prev
        self._env = robosuite_env
        self._fingers = fingers
        self._orig = robosuite_env._pre_action
        robosuite_env._reflexarc_orig = self._orig
        self._resolve(robosuite_env.sim)

        # Reads `self._env.sim` per substep rather than closing over the object
        # captured here: robosuite replaces its MjSim across resets, and a
        # captured handle goes stale mid-episode (observed as `data` being None
        # inside the physics loop).
        def wrapped(action, policy_step=False, _orig=self._orig):
            _orig(action, policy_step)
            self._substep(self._env.sim)

        robosuite_env._pre_action = wrapped

    def detach(self) -> None:
        if self._env is not None and self._orig is not None:
            # Restore the servo before unwrapping, or the boost leaks into the
            # next episode -- reset() recompiles from XML, but relying on that
            # would make this class silently order-dependent.
            try:
                self._set_boost(self._env.sim, False)
            except Exception:
                pass
            self._env._pre_action = self._orig
            try:
                del self._env._reflexarc_orig
            except AttributeError:
                pass
        self._env = self._orig = None

    def reset(self) -> None:
        self._fired = self._triggers = self._substeps = 0
        self._first_at = None
        self._hold = 0
        self._boosted = False

    # -- the loop -----------------------------------------------------------

    def _signal(self, reading) -> float:
        if self.channel == "cone_ratio":
            return min(reading.cone_ratio, 1e3)
        if self.channel == "pad_fraction":
            return reading.pad_fraction
        return getattr(reading, self.channel)

    def _substep(self, sim) -> None:
        self._substeps += 1
        # Decimation gates evaluation, not the hold: once triggered the boost
        # stays applied for `hold_substeps` regardless of rate, so a slow arm
        # is slow to *notice*, not weaker when it acts.
        if (self._substeps - 1) % max(self.decimation, 1) != 0:
            active = self._hold > 0
            if self._hold > 0:
                self._hold -= 1
                self._fired += 1
            self._set_boost(sim, active)
            return
        if self.schedule is not None:
            trig = self._substeps in self.schedule
        elif self.oracle_threshold is not None:
            trig = self.oracle_load >= self.oracle_threshold
        elif self.always:
            trig = True
        else:
            reading = read(sim, self._fingers)
            if self.require_contact and not reading.in_contact:
                trig = False
            else:
                s = self._signal(reading)
                trig = s > self.threshold if self.comparison == "gt" else s < self.threshold
        if trig:
            if self._hold == 0:
                self._triggers += 1
            self._hold = self.hold_substeps
            if self._first_at is None:
                self._first_at = self._substeps
        active = self._hold > 0
        if self._hold > 0:
            self._hold -= 1
            self._fired += 1
        self._set_boost(sim, active)
