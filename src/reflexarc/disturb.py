"""Physical disturbances: the thing the policy cannot see coming.

Every perturbation axis in the VLA robustness literature that this project
grew out of is photometric, geometric or linguistic -- move the camera, change
the bulb, reword the instruction. Those all corrupt the *observation*. This
module corrupts the *dynamics*, which is a different failure mode: the policy's
plan was correct and the world moved anyway.

The primary instrument is a timed external wrench on the grasped object. It is
the right instrument for a latency experiment for one reason: a disturbance
that fires at a known step gives every measurement a clock. Detection latency,
response latency and "did the response land before the next replan" are all
differences against that timestamp, and none of them are recoverable from a
disturbance that is simply always present.

The impulse is armed by a grasp event rather than fired at a fixed step. Grasp
timing varies by tens of steps across seeds, so a fixed-step impulse would hit
some episodes mid-reach and others mid-transport, which confounds "the reflex
helped" with "the disturbance happened at a survivable moment".

Calibration note. LIBERO's object masses are not physically meaningful -- the
akita black bowl of `libero_spatial` task 0 has mass 0.0056 kg, so its weight
is 0.055 N and a 30 N tug throws it 17 metres. Forces here are therefore
specified in newtons and expected to be O(0.01-1 N). `MassScale` exists because
"give the bowl the mass a bowl has" is itself a perturbation worth running.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reflexarc.rng import stable_rng
from reflexarc.sense import FingerGeoms, _unwrap, grasped_body


@dataclass
class Impulse:
    """A timed external force on whatever the gripper is holding.

    Fires `delay_steps` control steps after the object is first lifted clear of
    its resting height, then holds for `duration_steps` before releasing.
    """

    magnitude: float = 0.0            # newtons; 0.0 is an exact no-op
    direction: str = "lateral"        # lateral | down | up | random
    lift_m: float = 0.02              # how far the object must rise to arm
    delay_steps: int = 5              # control steps between arming and firing
    duration_steps: int = 4           # 4 steps = 200 ms at 20 Hz
    seed: int = 0

    # --- runtime state, reset per episode
    _armed_at: int | None = field(default=None, init=False, repr=False)
    _fired_at: int | None = field(default=None, init=False, repr=False)
    _released_at: int | None = field(default=None, init=False, repr=False)
    _rest_height: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _body: int = field(default=-1, init=False, repr=False)

    @property
    def is_control(self) -> bool:
        return self.magnitude == 0.0

    @property
    def fired_at(self) -> int | None:
        """The step the disturbance began, or None if it never fired."""
        return self._fired_at

    @property
    def target_body(self) -> int:
        return self._body

    def describe(self) -> str:
        if self.is_control:
            return "none"
        ms = self.duration_steps * 50
        return f"{self.magnitude:.3f}N {self.direction} for {ms}ms"

    def unit(self) -> np.ndarray:
        rng = stable_rng("impulse", self.seed)
        if self.direction == "down":
            return np.array([0.0, 0.0, -1.0])
        if self.direction == "up":
            return np.array([0.0, 0.0, 1.0])
        if self.direction == "lateral":
            theta = rng.uniform(0, 2 * np.pi)
            return np.array([np.cos(theta), np.sin(theta), 0.0])
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)

    def reset(self, sim) -> None:
        """Clear state and record resting heights. Call once per episode."""
        self._armed_at = self._fired_at = self._released_at = None
        self._body = -1
        model, data = _unwrap(sim)
        data.xfrc_applied[:] = 0.0
        self._rest_height = {
            bid: float(data.xpos[bid][2]) for bid in range(int(model.nbody))
        }

    def update(self, sim, fingers: FingerGeoms, step: int) -> None:
        """Advance the impulse state machine by one control step.

        Called immediately before `env.step`, so the wrench set here is in
        force for all 25 physics substeps of that control step.
        """
        if self.is_control:
            return
        model, data = _unwrap(sim)

        if self._fired_at is not None:
            if step >= self._fired_at + self.duration_steps:
                if self._released_at is None:
                    data.xfrc_applied[self._body] = 0.0
                    self._released_at = step
                return
            # Re-assert every step: harmless, and robust to anything else in
            # the stack zeroing the buffer between control steps.
            data.xfrc_applied[self._body, :3] = self.unit() * self.magnitude
            return

        bid = grasped_body(sim, fingers)
        if bid < 0:
            self._armed_at = None
            return
        lifted = float(data.xpos[bid][2]) - self._rest_height.get(bid, 0.0)
        if lifted < self.lift_m:
            self._armed_at = None
            return
        if self._armed_at is None:
            self._armed_at = step
            self._body = bid
        elif step - self._armed_at >= self.delay_steps:
            self._body = bid
            data.xfrc_applied[bid, :3] = self.unit() * self.magnitude
            self._fired_at = step


def free_bodies(sim) -> list[int]:
    """Root ids of every free-floating body: the things a robot can pick up.

    Selected by joint type rather than by name. LIBERO scenes also contain
    hinged articulated parts (cabinet drawers at 3 kg each) whose names look
    just as object-like, and scaling those changes the task rather than the
    grasp.
    """
    import mujoco

    model, _ = _unwrap(sim)
    out = []
    for bid in range(int(model.nbody)):
        n = int(model.body_jntnum[bid])
        adr = int(model.body_jntadr[bid])
        if n >= 1 and int(model.jnt_type[adr]) == int(mujoco.mjtJoint.mjJNT_FREE):
            out.append(bid)
    return out


@dataclass
class MassScale:
    """Multiply the mass and inertia of every free-floating object.

    Applied after reset, like every scene mutation in this stack: `reset()`
    recompiles the model from XML and silently discards edits made before it.

    This is not a stress test bolted onto a realistic scene -- it is a
    correction to an unrealistic one. `libero_spatial` task 0 asks the robot to
    pick up a ceramic bowl whose mass is 5.6 grams. The gripper closes to about
    1.8 N of normal force against a pad friction of 2.0, so the grasp can
    resist roughly 3.6 N while the object weighs 0.055 N: a safety factor near
    65. Nothing the policy does to that grasp can lose it. A real 300 g bowl
    would sit near a factor of 1.2.
    """

    factor: float = 1.0

    @property
    def is_control(self) -> bool:
        return self.factor == 1.0

    def describe(self) -> str:
        return "none" if self.is_control else f"mass x{self.factor:g}"

    def apply(self, sim) -> dict[int, float]:
        """-> {body id: new mass}, so a trial can record what it actually ran."""
        import mujoco

        model, data = _unwrap(sim)
        out: dict[int, float] = {}
        for root in free_bodies(sim):
            for b in range(int(model.nbody)):
                if int(model.body_rootid[b]) == root:
                    if not self.is_control:
                        model.body_mass[b] *= self.factor
                        model.body_inertia[b] *= self.factor
            out[root] = float(model.body_mass[root])
        if not self.is_control:
            # `body_subtreemass` and `body_invweight0` are derived at compile
            # time and are used by the constraint solver. Editing body_mass
            # without refreshing them leaves the solver referencing the old
            # inertia -- contacts behave as if the object were still light,
            # which is a silent no-op in exactly the direction that would fake
            # the result this experiment looks for.
            #
            # But `mj_setConst` computes those constants *at* qpos0 and uses
            # mjData as its scratch space, so it overwrites the live state:
            # measured here, every object jumps from z = 970 mm on the table to
            # z = 0 and qpos becomes qpos0 exactly. The episode then runs to
            # completion on a scrambled scene and reports an ordinary-looking
            # success rate. Save and restore around it.
            qpos = np.array(data.qpos, copy=True)
            qvel = np.array(data.qvel, copy=True)
            mujoco.mj_setConst(model, data)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
        return out


@dataclass
class ContactFriction:
    """Scale sliding friction on *both* sides of the grasp contact.

    Lower friction is the direct analogue of a worn or contaminated pad, and
    unlike the impulse it degrades the grasp continuously rather than at one
    instant.

    Scaling only the gripper is close to a no-op, which is how this was first
    written and what it measured for a full sweep. MuJoCo combines the friction
    of a contacting pair by taking the elementwise **maximum** of the two geoms
    unless an explicit `<pair>` exists, and `libero_object` declares none
    (`npair == 0`). Its objects carry friction 0.95 against fingers at 1.0-2.0,
    so driving the fingers to 0.02 leaves the effective coefficient at 0.95: a
    2x reduction at most, and nothing below it however small the factor.
    Measured, a 50x reduction in finger friction moved success not at all --
    9/10 at every rung of the ladder.

    Both sides are therefore scaled, and `effective()` reports the coefficient
    the simulator will actually use, so a trial records the real value rather
    than the requested one.
    """

    factor: float = 1.0

    @property
    def is_control(self) -> bool:
        return self.factor == 1.0

    def describe(self) -> str:
        return "none" if self.is_control else f"contact friction x{self.factor:g}"

    @staticmethod
    def _object_geoms(sim) -> list[int]:
        model, _ = _unwrap(sim)
        roots = set(free_bodies(sim))
        return [g for g in range(int(model.ngeom))
                if int(model.body_rootid[model.geom_bodyid[g]]) in roots]

    def effective(self, sim, fingers: FingerGeoms) -> float:
        """The largest coefficient MuJoCo could pick for a finger/object pair."""
        model, _ = _unwrap(sim)
        f = max(float(model.geom_friction[g][0]) for g in fingers.all)
        o = max((float(model.geom_friction[g][0])
                 for g in self._object_geoms(sim)), default=0.0)
        return max(f, o)

    def apply(self, sim, fingers: FingerGeoms) -> float:
        if not self.is_control:
            model, _ = _unwrap(sim)
            for gid in list(fingers.all) + self._object_geoms(sim):
                model.geom_friction[gid][0] *= self.factor
        return self.effective(sim, fingers)


# Retained so older run scripts keep importing; the behaviour is the corrected one.
PadFriction = ContactFriction
