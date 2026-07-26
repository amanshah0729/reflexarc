"""One instrumented rollout: policy, disturbance, reflex.

The architecture question this project asks is about *where in the stack* a
fast loop belongs, so the rollout loop exposes exactly three injection points
and nothing else:

    read tactile          <- what a fingertip knows at time t
    policy.select_action  <- refreshed only every n_action_steps
    reflex                <- runs every step, sees only the tactile reading
    impulse.update        <- the world, doing something unplanned
    env.step

The asymmetry between lines 2 and 3 is the experiment. LIBERO runs at 20 Hz, so
a chunk of 20 actions is a full second during which the policy's decision is
frozen; the reflex re-decides every 50 ms. Nothing else differs between arms.

Environment and policy construction is adapted from the `Faultline` harness in
the sibling `RoboticsResearch` repo, which established the parts that fail
silently rather than loudly: camera-key mapping derived from the checkpoint
(a key the policy does not recognise is dropped, not raised, leaving it blind
on one camera), `init_state_id` pinned to the seed (LeRobot advances it on
every reset, so a trial's scene depends on how many resets preceded it), and
global RNG seeding before construction (robosuite's placement sampler draws
from global state).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from reflexarc.disturb import (GripStrength, Impulse, MassScale, PadFriction,
                               PostLiftDegrade)
from reflexarc.sense import FingerGeoms, TactileReading, TactileTrace, read


@dataclass
class ReflexOutput:
    action: np.ndarray
    fired: bool = False
    interrupt: bool = False


class Reflex(Protocol):
    def reset(self) -> None: ...
    def __call__(
        self, action: np.ndarray, reading: TactileReading, step: int
    ) -> ReflexOutput: ...


def attach_task(vec_env, obs: dict) -> dict:
    """Attach the language instruction. LeRobot 0.4.4 and 0.6.x differ here."""
    try:
        from lerobot.envs.utils import add_envs_task  # 0.4.x
        return add_envs_task(vec_env, obs)
    except ImportError:
        pass
    for attr in ("task_description", "task"):
        try:
            obs["task"] = list(vec_env.call(attr))
            return obs
        except Exception:
            continue
    obs["task"] = [""] * getattr(vec_env, "num_envs", 1)
    return obs


@dataclass
class Rollout:
    """Everything recorded from one episode."""

    seed: int
    success: bool
    steps: int
    wall_time: float
    arm: str = "policy"
    impulse_desc: str = "none"
    impulse_step: int | None = None
    dropped: bool = False
    actions: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))
    eef_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    tactile: TactileTrace = field(default_factory=TactileTrace)
    # (steps, n_free_bodies) heights, and their names. Recorded unconditionally
    # because "was it ever lifted" and "was it lost" cannot be reconstructed
    # from contact alone: an object never picked up and an object dropped
    # immediately both read as no contact.
    obj_height: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    obj_names: list[str] = field(default_factory=list)
    obj_masses: list[float] = field(default_factory=list)
    # Breaking load: the disturbance multiplier in force when contact was lost,
    # or the ramp maximum if it never was. `load_observed` False means the
    # episode is right-censored -- the breaking load is known only to exceed
    # this -- which survival analysis uses and a mean silently mishandles.
    load_at_drop: float | None = None
    load_observed: bool = False
    reflex_steps: list[int] = field(default_factory=list)
    interrupt_steps: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def max_lift(self) -> float:
        """Greatest rise of any free object above its own starting height, m."""
        if self.obj_height.size == 0:
            return 0.0
        return float(np.max(self.obj_height - self.obj_height[0]))

    @property
    def lost_after_lift(self) -> bool:
        """Lifted clear of the table at some point, not held at the end."""
        if self.obj_height.size == 0 or len(self.tactile) == 0:
            return False
        lifted = (self.obj_height - self.obj_height[0]).max(axis=1) > 0.02
        if not lifted.any():
            return False
        first = int(np.argmax(lifted))
        return bool(self.tactile.array("n_contact")[first:][-1] == 0) and not self.success

    @property
    def lift_step(self) -> int | None:
        """First step at which any object is 20 mm clear of where it started."""
        if self.obj_height.size == 0:
            return None
        lifted = (self.obj_height - self.obj_height[0]).max(axis=1) > 0.02
        return int(np.argmax(lifted)) if lifted.any() else None

    @property
    def drop_step(self) -> int | None:
        """First step after the lift at which the gripper loses all contact.

        This is the event a reflex has to beat. Defined on contact rather than
        on height, because an object still held but sliding has not been
        dropped yet -- and the gap between those two moments is exactly the
        reaction window this project measures.
        """
        start = self.lift_step
        if start is None or len(self.tactile) == 0:
            return None
        ncon = self.tactile.array("n_contact")[start:]
        lost = ncon == 0
        if not lost.any():
            return None
        return start + int(np.argmax(lost))

    @property
    def outcome(self) -> str:
        """success | never_lifted | lifted_lost | carried_missed.

        A single success rate cannot drive this project: only `lifted_lost` is
        a failure a grasp reflex could prevent, and folding the other two into
        the same number would let a reflex look effective by shuffling
        failures between categories it never touched.
        """
        if self.success:
            return "success"
        if self.lift_step is None:
            return "never_lifted"
        return "lifted_lost" if self.drop_step is not None else "carried_missed"

    @property
    def reflex_latency(self) -> int | None:
        """Control steps between the impulse firing and the first reflex action."""
        if self.impulse_step is None:
            return None
        after = [s for s in self.reflex_steps if s >= self.impulse_step]
        return (after[0] - self.impulse_step) if after else None

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "arm": self.arm,
            "success": self.success,
            "steps": self.steps,
            "wall_time": round(self.wall_time, 2),
            "impulse": self.impulse_desc,
            "impulse_step": self.impulse_step,
            "dropped": self.dropped,
            "n_reflex": len(self.reflex_steps),
            "n_interrupt": len(self.interrupt_steps),
            "reflex_latency": self.reflex_latency,
            "load_at_drop": self.load_at_drop,
            "load_observed": self.load_observed,
            "max_lift_mm": round(self.max_lift * 1000, 1),
            "lost_after_lift": self.lost_after_lift,
            "outcome": self.outcome,
            "lift_step": self.lift_step,
            "drop_step": self.drop_step,
            "min_fn": float(np.nanmin(self.tactile.array("fn"))) if len(self.tactile) else None,
            "max_cone_ratio": float(np.nanmax(np.minimum(self.tactile.array("cone_ratio"), 1e3)))
            if len(self.tactile) else None,
            **self.meta,
        }


class PolicyRunner:
    """Holds one env + policy across many rollouts (construction is slow)."""

    def __init__(
        self,
        checkpoint: str = "ishandotsh/act_libero_spatial_test",
        suite: str = "libero_spatial",
        task_id: int = 0,
        device: str = "mps",
        resolution: int = 256,
        max_steps: int = 400,
        n_action_steps: int = 20,
    ) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.suite, self.task_id = suite, task_id
        self.max_steps, self.device = max_steps, device
        self.n_action_steps = n_action_steps
        self.checkpoint = checkpoint

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        self.policy_type = cfg.type
        policy_cls = get_policy_class(cfg.type)

        env_kwargs: dict[str, Any] = dict(
            task=suite, task_ids=[task_id],
            observation_height=resolution, observation_width=resolution,
            episode_length=max_steps,
        )
        expected = set(getattr(cfg, "input_features", {}) or {})
        if "observation.images.wrist_image" in expected:
            env_kwargs["camera_name_mapping"] = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "wrist_image",
            }
        try:
            self.env_cfg = LiberoEnvCfg(**env_kwargs)
        except TypeError:
            env_kwargs.pop("camera_name_mapping", None)
            self.env_cfg = LiberoEnvCfg(**env_kwargs)
        missing = expected - set(self.env_cfg.features_map.values())
        if missing and any(k.startswith("observation.images.") for k in missing):
            raise ValueError(
                f"{checkpoint} expects camera keys the env does not produce: "
                f"{sorted(missing)}"
            )

        envs = make_env(self.env_cfg, n_envs=1)
        self.vec_env = envs[list(envs)[0]][task_id]

        self.policy = policy_cls.from_pretrained(checkpoint).to(device).eval()
        self.policy.config.n_action_steps = n_action_steps
        dev = {"device_processor": {"device": device}}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, checkpoint,
            preprocessor_overrides=dev, postprocessor_overrides=dev,
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=self.policy.config
        )
        self.fingers: FingerGeoms | None = None

    # -- sim access ---------------------------------------------------------

    def _sub(self):
        return self.vec_env.envs[0]

    def _sim(self):
        sub = self._sub()
        return getattr(sub, "unwrapped", sub)._env.sim

    def _eef(self) -> np.ndarray:
        try:
            inner = getattr(self._sub(), "unwrapped", self._sub())._env
            return np.asarray(inner.env.robots[0]._hand_pos, dtype=float).reshape(3)
        except Exception:
            return np.zeros(3)

    def _hand_vel(self) -> np.ndarray:
        try:
            inner = getattr(self._sub(), "unwrapped", self._sub())._env
            return np.asarray(inner.env.robots[0]._hand_vel, dtype=float).reshape(3)
        except Exception:
            return np.zeros(3)

    def _reobserve(self):
        """Re-render after mutating the model, or the first frame is stale."""
        sub = self._sub()
        inner = getattr(sub, "unwrapped", sub)._env
        try:
            inner.sim.forward()
            raw = inner.env._get_observations(force_update=True)
            fresh = getattr(sub, "unwrapped", sub)._format_raw_obs(raw)
            return self._batch_like(fresh)
        except Exception:
            return None

    @classmethod
    def _batch_like(cls, d: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = cls._batch_like(v)
            else:
                arr = np.asarray(v)
                out[k] = arr[None] if arr.dtype != object else arr
        return out

    # -- rollout ------------------------------------------------------------

    def run(
        self,
        seed: int = 0,
        impulse: Impulse | None = None,
        mass: MassScale | None = None,
        friction: PadFriction | None = None,
        grip: GripStrength | None = None,
        degrade: PostLiftDegrade | None = None,
        reflex: Reflex | None = None,
        sim_reflex: Any = None,
        arm: str = "policy",
    ) -> Rollout:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

        sub = self._sub()
        pinned = getattr(sub, "unwrapped", sub)
        if hasattr(pinned, "init_state_id"):
            pinned.init_state_id = seed
        obs, _ = self.vec_env.reset(seed=seed)

        sim = self._sim()
        if self.fingers is None:
            self.fingers = FingerGeoms.resolve(sim)

        # Scene mutations must follow reset: reset recompiles from XML.
        mutated = False
        masses: dict[int, float] = {}
        if mass is not None:
            # Called even for the control, which mutates nothing but reports
            # the masses actually in force -- so every trial records what it
            # ran against rather than what it intended.
            masses = mass.apply(sim)
            mutated = mutated or not mass.is_control
        mu_eff: float | None = None
        if friction is not None:
            # Also called for the control, to record the coefficient MuJoCo
            # will actually use. Requesting x0.02 and getting 0.95 is what a
            # whole sweep measured before this was recorded.
            mu_eff = friction.apply(sim, self.fingers)
            mutated = mutated or not friction.is_control
        if mutated:
            fresh = self._reobserve()
            if fresh is not None:
                obs = fresh
        grip_kp: float | None = None
        if grip is not None:
            grip_kp = grip.apply(sim)
        if impulse is not None:
            impulse.reset(sim)
        if degrade is not None:
            degrade.reset(sim)

        self.policy.reset()
        if reflex is not None:
            reflex.reset()
        if sim_reflex is not None:
            # Attaches inside robosuite's physics loop, so it must be installed
            # after the scene mutations above and removed in the `finally`
            # below -- a leaked wrapper would keep firing on the next episode.
            sim_reflex.attach(getattr(sub, "unwrapped", sub)._env.env, self.fingers)

        import mujoco

        from reflexarc.disturb import free_bodies
        from reflexarc.sense import _unwrap

        model, data = _unwrap(sim)
        watch = free_bodies(sim)
        watch_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            for b in watch
        ]
        # Subtree mass, not `body_mass`: LIBERO objects are assemblies whose
        # root carries only part of the total, so the root's own mass
        # understates what the gripper is actually lifting.
        watch_masses = [
            float(sum(model.body_mass[b] for b in range(int(model.nbody))
                      if int(model.body_rootid[b]) == root))
            for root in watch
        ]

        actions, eefs, heights = [], [], []
        tactile = TactileTrace()
        reflex_steps: list[int] = []
        interrupt_steps: list[int] = []
        success = False
        step = 0
        t0 = time.time()

        for step in range(self.max_steps):
            reading = read(sim, self.fingers, hand_vel=self._hand_vel())
            tactile.append(reading)
            heights.append([float(data.xpos[b][2]) for b in watch])

            o = preprocess_observation(obs)
            o = attach_task(self.vec_env, o)
            o = self.env_pre(o)
            o = self.pre(o)
            with torch.inference_mode():
                raw_action = self.policy.select_action(o)
            a = self.env_post({ACTION: self.post(raw_action)})[ACTION].to("cpu").numpy()
            a = a.reshape(-1).copy()

            if reflex is not None:
                out = reflex(a, reading, step)
                a = np.asarray(out.action, dtype=float).reshape(-1)
                # A reflex may also reach into the simulator -- used by the
                # arm that raises grip force, which the action space cannot
                # express. Kept as an explicit opt-in hook so that an arm with
                # capabilities beyond the policy's is visible in the code
                # rather than hidden inside a controller.
                if hasattr(reflex, "apply_sim"):
                    reflex.apply_sim(sim, out.fired)
                if out.fired:
                    reflex_steps.append(step)
                if out.interrupt:
                    # Clearing the queue forces re-inference on the next step,
                    # which is what "abort the chunk" means for a chunked policy.
                    self.policy.reset()
                    interrupt_steps.append(step)

            if impulse is not None:
                impulse.update(sim, self.fingers, step)
            if degrade is not None:
                degrade.update(sim, self.fingers, step)
                if sim_reflex is not None:
                    # Ground truth for the oracle arm: the load actually in
                    # force. Never reaches the measurable arms.
                    sim_reflex.oracle_load = degrade.multiplier_at(step) or 1.0

            actions.append(a.copy())
            eefs.append(self._eef())

            obs, reward, terminated, truncated, info = self.vec_env.step(a[None])
            if bool(np.any(info.get("is_success", False))) or float(np.max(reward)) > 0:
                success = True
                break
            if np.any(terminated) or np.any(truncated):
                break

        sim_stats: dict[str, Any] = {}
        if sim_reflex is not None:
            sim_stats = {
                "sim_reflex": sim_reflex.describe(),
                "sim_fired_substeps": sim_reflex.fired_substeps,
                "sim_triggers": sim_reflex.triggers,
                "sim_substeps": sim_reflex.substeps_seen,
                "sim_first_substep": sim_reflex.first_substep,
            }
            sim_reflex.detach()

        # "Dropped" = the impulse fired, and afterwards the object was no
        # longer pinched between both pads. Distinguishes losing the object
        # from failing the task for some other reason.
        dropped = False
        fired_at = impulse.fired_at if impulse is not None else None
        if fired_at is not None and len(tactile) > fired_at + 1:
            after = tactile.array("n_contact")[fired_at:]
            dropped = bool(np.any(after == 0)) and not success

        roll = Rollout(
            seed=seed, success=success, steps=step + 1, wall_time=time.time() - t0,
            arm=arm,
            impulse_desc=impulse.describe() if impulse else "none",
            impulse_step=fired_at, dropped=dropped,
            actions=np.array(actions), eef_pos=np.array(eefs), tactile=tactile,
            obj_height=np.array(heights), obj_names=watch_names,
            obj_masses=watch_masses,
            reflex_steps=reflex_steps, interrupt_steps=interrupt_steps,
            meta={
                "checkpoint": self.checkpoint,
                "n_action_steps": self.n_action_steps,
                "task": f"{self.suite}:{self.task_id}",
                "mass_factor": mass.factor if mass else 1.0,
                "obj_mass_max": round(max(masses.values()), 5) if masses else None,
                "friction_factor": friction.factor if friction else 1.0,
                "effective_friction": round(mu_eff, 4) if mu_eff is not None else None,
                "grip_factor": grip.factor if grip else 1.0,
                "degrade": degrade.describe() if degrade else "none",
                "degrade_step": degrade.started_at if degrade else None,
                "degrade_fraction": round(degrade.fraction, 3) if degrade else None,
                "servo_kp": round(grip_kp, 2) if grip_kp is not None else None,
                **sim_stats,
            },
        )

        if degrade is not None and degrade.mass_final != 1.0:
            # Read off the same trace `outcome` uses, so "dropped" and "broke
            # at load L" cannot disagree.
            #
            # Two distinctions this has to get right, both found in the pilot:
            #
            # A successful episode also ends with contact lost -- that is the
            # policy putting the object down. Scoring that as a breaking load
            # would record every success as a failure at whatever load the ramp
            # happened to reach, which in the pilot read as 119x. A grasp is
            # only *broken* if the object was lost and the task then failed.
            #
            # And the censoring load is the load the episode actually reached,
            # not the ramp maximum. An episode that ended at step 145 was never
            # tested beyond the load in force at step 145; censoring it at 400x
            # would claim it survived a load it never saw.
            last = roll.steps - 1
            d = roll.drop_step
            if roll.lift_step is None or degrade.started_at is None:
                # Never acquired, so no grasp was ever at risk. Excluded from
                # the survival analysis rather than censored at load 1, which
                # would drag every arm's curve down by its acquisition rate.
                roll.load_at_drop = None
                roll.load_observed = False
            elif d is not None and not success:
                roll.load_at_drop = round(degrade.multiplier_at(d) or 1.0, 3)
                roll.load_observed = True
            else:
                roll.load_at_drop = round(degrade.multiplier_at(last) or 1.0, 3)
                roll.load_observed = False
        return roll

    def close(self) -> None:
        try:
            self.vec_env.close()
        except Exception:
            pass
