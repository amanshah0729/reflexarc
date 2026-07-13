"""What a fingertip is allowed to know.

The whole experiment turns on a discipline: the reflex may only read signals a
real fingertip sensor could produce. In simulation it is trivially easy to read
the object's true velocity and build a slip detector that no hardware could
ever match, and equally easy to do it by accident. So the sensor model is a
module rather than a few lines inside the controller, and the oracle channels
are named `oracle_*` so that using one is a visible choice.

The sensor model is the contact wrench on the two Panda fingerpads. On the real
Panda that corresponds roughly to a 3-axis force sensor behind each pad, which
is the cheap end of tactile hardware -- no taxel array, no vision-based skin.
If a reflex works on this, it works on almost anything.

Three facts about the simulator shape what is measurable:

1. Contacts are per-geom-pair *points*, not surfaces. A bowl gripped by one pad
   typically registers several simultaneous contacts. Forces must be summed
   over all contacts involving that pad, not read from contact[0].

2. `mj_contactForce` returns the wrench in the *contact frame*: component 0 is
   along the contact normal, components 1-2 span the tangent plane. That is
   exactly the decomposition a slip detector wants, and it is why this reads
   contacts directly rather than using `cfrc_ext`, which is in world frame and
   would have to be projected back.

3. LIBERO grasps mostly do not happen on the fingertips. Measured on
   `libero_spatial` task 0 during a successful carry, every load-bearing
   contact is between the bowl and `gripper0_finger{1,2}_collision` -- the
   finger *shafts* -- while the `_pad_collision` geoms touch for a single step
   during closure and then separate. The bowl rim wedges between the fingers
   rather than being pinched by their tips.

   So a sensor restricted to the pads, which is where real tactile hardware
   sits, reads zero through most of a successful grasp. This module therefore
   senses every finger geom and reports the pad share separately via
   `pad_fraction`, because the gap between the two is a fact about the
   benchmark that a tactile project should state rather than paper over.

The slip signal is the friction-cone ratio |Ft| / (mu |Fn|). At 1.0 the contact
is at the edge of Coulomb friction and about to slide. It requires no learning,
no training data, and is computable from the two forces above, which is the
point: if a reflex needs more than this, that is a finding about what tactile
hardware manipulation actually requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FINGER_SUFFIXES = (
    "finger1_collision", "finger1_pad_collision",
    "finger2_collision", "finger2_pad_collision",
)


def _unwrap(sim):
    """robosuite wraps MjModel/MjData one level deep; accept either."""
    model = getattr(sim, "model", sim)
    data = getattr(sim, "data", None)
    return getattr(model, "_model", model), getattr(data, "_data", data)


@dataclass(frozen=True)
class FingerGeoms:
    """Collision geoms of the two fingers, split by finger and by pad/shaft."""

    left: tuple[int, ...]
    right: tuple[int, ...]
    pads: tuple[int, ...]

    @property
    def all(self) -> tuple[int, ...]:
        return self.left + self.right

    @classmethod
    def resolve(cls, sim) -> FingerGeoms:
        import mujoco

        model, _ = _unwrap(sim)
        left, right, pads = [], [], []
        for gid in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.endswith(FINGER_SUFFIXES):
                continue
            (left if "finger1" in name else right).append(gid)
            if "pad" in name:
                pads.append(gid)
        if not left or not right:
            raise RuntimeError(
                "could not resolve finger collision geoms "
                f"(left={left}, right={right}). A tactile sensor that silently "
                "reads nothing looks exactly like a stable grasp."
            )
        return cls(tuple(left), tuple(right), tuple(pads))


@dataclass
class TactileReading:
    """One timestep of fingertip sensing, plus clearly-labelled oracle channels."""

    fn_left: float = 0.0      # summed normal force on the left pad, N
    fn_right: float = 0.0
    ft_left: float = 0.0      # summed tangential force magnitude, N
    ft_right: float = 0.0
    n_contact_left: int = 0
    n_contact_right: int = 0
    fn_pad: float = 0.0       # of the above, the share borne by the fingertips
    mu: float = 1.0           # sliding friction of the finger/object pair

    # --- oracle: physically unmeasurable, for ground truth and labelling only
    oracle_obj_speed_in_hand: float = 0.0   # |v_object - v_hand|, m/s
    oracle_obj_height: float = 0.0
    oracle_grasped_body: int = -1

    @property
    def in_contact(self) -> bool:
        return self.n_contact_left > 0 and self.n_contact_right > 0

    @property
    def fn(self) -> float:
        """Grip normal force: the smaller of the two pads.

        The min rather than the sum, because a two-finger grasp can only
        squeeze as hard as its weaker contact -- if one pad has slipped off
        entirely, summing hides that behind the other pad's force.
        """
        return min(self.fn_left, self.fn_right)

    @property
    def ft(self) -> float:
        return max(self.ft_left, self.ft_right)

    @property
    def cone_ratio(self) -> float:
        """|Ft| / (mu |Fn|). >= 1 means the contact is sliding.

        Returns inf when the pads report no normal force at all, which is the
        honest answer: an object held by nothing is maximally unstable, and
        returning 0 there (the naive guard) would read as a perfect grasp.
        """
        denom = self.mu * self.fn
        if denom < 1e-9:
            return float("inf") if (self.ft > 1e-9 or not self.in_contact) else 0.0
        return self.ft / denom

    @property
    def pad_fraction(self) -> float:
        """Share of grip normal force borne by the fingertips rather than the
        finger shafts. Near 0 for most LIBERO grasps -- see the module note."""
        total = self.fn_left + self.fn_right
        return (self.fn_pad / total) if total > 1e-9 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "fn": self.fn, "ft": self.ft, "cone_ratio": min(self.cone_ratio, 1e3),
            "fn_left": self.fn_left, "fn_right": self.fn_right,
            "ft_left": self.ft_left, "ft_right": self.ft_right,
            "fn_pad": self.fn_pad, "pad_fraction": self.pad_fraction,
            "n_contact": self.n_contact_left + self.n_contact_right,
            "oracle_slip_speed": self.oracle_obj_speed_in_hand,
            "oracle_obj_height": self.oracle_obj_height,
        }


def grasped_body(sim, fingers: FingerGeoms) -> int:
    """The body currently pinched between both pads, or -1.

    Resolved from contact rather than from the task definition, so the same
    code works on any LIBERO task without a per-task object table -- and so
    that a disturbance is applied to whatever the robot is *actually* holding,
    which is not always what the task intended.
    """
    import mujoco

    model, data = _unwrap(sim)
    left_bodies: dict[int, int] = {}
    right_bodies: dict[int, int] = {}
    for i in range(int(data.ncon)):
        c = data.contact[i]
        for g_self, g_other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if g_self in fingers.left or g_self in fingers.right:
                root = int(model.body_rootid[model.geom_bodyid[g_other]])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ""
                if not name or "robot" in name or "gripper" in name or name == "table":
                    continue
                bucket = left_bodies if g_self in fingers.left else right_bodies
                bucket[root] = bucket.get(root, 0) + 1
    shared = set(left_bodies) & set(right_bodies)
    if not shared:
        return -1
    return max(shared, key=lambda b: left_bodies[b] + right_bodies[b])


def read(sim, fingers: FingerGeoms, hand_vel: np.ndarray | None = None) -> TactileReading:
    """Sum the contact wrench on each pad and derive the slip margin."""
    import mujoco

    model, data = _unwrap(sim)
    r = TactileReading()
    wrench = np.zeros(6)
    mu_samples = []

    for i in range(int(data.ncon)):
        c = data.contact[i]
        if c.geom1 not in fingers.all and c.geom2 not in fingers.all:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        fn = abs(float(wrench[0]))
        ft = float(np.linalg.norm(wrench[1:3]))
        mu_samples.append(float(c.friction[0]))
        if c.geom1 in fingers.pads or c.geom2 in fingers.pads:
            r.fn_pad += fn
        if c.geom1 in fingers.left or c.geom2 in fingers.left:
            r.fn_left += fn
            r.ft_left += ft
            r.n_contact_left += 1
        else:
            r.fn_right += fn
            r.ft_right += ft
            r.n_contact_right += 1

    if mu_samples:
        r.mu = float(np.mean(mu_samples))

    bid = grasped_body(sim, fingers)
    r.oracle_grasped_body = bid
    if bid >= 0:
        r.oracle_obj_height = float(data.xpos[bid][2])
        # `data.cvel` is expressed in the body's own com-centred frame, while
        # robosuite reports hand velocity in world coordinates. Differencing
        # them directly compares two frames and produces a "slip speed" that is
        # not a speed. `mj_objectVelocity` with flg_local=0 returns
        # [angular, linear] in world frame, which is what this needs.
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
        v_obj = vel6[3:6]
        v_hand = np.zeros(3) if hand_vel is None else np.asarray(hand_vel, dtype=float)
        r.oracle_obj_speed_in_hand = float(np.linalg.norm(v_obj - v_hand))
    return r


@dataclass
class TactileTrace:
    """Per-step sensor history for one episode."""

    rows: list[dict[str, float]] = field(default_factory=list)

    def append(self, reading: TactileReading) -> None:
        self.rows.append(reading.as_dict())

    def array(self, key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in self.rows], dtype=float)

    def __len__(self) -> int:
        return len(self.rows)
