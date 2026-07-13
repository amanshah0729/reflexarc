"""Print the per-step story of a single rollout: contact, force, lift.

Used to answer "how did that fail", which a success rate cannot. Failure modes
worth telling apart:

  never grasped   - the gripper closed on nothing, or on the wrong object
  never lifted    - grasped, but the grip could not carry the weight
  lifted and lost - a real slip, which is the only failure a reflex can fix
  carried, missed - the grasp held and the policy still failed the task
"""

from __future__ import annotations

import argparse

import numpy as np

from reflexarc.disturb import Impulse, MassScale, PadFriction
from reflexarc.runner import PolicyRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ishandotsh/act_libero_spatial_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--impulse", type=float, default=0.0)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    r = PolicyRunner(checkpoint=args.ckpt, max_steps=args.max_steps)
    roll = r.run(
        seed=args.seed,
        mass=MassScale(args.mass),
        friction=PadFriction(args.friction),
        impulse=Impulse(magnitude=args.impulse, seed=args.seed) if args.impulse else None,
    )
    r.close()

    h = roll.obj_height
    rise = h - h[0]
    print(f"\nsuccess={roll.success} steps={roll.steps} "
          f"mass x{args.mass:g} friction x{args.friction:g} impulse={args.impulse:g}N")
    print(f"impulse fired at step {roll.impulse_step}")
    print(f"objects watched: {roll.obj_names}")
    print(f"max lift: {roll.max_lift*1000:.1f} mm   lost_after_lift={roll.lost_after_lift}")

    mover = int(np.argmax(rise.max(axis=0)))
    print(f"most-lifted object: {roll.obj_names[mover]}\n")

    print(f"{'step':>5} {'ncon':>5} {'fn':>8} {'ft':>8} {'cone':>7} "
          f"{'lift_mm':>8} {'grip_cmd':>9} {'|dxyz|':>8}")
    fn = roll.tactile.array("fn")
    ft = roll.tactile.array("ft")
    cone = roll.tactile.array("cone_ratio")
    ncon = roll.tactile.array("n_contact")
    for s in range(0, roll.steps, args.every):
        a = roll.actions[s]
        print(f"{s:>5} {int(ncon[s]):>5} {fn[s]:>8.3f} {ft[s]:>8.3f} "
              f"{min(cone[s], 999):>7.2f} {rise[s, mover]*1000:>8.1f} "
              f"{a[-1]:>9.2f} {np.linalg.norm(a[:3]):>8.3f}")


if __name__ == "__main__":
    main()
