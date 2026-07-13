"""Collect full tactile traces so the detector can be designed from data.

Picking a slip threshold by eye is how you get a reflex that fires constantly
during normal carries and calls it robustness. These traces support the prior
question: is an impending drop visible in the fingertip signal *before* it
happens, and by how many steps? If the answer is one step, no reflex at any
speed can help, and that is the result.

Saves one npz per rollout with every sensor channel at full temporal
resolution, plus an index recording the outcome class of each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reflexarc.disturb import Impulse, MassScale
from reflexarc.runner import PolicyRunner

CHANNELS = ("fn", "ft", "cone_ratio", "fn_left", "fn_right", "ft_left", "ft_right",
            "fn_pad", "pad_fraction", "n_contact", "oracle_slip_speed",
            "oracle_obj_height")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ishandotsh/act_libero_spatial_test")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--mass", type=float, default=200.0)
    ap.add_argument("--impulse", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--n-action-steps", type=int, default=20)
    ap.add_argument("--out", default="runs/traces_m200")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.jsonl"
    done = set()
    if index.exists():
        for line in index.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["seed"])

    runner = PolicyRunner(checkpoint=args.ckpt, suite=args.suite, task_id=args.task,
                          n_action_steps=args.n_action_steps, max_steps=400)
    counts: dict[str, int] = {}
    try:
        for seed in range(args.seeds):
            if seed in done:
                continue
            roll = runner.run(
                seed=seed,
                mass=MassScale(args.mass),
                impulse=Impulse(magnitude=args.impulse, seed=seed) if args.impulse else None,
            )
            arrs = {c: roll.tactile.array(c) for c in CHANNELS}
            arrs["actions"] = roll.actions
            arrs["eef_pos"] = roll.eef_pos
            arrs["obj_height"] = roll.obj_height
            np.savez_compressed(out_dir / "npz" / f"seed{seed:03d}.npz", **arrs)

            rec = roll.summary()
            rec["seed"] = seed
            with index.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            counts[roll.outcome] = counts.get(roll.outcome, 0) + 1
            print(f"  seed={seed:<3d} {roll.outcome:15s} lift={roll.lift_step} "
                  f"drop={roll.drop_step} steps={roll.steps:3d} "
                  f"({roll.wall_time:.1f}s)", flush=True)
    finally:
        runner.close()

    print(f"\noutcomes: {counts}")


if __name__ == "__main__":
    main()
