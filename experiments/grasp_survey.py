"""Where does a LIBERO grasp actually carry its load?

On `libero_spatial` task 0 every load-bearing contact during a successful carry
is between the object and the finger *shafts*; the fingertip pads touch for one
step during closure and then separate. If that holds across tasks, a fingertip
sensor -- which is where real tactile hardware sits -- is blind to most of this
benchmark, and that is a statement about LIBERO. If it is specific to the bowl
rim, it is a statement about bowls, and the experiment should move to a suite
whose objects are pinched rather than wedged.

Reports, per task, the share of grip normal force borne by the pads during the
carry phase (both fingers in contact and the object off the table).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reflexarc.runner import PolicyRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--grid", default="libero_spatial:0,1,2,3;libero_object:0,1,2,3")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="runs/grasp_survey")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "survey.jsonl"

    cells = []
    for part in args.grid.split(";"):
        suite, ids = part.split(":")
        cells += [(suite, int(t)) for t in ids.split(",")]

    rows = []
    for suite, task in cells:
        try:
            runner = PolicyRunner(checkpoint=args.ckpt, suite=suite, task_id=task,
                                  max_steps=400)
        except Exception as e:
            print(f"{suite}:{task}  SKIP ({type(e).__name__}: {e})", flush=True)
            continue
        try:
            for seed in range(args.seeds):
                roll = runner.run(seed=seed)
                fn = roll.tactile.array("fn_left") + roll.tactile.array("fn_right")
                pad = roll.tactile.array("fn_pad")
                ncon = roll.tactile.array("n_contact")
                lift = (roll.obj_height - roll.obj_height[0]).max(axis=1)
                # Carry phase: gripper loaded and the object clear of the table.
                carry = (ncon > 0) & (fn > 0.1) & (lift > 0.02)
                rec = {
                    "suite": suite, "task": task, "seed": seed,
                    "success": roll.success, "carry_steps": int(carry.sum()),
                    "pad_share": float(pad[carry].sum() / max(fn[carry].sum(), 1e-9))
                    if carry.any() else None,
                    "fn_mean": float(fn[carry].mean()) if carry.any() else None,
                    "cone_mean": float(np.minimum(
                        roll.tactile.array("cone_ratio"), 1e3)[carry].mean())
                    if carry.any() else None,
                }
                rows.append(rec)
                with manifest.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                ps = "n/a" if rec["pad_share"] is None else f"{rec['pad_share']:.3f}"
                print(f"{suite}:{task} seed={seed} success={str(roll.success):5s} "
                      f"carry={rec['carry_steps']:3d} pad_share={ps}", flush=True)
        finally:
            runner.close()

    print("\nper task: pad share of grip force during carry")
    print(f"{'suite':>16} {'task':>4} {'carried':>8} {'pad share':>10} {'fn mean':>8} {'cone':>6}")
    for suite, task in cells:
        rs = [r for r in rows if r["suite"] == suite and r["task"] == task
              and r["pad_share"] is not None]
        if not rs:
            print(f"{suite:>16} {task:>4} {'-':>8} {'no carry':>10}")
            continue
        print(f"{suite:>16} {task:>4} {len(rs):>8} "
              f"{np.mean([r['pad_share'] for r in rs]):>10.3f} "
              f"{np.mean([r['fn_mean'] for r in rs]):>8.2f} "
              f"{np.mean([r['cone_mean'] for r in rs]):>6.2f}")


if __name__ == "__main__":
    main()
