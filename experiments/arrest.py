"""Does the reflex predict slip, or arrest it after sliding has begun?

F11 shows a grip reflex raises breaking load. It does not say by what
mechanism, and the two candidates imply different hardware.

  Prediction: the fingertip signal contains warning, the reflex fires before
    the object moves, and the grasp never slips. This needs a sensor that
    resolves incipient slip -- the expensive kind.
  Arrest: the object starts sliding, the reflex notices and clamps, and the
    slide stops before the object escapes. This needs only a fast loop and a
    crude signal.

F1 already found no fingertip channel predicts a drop above AUC 0.67, and the
best of them fired at the moment of lift and never stopped. That points at
arrest. This measures it directly rather than inferring it.

Slip onset is defined from the oracle channel -- the object's speed relative to
the hand -- because the question is when the object *actually* started moving,
which is ground truth rather than something the reflex is allowed to see. The
threshold is set per episode from the stable carry rather than as an absolute,
since a carry that is already vibrating has a different noise floor from one
that is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reflexarc.disturb import PostLiftDegrade
from reflexarc.runner import PolicyRunner
from reflexarc.simreflex import SimRateReflex

SUBSTEPS_PER_STEP = 25


def slip_onset(roll, lift: int, mult: float = 4.0) -> int | None:
    """First step after the lift where relative speed exceeds the carry's own floor."""
    v = roll.tactile.array("oracle_slip_speed")
    end = roll.drop_step or len(v)
    if lift is None or end - lift < 8:
        return None
    # Baseline from the first third of the carry, before the load has climbed.
    base = v[lift:lift + max(4, (end - lift) // 3)]
    base = base[np.isfinite(base)]
    if base.size == 0:
        return None
    floor = float(np.median(base)) * mult + 1e-6
    for s in range(lift, end):
        if np.isfinite(v[s]) and v[s] > floor:
            return s
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=14)
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--ramp-steps", type=int, default=100)
    ap.add_argument("--mass-final", type=float, default=400.0)
    ap.add_argument("--out", default="runs/arrest")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "arrest.jsonl"

    runner = PolicyRunner(checkpoint=args.ckpt, suite=args.suite,
                          task_id=args.task, max_steps=400)
    rows = []
    try:
        for arm in ("policy", "reflex"):
            for seed in range(args.seeds):
                sr = None
                if arm == "reflex":
                    sr = SimRateReflex(decimation=int(round(500 / args.rate_hz)),
                                       channel="cone_ratio", threshold=0.9,
                                       force_gain=6.0, hold_substeps=25)
                roll = runner.run(
                    seed=seed, arm=arm, sim_reflex=sr,
                    degrade=PostLiftDegrade(mass_final=args.mass_final,
                                            ramp_steps=args.ramp_steps))
                lift = roll.lift_step
                onset = slip_onset(roll, lift)
                first_sub = roll.meta.get("sim_first_substep")
                fire = (first_sub // SUBSTEPS_PER_STEP) if first_sub else None
                rec = {
                    "arm": arm, "seed": seed, "outcome": roll.outcome,
                    "lift_step": lift, "slip_onset": onset,
                    "drop_step": roll.drop_step, "first_fire_step": fire,
                    "load_at_drop": roll.load_at_drop,
                    "window": (roll.drop_step - onset)
                    if (roll.drop_step and onset) else None,
                    "lead": (onset - fire) if (onset is not None and fire is not None) else None,
                }
                rows.append(rec)
                with manifest.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  {arm:>7} seed={seed:<3d} {roll.outcome:14s} lift={lift} "
                      f"slip@{onset} fire@{fire} drop={roll.drop_step} "
                      f"lead={rec['lead']}", flush=True)
    finally:
        runner.close()

    report(rows)


def report(rows: list[dict]) -> None:
    pol = [r for r in rows if r["arm"] == "policy"]
    ref = [r for r in rows if r["arm"] == "reflex"]

    w = [r["window"] for r in pol if r["window"] is not None]
    if w:
        print(f"\nWithout a reflex, sliding starts {np.median(w):.0f} control steps "
              f"({np.median(w)*50:.0f} ms) before the object is lost "
              f"(n={len(w)}, IQR {np.percentile(w,25):.0f}-{np.percentile(w,75):.0f}).")
        print("That window is the entire budget any reflex has to work in.")

    leads = [r["lead"] for r in ref if r["lead"] is not None]
    if leads:
        before = sum(1 for l in leads if l > 0)
        print(f"\nThe reflex fires before sliding starts in {before}/{len(leads)} "
              f"episodes; median lead {np.median(leads):+.0f} steps "
              f"({np.median(leads)*50:+.0f} ms).")
        verdict = ("PREDICTION -- it fires before the object moves"
                   if np.median(leads) > 0 else
                   "ARREST -- it fires after sliding has already begun")
        print(f"\n=> {verdict}")
        print("   A positive lead means warning existed in the signal; a negative "
              "one means\n   the reflex is stopping a slide rather than preventing "
              "it, which is a\n   weaker requirement on the sensor and a harder one "
              "on the loop rate.")


if __name__ == "__main__":
    main()
