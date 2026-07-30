"""The breaking-load experiment. Pre-registered in docs/PREREGISTRATION.md.

Every earlier arms experiment here spent a rollout on one bit -- at a fixed
load, did the object drop -- which needs the load calibrated into a narrow band
or it reads as all-drop or all-hold, and which at n=25 can only see a swing of
about twenty points.

This ramps the load instead and records **where the grasp broke**. One number
per rollout, no calibration, and the episodes that never break are censored
rather than discarded or scored as failures.

The ramp also supplies something no previous setup had: a ground-truth onset.
Under a fixed heavy object the grasp is marginal from the instant it closes, so
there is no moment to measure a detector against, which is why every channel
came out near chance in F1. Here the load crosses the friction limit at a known
step.

Arms are one implementation at five decimations, so a rate comparison varies
rate and nothing else. `yoked` copies each sensed run's activation count and
places it without the sensor; `oracle` fires on the true load and bounds what
any detector could do.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from reflexarc.disturb import PostLiftDegrade
from reflexarc.rng import stable_rng
from reflexarc.runner import PolicyRunner
from reflexarc.simreflex import SimRateReflex
from reflexarc.survival import holm, kaplan_meier, logrank

LADDER = (500, 100, 20, 5, 1)   # Hz


def make_arm(name: str, args, schedule: set[int] | None) -> SimRateReflex | None:
    if name == "policy":
        return None
    common = dict(channel=args.channel, threshold=args.threshold,
                  comparison=args.comparison, force_gain=args.force_gain,
                  hold_substeps=args.hold_substeps,
                  require_contact=not args.no_contact_gate)
    if name.startswith("reflex@"):
        hz = float(name.split("@")[1].rstrip("Hz"))
        return SimRateReflex(decimation=int(round(500 / hz)), **common)
    if name == "yoked":
        return SimRateReflex(decimation=1, schedule=schedule or set(), **common)
    if name == "oracle":
        return SimRateReflex(decimation=1, oracle_threshold=args.oracle_load,
                             **common)
    raise ValueError(f"unknown arm {name}")


def yoked_schedule(manifest: Path, seed: int, source: str) -> set[int]:
    """Copy a sensed arm's activation count for this seed, placed blind."""
    for line in manifest.read_text().splitlines() if manifest.exists() else []:
        if not line.strip():
            continue
        r = json.loads(line)
        if r["arm"] == source and r["seed"] == seed:
            n = int(r.get("sim_triggers") or 0)
            total = int(r.get("sim_substeps") or 0)
            if n == 0 or total == 0:
                return set()
            rng = stable_rng("yoke_substep", seed)
            return set(int(x) for x in
                       rng.choice(np.arange(1, total + 1),
                                  size=min(n, total), replace=False))
    return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--mass-final", type=float, default=400.0)
    ap.add_argument("--ramp-steps", type=int, default=250)
    ap.add_argument("--n-action-steps", type=int, default=20,
                    help="policy replan interval; 20 steps = 1 s at 20 Hz")
    ap.add_argument("--channel", default="cone_ratio")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--comparison", default="gt")
    ap.add_argument("--force-gain", type=float, default=6.0)
    ap.add_argument("--hold-substeps", type=int, default=25)
    ap.add_argument("--oracle-load", type=float, default=40.0,
                    help="oracle fires once the true load multiplier exceeds this")
    ap.add_argument("--no-contact-gate", action="store_true")
    ap.add_argument("--yoke-source", default="reflex@500Hz")
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default="runs/breaking_load")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a] or (
        ["policy"] + [f"reflex@{hz}Hz" for hz in LADDER] + ["yoked", "oracle"])
    # yoked copies a sensed arm, so it must run after it whatever order is given
    arms.sort(key=lambda a: (a == "yoked", a == "oracle"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "trials.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["seed"]))

    runner = PolicyRunner(checkpoint=args.ckpt, suite=args.suite,
                          task_id=args.task, max_steps=400,
                          n_action_steps=args.n_action_steps)
    try:
        for arm in arms:
            for seed in range(args.seeds):
                if (arm, seed) in done:
                    continue
                sched = (yoked_schedule(manifest, seed, args.yoke_source)
                         if arm == "yoked" else None)
                sim_reflex = make_arm(arm, args, sched)
                degrade = PostLiftDegrade(mass_final=args.mass_final,
                                          ramp_steps=args.ramp_steps)
                roll = runner.run(seed=seed, arm=arm, sim_reflex=sim_reflex,
                                  degrade=degrade)
                rec = roll.summary()
                rec["arm"] = arm
                with manifest.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  {arm:>13} seed={seed:<3d} {roll.outcome:14s} "
                      f"broke_at={rec['load_at_drop']}"
                      f"{'' if rec['load_observed'] else ' (censored)'} "
                      f"trig={rec.get('sim_triggers', 0)} "
                      f"({roll.wall_time:.1f}s)", flush=True)
    finally:
        runner.close()

    report(manifest)


def report(manifest: Path) -> None:
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    by: dict[str, list] = {}
    for r in rows:
        if r.get("load_at_drop") is not None:
            by.setdefault(r["arm"], []).append(r)
    if "policy" not in by:
        print("no policy arm yet")
        return

    order = ["policy"] + [a for a in by if a != "policy"]
    print(f"\n{'arm':>13} {'n':>3} {'broke':>6} {'median load':>12} "
          f"{'25th':>7} {'success':>8} {'trig/ep':>8}")
    curves = {}
    for arm in order:
        rs = by.get(arm)
        if not rs:
            continue
        loads = [r["load_at_drop"] for r in rs]
        obs = [bool(r["load_observed"]) for r in rs]
        km = kaplan_meier(loads, obs)
        curves[arm] = (loads, obs)
        med = km.median()
        q25 = km.quantile(0.25)
        c = Counter(r["outcome"] for r in rs)
        trig = np.mean([r.get("sim_triggers") or 0 for r in rs])
        print(f"{arm:>13} {len(rs):>3} {km.n_events:>3}/{len(rs):<2} "
              f"{('>max' if med is None else f'{med:.0f}'):>12} "
              f"{('-' if q25 is None else f'{q25:.0f}'):>7} "
              f"{c['success']:>3}/{len(rs):<3} {trig:>8.0f}")

    base = curves["policy"]
    raw = {}
    for arm, cur in curves.items():
        if arm == "policy":
            continue
        raw[arm] = logrank(*base, *cur)[1]
    adj = holm(raw)
    print("\nlog-rank vs policy (Holm-corrected):")
    for arm in raw:
        star = "  *" if adj[arm] < 0.05 else ""
        print(f"  {arm:>13}  p = {raw[arm]:.4f}  ->  {adj[arm]:.4f}{star}")

    # Prediction 2: does the sensor explain anything the timing does not?
    if "yoked" in curves:
        best = max((a for a in curves if a.startswith("reflex@")),
                   key=lambda a: kaplan_meier(*curves[a]).median() or 1e9,
                   default=None)
        if best:
            p = logrank(*curves[best], *curves["yoked"])[1]
            print(f"\nsensed ({best}) vs yoked: p = {p:.4f}"
                  f"{'  -- distinguishable' if p < 0.05 else '  -- indistinguishable'}")


if __name__ == "__main__":
    main()
