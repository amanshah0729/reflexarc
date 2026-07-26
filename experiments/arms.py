"""The main comparison: four arms, identical seeds, one disturbance regime.

  policy     the policy alone. Its decision is frozen for n_action_steps.
  reflex     policy + tactile reflex. Re-decides every step; arrests the arm.
  interrupt  policy + the same detector, but it only aborts the chunk. No
             action authority of its own, so it isolates "re-planned sooner"
             from "a fast controller acted".
  yoked      the same arrest, the same number of steps, placed without the
             sensor. Isolates "sensing mattered" from "moving slower mattered".

`yoked` is built from `reflex`'s own firing record, so it runs last and its
budget is matched per seed rather than on average.

Reported per arm as an outcome breakdown, not a success rate. Only
`lifted_lost` is a failure a grasp reflex could prevent, and a reflex that
converts drops into timeouts has not helped anyone.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from reflexarc.disturb import (ContactFriction, Impulse, MassScale,
                               PostLiftDegrade)
from reflexarc.reflex import ScheduledReflex, SlipReflex, SqueezeReflex, yoke
from reflexarc.runner import PolicyRunner
from reflexarc.simreflex import SimRateReflex
from reflexarc.stats import fisher_exact, wilson

OUTCOMES = ("success", "lifted_lost", "never_lifted", "carried_missed")


def build_reflex(args, act: bool, interrupt: bool) -> SlipReflex:
    return SlipReflex(
        channel=args.channel, threshold=args.threshold, comparison=args.comparison,
        arrest_gain=args.arrest_gain, hold_steps=args.hold_steps,
        act=act, interrupt=interrupt,
        require_contact=not args.no_contact_gate,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ishandotsh/act_libero_spatial_test")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--mass", type=float, default=200.0)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--post-mass", type=float, default=1.0,
                    help="ramp object mass to this multiple after the lift")
    ap.add_argument("--post-grip", type=float, default=1.0)
    ap.add_argument("--impulse", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n-action-steps", type=int, default=20)
    ap.add_argument("--channel", default="cone_ratio")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--comparison", default="gt")
    ap.add_argument("--arrest-gain", type=float, default=0.2)
    ap.add_argument("--hold-steps", type=int, default=5)
    ap.add_argument("--force-gain", type=float, default=4.0)
    ap.add_argument("--hold-substeps", type=int, default=25)
    ap.add_argument("--no-force-close", action="store_true",
                    help="squeeze arm changes servo gain only, so it differs\n                          from the 500 Hz arm in rate alone")
    ap.add_argument("--no-contact-gate", action="store_true",
                    help="let the detector fire when only one finger is loaded")
    ap.add_argument("--arms", default="policy,reflex,interrupt,yoked")
    ap.add_argument("--out", default="runs/arms")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "trials.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["seed"]))

    wanted = [a.strip() for a in args.arms.split(",")]
    # yoked depends on reflex having run, so force the order regardless of input
    order = [a for a in ("policy", "reflex", "interrupt", "squeeze",
                         "sim500", "sim500_noboost",
                         "always_squeeze", "yoked", "yoked_squeeze")
             if a in wanted]

    runner = PolicyRunner(checkpoint=args.ckpt, suite=args.suite, task_id=args.task,
                          n_action_steps=args.n_action_steps, max_steps=400)
    schedule: dict[int, set[int]] = {}
    try:
        for arm in order:
            if arm in ("yoked", "yoked_squeeze"):
                schedule = load_schedule(
                    manifest, args.seeds,
                    source="squeeze" if arm == "yoked_squeeze" else "reflex")
                if not schedule:
                    print("no reflex firings recorded; skipping yoked arm")
                    continue
            for seed in range(args.seeds):
                if (arm, seed) in done:
                    continue
                reflex = None
                if arm == "reflex":
                    reflex = build_reflex(args, act=True, interrupt=False)
                elif arm == "interrupt":
                    reflex = build_reflex(args, act=False, interrupt=True)
                elif arm == "squeeze":
                    # Isolates the grip-force channel: the arm motion is left
                    # untouched (gain 1.0) so any effect is the squeeze alone.
                    reflex = SqueezeReflex(
                        channel=args.channel, threshold=args.threshold,
                        comparison=args.comparison, arrest_gain=1.0,
                        hold_steps=args.hold_steps, act=True, interrupt=False,
                        force_gain=args.force_gain,
                        require_contact=not args.no_contact_gate,
                        close_gripper=not args.no_force_close,
                    )
                elif arm == "always_squeeze":
                    # Grip boost held for the whole episode, sensor ignored.
                    reflex = SqueezeReflex(
                        channel=args.channel, threshold=args.threshold,
                        comparison=args.comparison, arrest_gain=1.0,
                        hold_steps=args.hold_steps, act=True, interrupt=False,
                        force_gain=args.force_gain, always=True,
                    )
                elif arm == "yoked_squeeze":
                    # Timing control for the squeeze result: same boost, same
                    # number of activations per seed, times drawn blind.
                    reflex = SqueezeReflex(
                        channel=args.channel, threshold=args.threshold,
                        comparison=args.comparison, arrest_gain=1.0,
                        hold_steps=args.hold_steps, act=True, interrupt=False,
                        force_gain=args.force_gain,
                        schedule=schedule.get(seed, set()),
                    )
                elif arm == "yoked":
                    reflex = ScheduledReflex(schedule=schedule,
                                             arrest_gain=args.arrest_gain)
                    reflex.bind(seed)

                sim_reflex = None
                if arm in ("sim500", "sim500_noboost"):
                    sim_reflex = SimRateReflex(
                        channel=args.channel, threshold=args.threshold,
                        comparison=args.comparison,
                        force_gain=args.force_gain if arm == "sim500" else 1.0,
                        hold_substeps=args.hold_substeps,
                        require_contact=not args.no_contact_gate,
                    )

                roll = runner.run(
                    seed=seed, arm=arm, reflex=reflex, sim_reflex=sim_reflex,
                    mass=MassScale(args.mass),
                    friction=ContactFriction(args.friction),
                    degrade=PostLiftDegrade(grip_final=args.post_grip,
                                            mass_final=args.post_mass),
                    impulse=Impulse(magnitude=args.impulse, seed=seed)
                    if args.impulse else None,
                )
                rec = roll.summary()
                rec["reflex_steps"] = roll.reflex_steps
                rec["detector"] = (reflex.describe() if reflex is not None
                                   else sim_reflex.describe() if sim_reflex is not None
                                   else "none")
                with manifest.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                nfire = len(roll.reflex_steps) or roll.meta.get("sim_triggers", 0)
                print(f"  {arm:>10} seed={seed:<3d} {roll.outcome:15s} "
                      f"fired={nfire:<3d} steps={roll.steps:3d} "
                      f"({roll.wall_time:.1f}s)", flush=True)
    finally:
        runner.close()

    report(manifest)


def load_schedule(manifest: Path, n_seeds: int,
                  source: str = "reflex") -> dict[int, set[int]]:
    """Match a sensed arm's intervention budget, seed by seed."""
    out: dict[int, set[int]] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["arm"] != source or not r.get("reflex_steps"):
            continue
        lift = r.get("lift_step") or 0
        end = r.get("drop_step") or r["steps"]
        if end <= lift:
            continue
        out[r["seed"]] = yoke(r["reflex_steps"], (lift, end), r["seed"])
    return out


def report(manifest: Path) -> None:
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    print(f"\n{'arm':>10} {'n':>4} {'success':>17} {'lifted_lost':>12} "
          f"{'never_lifted':>13} {'missed':>7} {'fired':>7}")
    base = None
    for arm in ("policy", "reflex", "interrupt", "squeeze", "sim500",
                "sim500_noboost", "always_squeeze", "yoked_squeeze",
                "yoked"):
        rs = by_arm.get(arm)
        if not rs:
            continue
        n = len(rs)
        c = Counter(r["outcome"] for r in rs)
        p, lo, hi = wilson(c["success"], n)
        fired = sum(r["n_reflex"] or r.get("sim_triggers", 0) for r in rs) / n
        print(f"{arm:>10} {n:>4} {c['success']:>3}/{n:<3} [{lo:.2f},{hi:.2f}] "
              f"{c['lifted_lost']:>12} {c['never_lifted']:>13} "
              f"{c['carried_missed']:>7} {fired:>7.1f}")
        if arm == "policy":
            base = (c, n)

    if base is not None:
        bc, bn = base
        print("\nvs policy (Fisher exact, two-sided):")
        for arm in ("reflex", "interrupt", "squeeze", "sim500",
                    "sim500_noboost", "always_squeeze", "yoked_squeeze", "yoked"):
            rs = by_arm.get(arm)
            if not rs:
                continue
            c, n = Counter(r["outcome"] for r in rs), len(rs)
            for metric in ("success", "lifted_lost"):
                p = fisher_exact(c[metric], n - c[metric], bc[metric], bn - bc[metric])
                print(f"  {arm:>10} {metric:>13}: "
                      f"{c[metric]}/{n} vs {bc[metric]}/{bn}   p = {p:.3f}")


if __name__ == "__main__":
    main()
