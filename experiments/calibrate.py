"""Find the regime where the grasp is actually at risk.

Nothing in the reflex experiment is measurable until failures exist to prevent.
These policies score at or near ceiling on nominal LIBERO, so the first job is
to locate a disturbance magnitude that costs real success without making the
task impossible -- roughly the 40-60% band, where both arms have room to move.

Two ladders, run separately:

  --axis mass      scale every free object's mass. Answers whether LIBERO's
                   5.6 g objects are the reason nothing is ever dropped.
  --axis impulse   timed external tug on the held object, at a fixed mass.

Resumable: trials append to trials.jsonl keyed by (axis, value, seed), so an
interrupted run continues and a partial run stays analyzable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from reflexarc.disturb import GripStrength, Impulse, MassScale, PadFriction
from reflexarc.runner import PolicyRunner
from reflexarc.stats import wilson


def key(axis: str, value: float, seed: int) -> str:
    return f"{axis}@{value:g}#{seed}"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["key"])
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("mass", "impulse", "friction", "grip"),
                    default="mass")
    ap.add_argument("--values", default="1,10,25,50,75,100")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--ckpt", default="ishandotsh/act_libero_spatial_test")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--n-action-steps", type=int, default=20)
    ap.add_argument("--mass", type=float, default=1.0,
                    help="fixed mass factor, for the impulse/friction ladders")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--budget", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    values = [float(v) for v in args.values.split(",")]
    out_dir = Path(args.out or f"runs/calib_{args.axis}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "trials.jsonl"
    done = load_done(manifest)

    pending = [(v, s) for v in values for s in range(args.seeds)
               if key(args.axis, v, s) not in done]
    print(f"{args.axis} ladder: {len(values)} values x {args.seeds} seeds "
          f"= {len(values) * args.seeds} trials, {len(done)} already done, "
          f"{len(pending)} pending")
    if not pending:
        report(manifest, args.axis)
        return

    runner = PolicyRunner(
        checkpoint=args.ckpt, suite=args.suite, task_id=args.task,
        n_action_steps=args.n_action_steps, max_steps=args.max_steps,
    )
    t0 = time.time()
    try:
        for i, (value, seed) in enumerate(pending):
            if time.time() - t0 > args.budget:
                print(f"budget reached; {len(pending) - i} trials left")
                break
            mass = MassScale(value if args.axis == "mass" else args.mass)
            friction = PadFriction(value if args.axis == "friction" else 1.0)
            impulse = (Impulse(magnitude=value, seed=seed)
                       if args.axis == "impulse" else None)

            grip = GripStrength(value if args.axis == "grip" else 1.0)
            roll = runner.run(seed=seed, impulse=impulse, mass=mass,
                              friction=friction, grip=grip, arm="policy")
            rec = roll.summary()
            rec.update({"key": key(args.axis, value, seed),
                        "axis": args.axis, "value": value})
            with manifest.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  [{i+1}/{len(pending)}] {args.axis}={value:<6g} seed={seed:<3d} "
                  f"success={str(roll.success):5s} dropped={str(roll.dropped):5s} "
                  f"steps={roll.steps:3d} obj_mass={rec.get('obj_mass_max')} "
                  f"({roll.wall_time:.1f}s)", flush=True)
    finally:
        runner.close()

    report(manifest, args.axis)


def report(manifest: Path, axis: str) -> None:
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    by_value: dict[float, list[dict]] = {}
    for r in rows:
        by_value.setdefault(r["value"], []).append(r)

    print(f"\n{axis} ladder")
    print(f"{'value':>8}  {'obj mass':>9}  {'success':>16}  {'dropped':>7}  {'steps':>6}")
    for v in sorted(by_value):
        rs = by_value[v]
        k, n = sum(bool(r["success"]) for r in rs), len(rs)
        p, lo, hi = wilson(k, n)
        drop = sum(bool(r.get("dropped")) for r in rs)
        m = rs[0].get("obj_mass_max")
        steps = sum(r["steps"] for r in rs) / n
        print(f"{v:>8g}  {(f'{m*1000:.1f}g' if m else '-'):>9}  "
              f"{k:>2}/{n:<2} [{lo:.2f},{hi:.2f}]  {drop:>7}  {steps:>6.0f}")


if __name__ == "__main__":
    main()
