"""The headline sweep: how much does the reflex buy at each replan rate?

`n_action_steps` is how long the policy's decision stays frozen. At 20 Hz,
n = 5 is 250 ms of open-loop execution and n = 50 is 2.5 seconds. The reflex
re-decides every step regardless.

If the policy-only curve falls as the chunk lengthens while the reflex curve
stays flat, the gap at each n is what a 50 ms sub-policy loop is worth in units
of policy inference rate -- which is the trade every VLA deployment is making
implicitly. If both curves fall together, chunk length is not what is losing
the object, and the reflex framing is wrong for this failure mode.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from reflexarc.disturb import Impulse, MassScale
from reflexarc.reflex import SlipReflex, SqueezeReflex
from reflexarc.runner import PolicyRunner
from reflexarc.stats import wilson

LADDER = (5, 10, 20, 50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ishandotsh/act_libero_spatial_test")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--mass", type=float, default=200.0)
    ap.add_argument("--impulse", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--ladder", default=",".join(str(n) for n in LADDER))
    ap.add_argument("--arms", default="policy,reflex")
    ap.add_argument("--channel", default="cone_ratio")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--comparison", default="gt")
    ap.add_argument("--arrest-gain", type=float, default=0.2)
    ap.add_argument("--hold-steps", type=int, default=5)
    ap.add_argument("--force-gain", type=float, default=6.0)
    ap.add_argument("--no-contact-gate", action="store_true")
    ap.add_argument("--budget", type=float, default=7200.0)
    ap.add_argument("--out", default="runs/ladder")
    args = ap.parse_args()

    import time

    ladder = [int(x) for x in args.ladder.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "trials.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["n_action_steps"], r["seed"]))

    runner = PolicyRunner(checkpoint=args.ckpt, suite=args.suite, task_id=args.task,
                          n_action_steps=ladder[0], max_steps=400)
    t0 = time.time()
    try:
        for n in ladder:
            # Read at policy.reset(), which run() calls, so this takes effect
            # for the trial about to start.
            runner.n_action_steps = n
            runner.policy.config.n_action_steps = n
            for arm in arms:
                for seed in range(args.seeds):
                    if (arm, n, seed) in done:
                        continue
                    if time.time() - t0 > args.budget:
                        print("budget reached")
                        raise SystemExit(0)
                    reflex = None
                    if arm == "reflex":
                        reflex = SlipReflex(
                            channel=args.channel, threshold=args.threshold,
                            comparison=args.comparison, arrest_gain=args.arrest_gain,
                            hold_steps=args.hold_steps, act=True, interrupt=False,
                            require_contact=not args.no_contact_gate,
                        )
                    elif arm == "squeeze":
                        reflex = SqueezeReflex(
                            channel=args.channel, threshold=args.threshold,
                            comparison=args.comparison, arrest_gain=1.0,
                            hold_steps=args.hold_steps, act=True, interrupt=False,
                            force_gain=args.force_gain,
                            require_contact=not args.no_contact_gate,
                        )
                    roll = runner.run(
                        seed=seed, arm=arm, reflex=reflex,
                        mass=MassScale(args.mass),
                        impulse=Impulse(magnitude=args.impulse, seed=seed)
                        if args.impulse else None,
                    )
                    rec = roll.summary()
                    with manifest.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"  n={n:<3d} {arm:>7} seed={seed:<3d} {roll.outcome:15s} "
                          f"fired={len(roll.reflex_steps):<3d} ({roll.wall_time:.1f}s)",
                          flush=True)
    finally:
        runner.close()
        report(manifest)


def report(manifest: Path) -> None:
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    cells: dict[tuple, list] = {}
    for r in rows:
        cells.setdefault((r["n_action_steps"], r["arm"]), []).append(r)

    print(f"\n{'n':>4} {'blind ms':>9} {'arm':>8} {'success':>18} {'lifted_lost':>12}")
    for n in sorted({k[0] for k in cells}):
        for arm in ("policy", "reflex", "squeeze", "interrupt", "yoked"):
            rs = cells.get((n, arm))
            if not rs:
                continue
            c = Counter(r["outcome"] for r in rs)
            p, lo, hi = wilson(c["success"], len(rs))
            print(f"{n:>4} {n*50:>9} {arm:>8} {c['success']:>3}/{len(rs):<3} "
                  f"[{lo:.2f},{hi:.2f}] {c['lifted_lost']:>12}")


if __name__ == "__main__":
    main()
