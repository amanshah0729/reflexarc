"""How close does a LIBERO grasp ever come to failing?

C1 measured, on one task, that the akita black bowl weighs 5.6 g while the
grasp holding it can resist several newtons -- a margin so large that nothing
the policy does can lose the object, and therefore a benchmark score that
cannot depend on holding on. This asks whether that is one bowl or the whole
benchmark.

The metric is the **grasp safety factor**: the tangential load the contact can
carry before sliding, divided by the object's weight.

    S = mu * (Fn_left + Fn_right) / (m g)

S = 1 is the point of slipping. Human and robot grasp controllers are usually
described as targeting somewhere around 1.5-3; a factor of 50 means the object
would have to become fifty times heavier before the grasp cared.

Reported per (suite, task) with the pad share alongside, because C3 established
that where the load sits on the finger is policy-dependent and decides whether
a fingertip sensor sees anything at all.

Only successful carries are measured. A grasp that failed has no margin to
report, and including failures would mix "this grasp was secure" with "there
was no grasp".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reflexarc.runner import PolicyRunner

G = 9.80665


def audit_rollout(roll) -> dict | None:
    """Grasp mechanics during the carry phase of one rollout, or None."""
    fn_l = roll.tactile.array("fn_left")
    fn_r = roll.tactile.array("fn_right")
    pad = roll.tactile.array("fn_pad")
    cone = np.minimum(roll.tactile.array("cone_ratio"), 1e3)
    ncon = roll.tactile.array("n_contact")
    if roll.obj_height.size == 0:
        return None

    rise = roll.obj_height - roll.obj_height[0]
    lift = rise.max(axis=1)
    # Carry: both fingers loaded and the object clear of the table.
    carry = (ncon > 0) & (fn_l + fn_r > 0.1) & (lift > 0.02)
    if carry.sum() < 3:
        return None

    # The object being carried is the one that rose furthest.
    idx = int(np.argmax(rise.max(axis=0)))
    mass = roll.obj_masses[idx] if idx < len(roll.obj_masses) else float("nan")
    if not np.isfinite(mass) or mass <= 0:
        return None

    mu = float(np.median([r.get("mu", 1.0) for r in roll.tactile.rows])) \
        if "mu" in roll.tactile.rows[0] else None
    fn_total = float(np.median((fn_l + fn_r)[carry]))
    # mu is not stored per step; recover it from the cone ratio, which is
    # defined as ft / (mu * fn_min), so mu = ft / (cone * fn_min).
    ft = roll.tactile.array("ft")
    fn_min = np.minimum(fn_l, fn_r)
    ok = carry & (cone > 1e-6) & (fn_min > 1e-6)
    mu = float(np.median((ft[ok] / (cone[ok] * fn_min[ok])))) if ok.any() else 1.0

    weight = mass * G
    return {
        "object": roll.obj_names[idx],
        "mass_kg": round(mass, 5),
        "weight_N": round(weight, 4),
        "mu": round(mu, 3),
        "fn_total_N": round(fn_total, 3),
        "safety_factor": round(mu * fn_total / weight, 2),
        "cone_ratio_med": round(float(np.median(cone[carry])), 3),
        "pad_share": round(float(pad[carry].sum() / max((fn_l + fn_r)[carry].sum(), 1e-9)), 3),
        "carry_steps": int(carry.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--grid", default="libero_spatial:0,1,2,3;libero_object:0,1,2,3;"
                                      "libero_goal:0,1,2,3;libero_10:0,1,2,3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/grasp_audit")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "audit.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task"], r["seed"], r["ckpt"]))

    cells = []
    for part in args.grid.split(";"):
        if not part.strip():
            continue
        suite, ids = part.split(":")
        cells += [(suite, int(t)) for t in ids.split(",")]

    for suite, task in cells:
        if all((suite, task, s, args.ckpt) in done for s in range(args.seeds)):
            continue
        try:
            runner = PolicyRunner(checkpoint=args.ckpt, suite=suite, task_id=task,
                                  max_steps=400)
        except Exception as e:
            print(f"{suite}:{task}  SKIP ({type(e).__name__}: {e})", flush=True)
            continue
        try:
            for seed in range(args.seeds):
                if (suite, task, seed, args.ckpt) in done:
                    continue
                roll = runner.run(seed=seed)
                rec = {"suite": suite, "task": task, "seed": seed,
                       "ckpt": args.ckpt, "success": roll.success,
                       "outcome": roll.outcome}
                a = audit_rollout(roll)
                rec.update(a or {"safety_factor": None})
                with manifest.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                sf = rec.get("safety_factor")
                print(f"{suite}:{task} seed={seed} {roll.outcome:14s} "
                      f"S={'n/a' if sf is None else f'{sf:7.1f}'} "
                      f"m={rec.get('mass_kg')} pad={rec.get('pad_share')}", flush=True)
        finally:
            runner.close()

    report(manifest)


def report(manifest: Path) -> None:
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r.get("safety_factor") is not None]
    if not keep:
        print("no measurable carries")
        return
    by: dict[tuple, list] = {}
    for r in keep:
        by.setdefault((r["suite"], r["task"]), []).append(r)

    print(f"\n{'suite':>16} {'task':>4} {'n':>3} {'object':>26} {'mass':>8} "
          f"{'Fn':>7} {'S':>8} {'pad':>6}")
    for (suite, task), rs in sorted(by.items()):
        sf = np.median([r["safety_factor"] for r in rs])
        print(f"{suite:>16} {task:>4} {len(rs):>3} {rs[0]['object'][:26]:>26} "
              f"{rs[0]['mass_kg']*1000:>6.1f}g {np.median([r['fn_total_N'] for r in rs]):>7.2f} "
              f"{sf:>8.1f} {np.median([r['pad_share'] for r in rs]):>6.2f}")

    allsf = np.array([r["safety_factor"] for r in keep])
    print(f"\n{len(keep)} measured carries across {len(by)} tasks")
    print(f"safety factor: median {np.median(allsf):.1f}, "
          f"range {allsf.min():.1f} to {allsf.max():.1f}")
    print(f"fraction below 3 (the range a real grasp controller targets): "
          f"{(allsf < 3).mean():.0%}")


if __name__ == "__main__":
    main()
