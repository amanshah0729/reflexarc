"""Is an impending drop visible in the fingertip signal before it happens?

This is the question the whole project rests on, and it is prior to any reflex
design. A reflex can only act on warning it actually receives, so the quantity
that matters is *lead time*: how many control steps before the object is lost
does a detector first fire, and at what cost in false alarms during carries
that would have succeeded anyway.

Two failure modes for this analysis, both guarded against:

  Threshold picked to look good. Every detector is swept across its whole
  threshold range and reported as a lead-time / false-alarm trade-off, not as
  a single tuned number.

  Detecting the drop rather than predicting it. Lead time is measured to the
  step the gripper loses contact, and a detector that only fires at lead 0 is
  reported as useless however high its accuracy, because a reflex firing on
  the step the object leaves the hand has nothing left to act on.

The oracle channels are loaded but excluded from every candidate detector;
they appear only as the upper bound a physically-measurable signal is
competing against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# name -> (channel, comparison, sweep values). Comparison "gt" fires when the
# signal exceeds the threshold, "lt" when it falls below.
DETECTORS: dict[str, tuple[str, str, np.ndarray]] = {
    "cone_ratio":      ("cone_ratio", "gt", np.arange(0.5, 1.55, 0.05)),
    "grip_force_low":  ("fn", "lt", np.arange(0.2, 6.2, 0.2)),
    "tangential_high": ("ft", "gt", np.arange(0.5, 10.5, 0.5)),
    "force_collapse":  ("d_fn", "lt", -np.arange(0.1, 5.1, 0.1)[::-1]),
    "cone_rising":     ("d_cone", "gt", np.arange(0.02, 1.02, 0.02)),
    # oracle upper bound; never a candidate reflex input
    "ORACLE_slip":     ("oracle_slip_speed", "gt", np.arange(0.01, 0.51, 0.01)),
}


def derived(tr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(tr)
    cone = np.minimum(np.nan_to_num(tr["cone_ratio"], nan=0.0, posinf=1e3), 1e3)
    out["cone_ratio"] = cone
    out["d_fn"] = np.diff(tr["fn"], prepend=tr["fn"][:1])
    out["d_cone"] = np.diff(cone, prepend=cone[:1])
    return out


def fires(tr: dict[str, np.ndarray], det: str, thr: float) -> np.ndarray:
    ch, cmp_, _ = DETECTORS[det]
    x = tr[ch]
    return (x > thr) if cmp_ == "gt" else (x < thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/traces_m200")
    ap.add_argument("--min-lead", type=int, default=1,
                    help="a detection at lead < this is treated as no warning")
    ap.add_argument("--fig", default="docs/figures/detectability.png")
    args = ap.parse_args()

    root = Path(args.dir)
    index = [json.loads(l) for l in (root / "index.jsonl").read_text().splitlines()
             if l.strip()]
    traces = {}
    for rec in index:
        p = root / "npz" / f"seed{rec['seed']:03d}.npz"
        if p.exists():
            traces[rec["seed"]] = derived({k: v for k, v in np.load(p).items()})

    drops = [r for r in index if r["outcome"] == "lifted_lost" and r.get("drop_step")]
    holds = [r for r in index if r["outcome"] in ("success", "carried_missed")
             and r.get("lift_step") is not None]
    print(f"{len(drops)} episodes that lifted then lost the object")
    print(f"{len(holds)} episodes that lifted and kept it\n")
    if not drops:
        print("no drop episodes -- nothing to detect. Raise the mass or add an impulse.")
        return

    print(f"{'detector':>16} {'thr':>7} {'detected':>9} {'median lead':>12} "
          f"{'p90 lead':>9} {'false alarm':>12}")
    best: dict[str, tuple] = {}
    for det in DETECTORS:
        for thr in DETECTORS[det][2]:
            leads = []
            for r in drops:
                tr = traces.get(r["seed"])
                if tr is None:
                    continue
                lift, drop = r["lift_step"], r["drop_step"]
                f = fires(tr, det, float(thr))[lift:drop]
                # First firing inside the carry window; lead is steps to the drop.
                if f.any():
                    leads.append(drop - (lift + int(np.argmax(f))))
            # False alarm: fires at any point during a carry that never failed.
            fa = 0
            for r in holds:
                tr = traces.get(r["seed"])
                if tr is None:
                    continue
                end = r["drop_step"] or len(tr["fn"])
                if fires(tr, det, float(thr))[r["lift_step"]:end].any():
                    fa += 1
            usable = [l for l in leads if l >= args.min_lead]
            if not usable:
                continue
            rate_fa = fa / max(len(holds), 1)
            med = float(np.median(usable))
            # Score: warning that arrives in time, without crying wolf.
            score = (len(usable) / len(drops)) * med * (1 - rate_fa)
            if det not in best or score > best[det][0]:
                best[det] = (score, thr, len(usable), med,
                             float(np.percentile(usable, 90)), rate_fa)

    for det, (score, thr, n, med, p90, rate_fa) in sorted(
            best.items(), key=lambda kv: -kv[1][0]):
        print(f"{det:>16} {thr:>7.2f} {n:>4}/{len(drops):<4} {med:>10.1f} st "
              f"{p90:>7.1f} {rate_fa:>11.0%}")

    print("\nlead is in control steps; 1 step = 50 ms. A chunk of 20 = 1000 ms.")

    # --- figure: every channel aligned on the drop -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        win = 25
        channels = ["fn", "ft", "cone_ratio", "oracle_slip_speed"]
        fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 3.2))
        for ax, ch in zip(axes, channels):
            stack = []
            for r in drops:
                tr = traces.get(r["seed"])
                if tr is None:
                    continue
                d = r["drop_step"]
                seg = tr[ch][max(0, d - win):d + 1]
                if len(seg) < win + 1:
                    seg = np.pad(seg, (win + 1 - len(seg), 0), constant_values=np.nan)
                stack.append(seg)
            if not stack:
                continue
            arr = np.array(stack, dtype=float)
            x = np.arange(-win, 1)
            for row in arr:
                ax.plot(x, row, color="0.8", lw=0.7)
            ax.plot(x, np.nanmedian(arr, axis=0), color="crimson", lw=2)
            ax.axvline(0, color="k", ls=":", lw=1)
            ax.set_title(ch)
            ax.set_xlabel("control steps before drop")
        fig.suptitle("Fingertip signal in the 25 steps (1.25 s) before the object is lost",
                     fontsize=11)
        fig.tight_layout()
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=140)
        print(f"\nwrote {args.fig}")
    except Exception as e:
        print(f"\n(figure skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
