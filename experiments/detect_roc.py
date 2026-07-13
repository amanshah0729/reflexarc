"""Step-level discrimination: does the signal separate "about to drop" from "fine"?

The episode-level analysis in `detect.py` answers "would this detector have
fired before the drop", which is the operational question, but it has only as
many negatives as there are episodes that held on -- seven, at mass x200. A
false-alarm rate of 3/7 cannot support a conclusion.

This asks the same question per step, which is also the question the reflex
actually faces: standing at step t of a carry, is a drop coming within the next
`horizon` steps? Every carry step is a sample, so the negative class is large.

Reported as AUC, plus the operationally meaningful pair: at a threshold whose
per-episode false-alarm rate is at most `--max-fa`, how many drops are caught
and how much warning do they give.

AUC near 0.5 means the channel does not distinguish an imminent drop from an
ordinary marginal grasp, however early it fires -- a detector that is simply
always on has perfect recall and no information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANNELS = ("fn", "ft", "cone_ratio", "d_fn", "d_cone", "pad_fraction",
            "oracle_slip_speed")
HIGHER_IS_WORSE = {"ft": True, "cone_ratio": True, "d_cone": True,
                   "oracle_slip_speed": True, "fn": False, "d_fn": False,
                   "pad_fraction": True}


def derived(tr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(tr)
    cone = np.minimum(np.nan_to_num(tr["cone_ratio"], nan=0.0, posinf=1e3), 1e3)
    out["cone_ratio"] = cone
    out["d_fn"] = np.diff(tr["fn"], prepend=tr["fn"][:1])
    out["d_cone"] = np.diff(cone, prepend=cone[:1])
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / |pos||neg|, with tie correction. No scipy dependency."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(dirs: list[str]) -> list[tuple[dict, dict[str, np.ndarray]]]:
    out = []
    for d in dirs:
        root = Path(d)
        idx = root / "index.jsonl"
        if not idx.exists():
            continue
        for line in idx.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            p = root / "npz" / f"seed{rec['seed']:03d}.npz"
            if not p.exists():
                continue
            rec["_src"] = str(root)
            out.append((rec, derived({k: v for k, v in np.load(p).items()})))
    return out


def carry_window(rec: dict, tr: dict[str, np.ndarray]) -> tuple[int, int] | None:
    lift = rec.get("lift_step")
    if lift is None:
        return None
    end = rec.get("drop_step")
    if end is None:
        ncon = tr["n_contact"]
        held = np.nonzero(ncon > 0)[0]
        held = held[held >= lift]
        if len(held) == 0:
            return None
        end = int(held[-1]) + 1
    return (int(lift), int(end))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["runs/traces_m200"])
    ap.add_argument("--horizon", type=int, default=10,
                    help="steps ahead that count as 'about to drop'")
    ap.add_argument("--max-fa", type=float, default=0.20)
    ap.add_argument("--save-auc", default="runs/auc.json")
    args = ap.parse_args()
    aucs: dict[str, float] = {}

    data = load(args.dirs)
    drops = [(r, t) for r, t in data if r["outcome"] == "lifted_lost" and r.get("drop_step")]
    holds = [(r, t) for r, t in data
             if r["outcome"] in ("success", "carried_missed") and r.get("lift_step") is not None]
    print(f"{len(drops)} drop episodes, {len(holds)} hold episodes, "
          f"horizon = {args.horizon} steps ({args.horizon*50} ms)\n")
    if not drops or not holds:
        print("need both classes")
        return

    print(f"{'channel':>18} {'AUC':>6} {'n pos':>7} {'n neg':>7} "
          f"{'caught @FA<=':>13} {'median lead':>12} {'thr':>8}")
    for ch in CHANNELS:
        pos, neg = [], []
        for r, t in drops:
            w = carry_window(r, t)
            if w is None:
                continue
            lo, hi = w
            x = t[ch][lo:hi]
            steps = np.arange(lo, hi)
            imminent = (r["drop_step"] - steps) <= args.horizon
            pos += list(x[imminent])
            neg += list(x[~imminent])
        for r, t in holds:
            w = carry_window(r, t)
            if w is None:
                continue
            neg += list(t[ch][w[0]:w[1]])
        pos_a = np.array(pos, dtype=float)
        neg_a = np.array(neg, dtype=float)
        pos_a = pos_a[np.isfinite(pos_a)]
        neg_a = neg_a[np.isfinite(neg_a)]
        if HIGHER_IS_WORSE[ch]:
            a = auc(pos_a, neg_a)
        else:
            a = auc(-pos_a, -neg_a)

        aucs[ch] = round(a, 4)
        best = operating_point(ch, drops, holds, args.max_fa)
        cap = f"{best[0]}/{len(drops)}" if best else "-"
        lead = f"{best[1]:.1f} st" if best else "-"
        thr = f"{best[2]:.2f}" if best else "-"
        print(f"{ch:>18} {a:>6.3f} {len(pos_a):>7} {len(neg_a):>7} "
              f"{cap:>13} {lead:>12} {thr:>8}")

    print(f"\nAUC 0.5 = the channel cannot tell an imminent drop from an ordinary "
          f"marginal grasp.\n'caught' counts drop episodes detected at lead >= 1 "
          f"with per-episode false alarm <= {args.max_fa:.0%}.")

    if args.save_auc:
        p = Path(args.save_auc)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(aucs, indent=2))
        print(f"wrote {p}")


def operating_point(ch: str, drops, holds, max_fa: float):
    """Best (caught, median lead, threshold) subject to an episode-level FA cap."""
    vals = []
    for _, t in drops + holds:
        v = t[ch]
        vals += list(v[np.isfinite(v)])
    if not vals:
        return None
    grid = np.quantile(np.array(vals), np.linspace(0.01, 0.99, 60))
    worse_high = HIGHER_IS_WORSE[ch]
    best = None
    for thr in grid:
        fa = 0
        for r, t in holds:
            w = carry_window(r, t)
            if w is None:
                continue
            seg = t[ch][w[0]:w[1]]
            if (seg > thr if worse_high else seg < thr).any():
                fa += 1
        if fa / max(len(holds), 1) > max_fa:
            continue
        leads = []
        for r, t in drops:
            w = carry_window(r, t)
            if w is None:
                continue
            lo, hi = w
            seg = t[ch][lo:hi]
            f = seg > thr if worse_high else seg < thr
            if f.any():
                lead = r["drop_step"] - (lo + int(np.argmax(f)))
                if lead >= 1:
                    leads.append(lead)
        if not leads:
            continue
        cand = (len(leads), float(np.median(leads)), float(thr))
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return best


if __name__ == "__main__":
    main()
