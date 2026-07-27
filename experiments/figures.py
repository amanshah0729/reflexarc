"""Figures. One idea per figure, outcome breakdowns rather than success rates.

Success rate is the wrong summary for every result in this project: a reflex
that turns drops into timeouts moves no success rate and has changed the
failure mode entirely. Every bar here is stacked by outcome so that stays
visible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTCOMES = ("success", "carried_missed", "lifted_lost", "never_lifted")
COLORS = {
    "success": "#2f7d4f",
    "carried_missed": "#8fb8a0",
    "lifted_lost": "#c2402d",
    "never_lifted": "#9a9a9a",
}
LABELS = {
    "success": "success",
    "carried_missed": "carried, missed task",
    "lifted_lost": "lifted then dropped",
    "never_lifted": "never lifted",
}


def outcome_of(r: dict) -> str:
    """Outcome class, reconstructed for runs logged before it was recorded.

    The mass ladder predates the `outcome` field, and re-running it to add a
    derived column would burn an hour of simulator time to recompute something
    already implied by what was logged.
    """
    if "outcome" in r:
        return r["outcome"]
    if r.get("success"):
        return "success"
    if (r.get("max_lift_mm") or 0.0) < 20:
        return "never_lifted"
    return "lifted_lost" if r.get("lost_after_lift") else "carried_missed"


def rows(path: Path) -> list[dict]:
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in out:
        r["outcome"] = outcome_of(r)
    return out


def stacked(ax, groups: list[str], counts: list[Counter], totals: list[int]) -> None:
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for oc in OUTCOMES:
        vals = np.array([100.0 * c[oc] / max(t, 1) for c, t in zip(counts, totals)])
        ax.bar(x, vals, bottom=bottom, color=COLORS[oc], label=LABELS[oc],
               width=0.62, edgecolor="white", linewidth=0.6)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 100 else 100)
    ax.set_ylabel("% of episodes")
    ax.spines[["top", "right"]].set_visible(False)


def fig_mass(args) -> None:
    rs = rows(Path(args.mass) / "trials.jsonl")
    by = {}
    for r in rs:
        by.setdefault(r["value"], []).append(r)
    keys = sorted(by)
    counts = [Counter(r["outcome"] for r in by[k]) for k in keys]
    totals = [len(by[k]) for k in keys]
    labels = [f"x{k:g}\n{5.6*k/1000:.2f} kg" for k in keys]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    stacked(ax, labels, counts, totals)
    ax.set_xlabel("object mass (LIBERO default = x1, a 5.6 g ceramic bowl)")
    ax.set_title("LIBERO cannot test grasp stability at its own object masses\n"
                 "ACT, libero_spatial task 0, 15 seeds per cell", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    save(fig, args.out, "mass_ladder.png")


def fig_arms(args) -> None:
    rs = rows(Path(args.arms) / "trials.jsonl")
    by = {}
    for r in rs:
        by.setdefault(r["arm"], []).append(r)
    order = [a for a in ("policy", "reflex", "close_only", "squeeze",
                         "yoked_squeeze") if a in by]
    counts = [Counter(r["outcome"] for r in by[a]) for a in order]
    totals = [len(by[a]) for a in order]
    names = {
        "policy": "policy\nalone",
        "reflex": "+ reflex:\nfreeze arm",
        "close_only": "+ reflex:\nforce close",
        "interrupt": "+ reflex:\nreplan",
        "squeeze": "+ reflex:\ngrip x6",
        "yoked_squeeze": "grip x6,\ntiming shuffled",
        "always_squeeze": "grip x6,\nalways on",
        "yoked": "+ yoked\ncontrol",
    }

    fig, ax = plt.subplots(figsize=(2.0 * len(order) + 1.5, 4.4))
    stacked(ax, [names.get(a, a) for a in order], counts, totals)
    ax.set_title(
        "The one arm that works does not need the sensor\n"
        f"ACT, mass x200, n = {totals[0]} seeds per arm, identical seeds",
        fontsize=11)
    # The comparison the figure exists to make: sensed vs. timing-shuffled.
    if "squeeze" in order and "yoked_squeeze" in order:
        i, j = order.index("squeeze"), order.index("yoked_squeeze")
        ax.annotate("", xy=(i, 104), xytext=(j, 104),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.1))
        ax.text((i + j) / 2, 106, "indistinguishable", ha="center",
                fontsize=8.5, color="0.35")
        ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    save(fig, args.out, "arms.png")


def fig_ladder(args) -> None:
    rs = rows(Path(args.ladder) / "trials.jsonl")
    cells: dict[tuple, list] = {}
    for r in rs:
        cells.setdefault((r["n_action_steps"], r["arm"]), []).append(r)
    ns = sorted({k[0] for k in cells})
    arms = [a for a in ("policy", "reflex", "interrupt", "yoked")
            if any(k[1] == a for k in cells)]
    style = {"policy": ("#333333", "o", "policy alone"),
             "reflex": ("#c2402d", "s", "+ reflex (gated, fired 0x)"),
             "interrupt": ("#2f6f9f", "^", "+ replan trigger"),
             "yoked": ("#9a9a9a", "d", "+ yoked control")}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric, ylab in (
            (axes[0], "success", "task success (%)"),
            (axes[1], "lifted_lost", "lifted then dropped (%)")):
        for arm in arms:
            ys, los, his = [], [], []
            for n in ns:
                rr = cells.get((n, arm), [])
                c = Counter(r["outcome"] for r in rr)
                k, tot = c[metric], max(len(rr), 1)
                from reflexarc.stats import wilson
                p, lo, hi = wilson(k, tot)
                ys.append(100 * p); los.append(100 * lo); his.append(100 * hi)
            col, mark, lab = style[arm]
            ax.plot(ns, ys, marker=mark, color=col, label=lab, lw=1.8)
            ax.fill_between(ns, los, his, color=col, alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        # A log axis draws its own minor labels underneath these, which
        # collide with the two-line tick labels and read as noise.
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ns))
        ax.set_xticklabels([f"{n}\n{n*50} ms" for n in ns])
        ax.set_xlabel("action chunk length (open-loop blindness)")
        ax.set_ylabel(ylab)
        ax.set_ylim(-3, 103)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Replanning more often makes it worse, not better  "
                 "(ACT, mass x200, 20 seeds/cell)", fontsize=12)
    save(fig, args.out, "ladder.png")


def fig_auc(args) -> None:
    """Bar chart of step-level discrimination, produced from a saved json."""
    p = Path(args.auc)
    data = json.loads(p.read_text())
    chans = list(data.keys())
    vals = [data[c] for c in chans]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(chans))
    cols = ["#9a9a9a" if "oracle" in chans[i] else "#2f6f9f" for i in order]
    ax.barh(y, [vals[i] for i in order], color=cols, height=0.6)
    ax.axvline(0.5, color="crimson", ls="--", lw=1.2)
    ax.text(0.505, len(chans) - 0.4, "chance", color="crimson", fontsize=8)
    ax.set_yticks(y)
    # The oracle channel differenced two coordinate frames in these traces, so
    # its position on this axis is a defect rather than a physical claim.
    ax.set_yticklabels(
        [chans[i] + ("  (frame defect)" if "oracle" in chans[i] else "")
         for i in order], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC: imminent drop vs. ordinary marginal grasp")
    ax.set_title("No fingertip channel predicts when the object will be lost",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args.out, "discrimination.png")


def save(fig, out_dir: str, name: str) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / name, dpi=150)
    print(f"wrote {d / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--mass", default="runs/calib_mass")
    ap.add_argument("--arms", default="runs/arms_m200")
    ap.add_argument("--ladder", default="runs/ladder")
    ap.add_argument("--auc", default="runs/auc.json")
    ap.add_argument("--audit", default="runs/grasp_audit")
    ap.add_argument("--breaking", default="runs/breaking_load")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    which = {"mass": fig_mass, "arms": fig_arms, "ladder": fig_ladder,
             "auc": fig_auc, "audit": fig_audit,
             "breaking": fig_breaking}
    todo = [args.only] if args.only else list(which)
    for name in todo:
        try:
            which[name](args)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
        except Exception as e:
            print(f"skip {name}: {type(e).__name__}: {e}")




def fig_audit(args) -> None:
    """Grasp safety factor per task, against the range a real grasp lives in."""
    rows = [json.loads(l) for l in
            (Path(args.audit) / "audit.jsonl").read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r.get("safety_factor")]
    by: dict[str, list] = {}
    for r in keep:
        by.setdefault(f"{r['suite'].replace('libero_', '')}:{r['task']}", []).append(r)
    if not by:
        raise FileNotFoundError("no measurable carries in the audit")

    labels = sorted(by, key=lambda k: np.median([r["safety_factor"] for r in by[k]]))
    med = [np.median([r["safety_factor"] for r in by[k]]) for k in labels]
    lo = [min(r["safety_factor"] for r in by[k]) for k in labels]
    hi = [max(r["safety_factor"] for r in by[k]) for k in labels]
    mass = [by[k][0]["mass_kg"] * 1000 for k in labels]
    obj = [by[k][0]["object"].replace("_main", "").replace("_1", "") for k in labels]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(labels) + 2.4))
    ax.axvspan(1.5, 3.0, color="#2f7d4f", alpha=0.16, zorder=0)
    ax.axvline(1.0, color="#c2402d", lw=1.4, zorder=1)
    for i, (a, b) in enumerate(zip(lo, hi)):
        ax.plot([a, b], [i, i], color="0.72", lw=2.4, zorder=2, solid_capstyle="round")
    ax.scatter(med, y, s=42, color="#1f4e79", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  ·  {o[:18]} {m:.0f} g"
                        for l, o, m in zip(labels, obj, mass)], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3000)
    ax.set_xlabel("grasp safety factor  —  µ·Fn ⁄ weight   (1.0 = the object slips)")
    ax.text(2.1, len(labels) - 0.4, "where a real grasp\ncontroller operates",
            fontsize=8, color="#2f7d4f", ha="center", va="top")
    ax.text(1.05, 0.3, "slips", fontsize=8, color="#c2402d")
    allsf = np.array([r["safety_factor"] for r in keep])
    ax.set_title(
        "No grasp in LIBERO is anywhere near failing\n"
        f"SmolVLA, {len(keep)} carries across {len(by)} tasks in 4 suites — "
        f"median safety factor {np.median(allsf):.0f}×, minimum {allsf.min():.0f}×",
        fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, args.out, "grasp_audit.png")



def fig_breaking(args) -> None:
    """The headline: breaking load against reflex rate, with both controls."""
    import collections

    rows = [json.loads(l) for l in
            (Path(args.breaking) / "trials.jsonl").read_text().splitlines() if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("load_at_drop") is not None:
            by[r["arm"]].append(r["load_at_drop"])

    ladder = [("reflex@1Hz", 1), ("reflex@5Hz", 5), ("reflex@20Hz", 20),
              ("reflex@100Hz", 100), ("reflex@500Hz", 500)]
    ladder = [(a, hz) for a, hz in ladder if a in by]
    med = lambda v: float(np.median(v))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    base = med(by["policy"])
    ax.axhline(base, color="#333333", lw=1.6, ls="--", zorder=2)
    ax.text(1.05, base + 6, "policy alone, no reflex", fontsize=9, color="#333333")

    if "yoked" in by:
        y = med(by["yoked"])
        ax.axhline(y, color="#c2402d", lw=1.5, ls=":", zorder=2)
        ax.text(1.05, y + 5, "same squeezes, random times", fontsize=9,
                color="#c2402d")

    xs = [hz for _, hz in ladder]
    ys = [med(by[a]) for a, _ in ladder]
    for (a, hz) in ladder:
        v = by[a]
        ax.plot([hz, hz], [np.percentile(v, 25), np.percentile(v, 75)],
                color="#9fb8cc", lw=6, solid_capstyle="round", zorder=3)
    ax.plot(xs, ys, "-o", color="#1f4e79", lw=2.2, ms=9, zorder=4,
            label="tactile reflex")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{hz}" for hz in xs])
    ax.set_xlabel("reflex rate (Hz)   —   the policy acts at 20 Hz")
    ax.set_ylabel("breaking load  (x object mass)")
    ax.axvspan(20, 100, color="#2f7d4f", alpha=0.08, zorder=0)
    ax.text(45, max(ys) + 4, "the knee", fontsize=9, color="#2f7d4f", ha="center")
    ax.set_title(
        "A reflex has to be faster than the policy to be worth anything\n"
        "load-to-failure, libero_object task 0, SmolVLA, 14 seeds/arm; "
        "bars are the interquartile range", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, args.out, "breaking_load.png")

if __name__ == "__main__":
    main()
