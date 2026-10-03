"""Aggregate sweep results into results/summary.json and figures in docs/.

    python3 scripts/analyze.py

Also computes the human baseline (annotator vs annotator) on the same test
scans at the same 0.42 mm resolution, so model numbers have a yardstick.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fauxgraft.train import VOX_MM3, dice, load, load_index, split  # noqa: E402

RUNS = os.environ.get("FAUXGRAFT_RUNS", os.path.join(ROOT, "runs"))
if not os.path.isdir(RUNS):  # fall back to the per-run metrics committed in results/runs
    RUNS = os.path.join(ROOT, "results", "runs")
DOCS = os.path.join(ROOT, "docs")
COL = {"real": "#8b93a7", "naive": "#f6ad55", "fauxgraft": "#4fd1c5"}
LABEL = {"real": "real labels only", "naive": "+ naive spheres", "fauxgraft": "+ Fauxgraft tumors"}


def human_baseline():
    _, test = split(load_index())
    pair_d, pair_v = [], []
    for r in test:
        s = load(r)
        for x, y in [("a", "b"), ("a", "c"), ("b", "c")]:
            pair_d.append(dice(s[x], s[y]))
            vx, vy = s[x].sum() * VOX_MM3, s[y].sum() * VOX_MM3
            pair_v.append(abs(vx - vy) / max((vx + vy) / 2, 1e-6) * 100)
    return {"pairwise_dice_mean": float(np.mean(pair_d)), "pairwise_vol_diff_pct_mean": float(np.mean(pair_v)),
            "pairwise_vol_diff_pct_median": float(np.median(pair_v)), "n_test_scans": len(test)}


def collect():
    rows = []
    for p in glob.glob(os.path.join(RUNS, "*", "result.json")):
        with open(p) as f:
            r = json.load(f)
        row = {"arm": r["arm"], "k": r["k"], "seed": r["seed"], **r["metrics"], "per_scan": r["per_scan"]}
        rp = os.path.join(os.path.dirname(p), "reeval.json")
        if os.path.exists(rp):
            with open(rp) as f:
                re_ = json.load(f)
            for key, v in re_["metrics"]["filtered"].items():
                row["filt_" + key] = v
        rows.append(row)
    return rows


def paired_test(rows, k, a, b, metric="dice"):
    """Paired per-scan difference (b - a) pooled over seeds, with a bootstrap CI."""
    da = {(r["seed"], s["id"]): s[metric] for r in rows if r["arm"] == a and r["k"] == k for s in r["per_scan"]}
    db = {(r["seed"], s["id"]): s[metric] for r in rows if r["arm"] == b and r["k"] == k for s in r["per_scan"]}
    keys = sorted(set(da) & set(db))
    if not keys:
        return None
    d = np.array([db[x] - da[x] for x in keys])
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "b_better_frac": float((d > 0).mean()), "n_pairs": len(keys)}


def main():
    rows = collect()
    if not rows:
        print("no results yet")
        return
    agg = defaultdict(list)
    for r in rows:
        agg[(r["arm"], r["k"])].append(r)
    table = []
    for (arm, k), rs in sorted(agg.items(), key=lambda kv: (kv[0][1] or 999, kv[0][0])):
        def ms(key):
            v = np.array([x[key] for x in rs])
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        entry = {"arm": arm, "k": k, "n_seeds": len(rs),
                 "dice": ms("dice"), "vol_err_pct": ms("vol_err_pct"),
                 "median_vol_err_pct": ms("median_vol_err_pct"),
                 "lesion_recall": ms("lesion_recall"), "fp_components": ms("fp_components")}
        if all("filt_dice" in x for x in rs):
            for key in ("dice", "median_vol_err_pct", "lesion_recall", "fp_components"):
                entry["filt_" + key] = ms("filt_" + key)
        table.append(entry)
    tests = {}
    for k in sorted({r["k"] for r in rows if r["k"]}):
        for a, b in [("real", "fauxgraft"), ("real", "naive"), ("naive", "fauxgraft")]:
            t = paired_test(rows, k, a, b)
            if t:
                tests[f"k{k}_{b}_vs_{a}"] = t
    # label equivalence: how many real labels match each synthetic arm's Dice?
    real_pts = sorted([(t["k"], t["dice"][0]) for t in table if t["arm"] == "real" and t["k"]])
    equiv = {}
    if len(real_pts) >= 2:
        lk = np.log2([k for k, _ in real_pts])
        dv = np.array([d for _, d in real_pts])
        for t in table:
            if t["arm"] == "real" or not t["k"]:
                continue
            d = t["dice"][0]
            if d > dv.max():
                eq = f">{real_pts[-1][0]}"
            elif d < dv.min():
                eq = f"<{real_pts[0][0]}"
            else:  # monotone piecewise-linear in log2(K)
                order = np.argsort(dv)
                eq = round(float(2 ** np.interp(d, dv[order], lk[order])), 1)
            equiv[f"{t['arm']}_k{t['k']}"] = eq
    hb_path = os.path.join(ROOT, "results", "human_baseline.json")
    if os.path.exists(hb_path):
        hb = json.load(open(hb_path))
    else:
        hb = human_baseline()
        json.dump(hb, open(hb_path, "w"), indent=1)
    summary = {"table": table, "paired_tests": tests, "human_baseline": hb, "real_label_equivalent": equiv}
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    for t in table:
        print(f"{t['arm']:10s} k={str(t['k'] or 'all'):4s} n={t['n_seeds']} dice={t['dice'][0]:.3f}±{t['dice'][1]:.3f} "
              f"volerr={t['vol_err_pct'][0]:.1f}% med={t['median_vol_err_pct'][0]:.1f}% recall={t['lesion_recall'][0]:.2f} fp={t['fp_components'][0]:.1f}")
    for k, v in tests.items():
        print(k, json.dumps(v))
    print("human", hb)
    print("equivalent real labels", equiv)
    for t in table:
        if "filt_dice" in t:
            print(f"  filtered {t['arm']:10s} k={str(t['k'] or 'all'):4s} dice={t['filt_dice'][0]:.3f} "
                  f"medvol={t['filt_median_vol_err_pct'][0]:.1f}% recall={t['filt_lesion_recall'][0]:.2f} fp={t['filt_fp_components'][0]:.1f}")

    # ---- figure: label budget curve
    os.makedirs(DOCS, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for metric, ax, ylabel in [("dice", axes[0], "Dice vs consensus (higher is better)"),
                               ("median_vol_err_pct", axes[1], "median tumor-volume error % (lower is better)")]:
        for arm in ["real", "naive", "fauxgraft"]:
            pts = sorted([t for t in table if t["arm"] == arm and t["k"]], key=lambda t: t["k"])
            if not pts:
                continue
            ks = [t["k"] for t in pts]
            m = np.array([t[metric][0] for t in pts])
            s = np.array([t[metric][1] for t in pts])
            ax.plot(ks, m, "o-", color=COL[arm], label=LABEL[arm], lw=2)
            ax.fill_between(ks, m - s, m + s, color=COL[arm], alpha=0.15)
        full = [t for t in table if t["k"] == 0]
        if full:
            ax.axhline(full[0][metric][0], color="#555", ls=":", lw=1.2,
                       label=f"all {325} train scans (real)")
        if metric == "dice":
            ax.axhline(hb["pairwise_dice_mean"], color="#c05621", ls="--", lw=1.2, label="human vs human")
        else:
            ax.axhline(hb["pairwise_vol_diff_pct_median"], color="#c05621", ls="--", lw=1.2, label="human vs human")
        if metric != "dice":
            ax.set_ylim(bottom=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("real labeled scans available (K)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Mouse micro-CT tumor segmentation: what fake tumors are worth under a label budget", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "label_budget.png"), dpi=150, bbox_inches="tight")
    print("wrote docs/label_budget.png")


if __name__ == "__main__":
    main()
