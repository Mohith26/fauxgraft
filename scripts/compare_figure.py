"""docs/compare.png: same held-out test scans, models trained with K labeled scans.

Columns: expert consensus | real labels only | + naive spheres | + Fauxgraft.
Test scans are picked by a fixed rule (every 23rd test scan), not by result.
"""

import os
import sys

import matplotlib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fauxgraft.train import P, dice, load, load_index, make_net, norm, split  # noqa: E402

RUNS = os.environ.get("FAUXGRAFT_RUNS", os.path.join(ROOT, "runs"))
K, SEED = int(os.environ.get("K", 2)), int(os.environ.get("SEED", 0))


def predict(net, ct):
    x = torch.from_numpy(norm(ct))[None, None]
    with torch.no_grad():
        return (torch.sigmoid(sliding_window_inference(x, (P, P, P), 4, net, overlap=0.25, mode="gaussian"))[0, 0].numpy() > 0.5)


def main():
    torch.set_num_threads(4)
    nets = {}
    for arm in ["real", "naive", "fauxgraft"]:
        n = make_net()
        path = os.path.join(RUNS, f"{arm}_k{K}_s{SEED}", "model.pt")
        if not os.path.exists(path):  # committed copies for K=2, seed 0
            path = os.path.join(ROOT, "models", f"k{K}_s{SEED}", f"{arm}.pt")
        n.load_state_dict(torch.load(path, map_location="cpu"))
        n.eval()
        nets[arm] = n
    _, test = split(load_index())
    picks = test[::23][:4]
    cols = ["consensus", "real", "naive", "fauxgraft"]
    titles = {"consensus": "expert consensus", "real": f"real labels only (K={K})",
              "naive": f"+ naive spheres (K={K})", "fauxgraft": f"+ Fauxgraft (K={K})"}
    colors = {"consensus": "#fc8181", "real": "#e2e8f0", "naive": "#f6ad55", "fauxgraft": "#4fd1c5"}
    fig, ax = plt.subplots(len(picks), 4, figsize=(13, 3.4 * len(picks)), facecolor="#0b0d12")
    for i, r in enumerate(picks):
        s = load(r)
        ct, ref = s["ct"].astype(np.float32), s["staple"].astype(bool)
        preds = {"consensus": ref, **{a: predict(n, s["ct"]) for a, n in nets.items()}}
        # axial slice through the center of the largest consensus tumor
        lab, n = ndimage.label(ref)
        big = lab == 1 + int(np.argmax(ndimage.sum(ref, lab, range(1, n + 1))))
        z = int(np.array(np.nonzero(big))[2].mean())
        for j, col in enumerate(cols):
            a = ax[i, j]
            a.imshow(ct[:, :, z].T, cmap="gray", vmin=-300, vmax=500, origin="lower", aspect="equal")
            m = preds[col][:, :, z].T.astype(float)
            if m.any():
                a.contour(m, [0.5], colors=[colors[col]], linewidths=1.3, origin="lower")
            if col != "consensus":
                a.text(3, 4, f"Dice {dice(preds[col], ref):.2f}", color=colors[col], fontsize=10,
                       bbox=dict(facecolor="#0b0d12", alpha=0.7, edgecolor="none"))
            a.set_xticks([]), a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#232836")
            if i == 0:
                a.set_title(titles[col], color="#e6e9ef", fontsize=11)
    fig.suptitle(f"Held-out mice (axial slice through the tumor). Every model was trained with only {K} labeled scans. Dice is over the whole 3D scan.",
                 color="#e6e9ef", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "compare.png"), dpi=120, facecolor=fig.get_facecolor())
    print("wrote docs/compare.png", [r["id"] for r in picks])


if __name__ == "__main__":
    main()
