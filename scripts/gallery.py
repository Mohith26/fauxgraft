"""Make docs/gallery.png: real tumors next to Fauxgraft and naive fakes.

Zoomed 40 mm crops so the texture and edges are visible. Uses training-split
scans only (fake tumors are never painted into test scans).
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fauxgraft.synth import Inserter, tumor_stats  # noqa: E402
from fauxgraft.train import load, load_index, split  # noqa: E402

H = 48  # half-size of crop in voxels (0.42 mm) ~ 40 mm field of view


def crop2d(img, c, axis=2):
    sl = [slice(None)] * 3
    sl[axis] = int(np.clip(c[axis], 0, img.shape[axis] - 1))
    im = img[tuple(sl)]
    cc = [c[i] for i in range(3) if i != axis]
    out = np.full((2 * H, 2 * H), img.min(), dtype=np.float32)
    x0, y0 = int(cc[0]) - H, int(cc[1]) - H
    xs, ys = slice(max(x0, 0), min(x0 + 2 * H, im.shape[0])), slice(max(y0, 0), min(y0 + 2 * H, im.shape[1]))
    out[xs.start - x0: xs.stop - x0, ys.start - y0: ys.stop - y0] = im[xs, ys]
    return out.T


def main():
    train, _ = split(load_index())
    rng = np.random.default_rng(7)
    picks = [train[i] for i in (5, 60, 140, 230)]
    fig, ax = plt.subplots(3, 4, figsize=(12, 9.4), facecolor="#0b0d12")
    for j, r in enumerate(picks):
        s = load(r)
        ct, m = s["ct"].astype(np.float32), s["staple"]
        st = tumor_stats(ct, m)
        for i, mode in enumerate(["real", "fauxgraft", "naive"]):
            img, lab = ct.copy(), np.zeros_like(m)
            if mode == "real":
                lab = m
                c = np.array(np.nonzero(m)).mean(1)
            else:
                c = Inserter(ct, m, st, mode, rng).insert(img, lab)
            a = ax[i, j]
            a.imshow(crop2d(img, c), cmap="gray", vmin=-300, vmax=500, origin="lower")
            a.contour(crop2d(lab.astype(np.float32), c), [0.5], colors=["#fc8181" if mode == "real" else "#4fd1c5"],
                      linewidths=1.1, origin="lower")
            a.set_xticks([]), a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#232836")
            if j == 0:
                a.set_ylabel({"real": "real tumor\n(expert label)", "fauxgraft": "Fauxgraft\n(synthetic)",
                              "naive": "naive sphere\n(synthetic)"}[mode], color="#e6e9ef", fontsize=12)
    fig.suptitle("Top row: real xenografts with expert outlines. Rows 2-3: fake tumors painted into real scans. Axial, 40 mm field of view.",
                 color="#e6e9ef", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "docs", "gallery.png"), dpi=130, facecolor=fig.get_facecolor())
    print("wrote docs/gallery.png")


if __name__ == "__main__":
    main()
