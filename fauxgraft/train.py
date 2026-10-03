"""Train and evaluate one tumor segmenter under a fixed label budget.

    python3 -m fauxgraft.train --k 4 --arm fauxgraft --seed 0

  --k     number of real labeled scans the model may use (each from a different mouse)
  --arm   real       only those K real labels
          naive      K real + on-the-fly naive fake tumors (spheres)
          fauxgraft  K real + on-the-fly realistic fake tumors
  --seed  picks WHICH K mice are labeled, plus init/sampling randomness

Every arm gets the same number of iterations and the same share of
tumor-centered patches, so the only thing that changes is what the tumors are.
Results go to runs/<arm>_k<k>_s<seed>/result.json.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from scipy import ndimage

from .synth import VOXEL_MM, Inserter, tumor_stats

DATA = os.environ.get("FAUXGRAFT_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "npz"))
P = 64
VOX_MM3 = VOXEL_MM ** 3


# ------------------------------------------------------------------ data
def load_index():
    with open(os.path.join(DATA, "index.json")) as f:
        return json.load(f)


def split(idx, test_frac=0.25, seed=0):
    """Mouse-level split: no mouse appears in both train and test."""
    mice = sorted({r["mouse"] for r in idx})
    rng = np.random.default_rng(seed)
    rng.shuffle(mice)
    test_mice = set(mice[: int(len(mice) * test_frac)])
    train = [r for r in idx if r["mouse"] not in test_mice]
    test = [r for r in idx if r["mouse"] in test_mice]
    return train, test


def pick_labeled(train, k, seed):
    """K scans from K distinct mice. Nested: the k=4 set contains the k=2 set."""
    rng = np.random.default_rng(1000 + seed)
    by_mouse = {}
    for r in train:
        by_mouse.setdefault(r["mouse"], []).append(r)
    mice = sorted(by_mouse)
    rng.shuffle(mice)
    return [by_mouse[m][rng.integers(len(by_mouse[m]))] for m in mice[:k]]


def load(r):
    z = np.load(os.path.join(DATA, r["id"] + ".npz"))
    return {k: z[k] for k in z.files}


def norm(ct):
    return (ct.astype(np.float32) + 400.0) / 700.0 - 1.0


def crop(arr, center, size=P, fill=0.0):
    out = np.full((size,) * 3, fill, dtype=arr.dtype)
    lo = [int(c) - size // 2 for c in center]
    src, dst = [], []
    for d in range(3):
        a0, a1 = max(lo[d], 0), min(lo[d] + size, arr.shape[d])
        src.append(slice(a0, a1))
        dst.append(slice(a0 - lo[d], a1 - lo[d]))
    out[tuple(dst)] = arr[tuple(src)]
    return out, lo


# ------------------------------------------------------------------ sampler
class Sampler:
    def __init__(self, scans, arm, rng):
        self.scans, self.arm, self.rng = scans, arm, rng
        self.tumor_vox = [np.argwhere(s["staple"] > 0) for s in scans]
        self.body_vox = [np.argwhere(s["ct"][::2, ::2, ::2] > -200) * 2 for s in scans]
        self.inserters = []
        if arm != "real":
            stats = [tumor_stats(s["ct"].astype(np.float32), s["staple"]) for s in scans]
            pooled = {"mean": float(np.mean([s["mean"] for s in stats])),
                      "std": float(np.mean([s["std"] for s in stats]))}
            for s in scans:
                ins = Inserter(s["ct"].astype(np.float32), s["staple"], pooled, arm, rng)
                self.inserters.append(ins if ins.ok() else None)

    def one(self):
        rng = self.rng
        i = rng.integers(len(self.scans))
        s = self.scans[i]
        u = rng.random()
        synth = self.arm != "real" and self.inserters[i] is not None
        if synth and u < 0.25:
            c, alpha, tex = self.inserters[i].sample()
            center = c + rng.integers(-12, 13, size=3)
            ct, lo = crop(s["ct"].astype(np.float32), center, fill=-400.0)
            lab, _ = crop(s["staple"], center)
            lab = lab.copy()
            local = np.array(c) - np.array(lo)
            from .synth import paste
            paste(ct, lab, local, alpha, tex)
        else:
            want_tumor = u < (0.5 if self.arm == "real" else 0.5) and len(self.tumor_vox[i])
            pool = self.tumor_vox[i] if want_tumor else self.body_vox[i]
            center = pool[rng.integers(len(pool))] + rng.integers(-8, 9, size=3)
            ct, _ = crop(s["ct"].astype(np.float32), center, fill=-400.0)
            lab, _ = crop(s["staple"], center)
        x = norm(ct)
        for ax in range(3):
            if rng.random() < 0.5:
                x, lab = np.flip(x, ax), np.flip(lab, ax)
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.08, 0.08)
        return np.ascontiguousarray(x), np.ascontiguousarray(lab)

    def batch(self, n):
        xs, ys = zip(*[self.one() for _ in range(n)])
        return (torch.from_numpy(np.stack(xs)[:, None]).float(),
                torch.from_numpy(np.stack(ys)[:, None].astype(np.float32)))


# ------------------------------------------------------------------ metrics
def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def lesion_recall(pred, ref):
    lab, n = ndimage.label(ref)
    if n == 0:
        return None
    hits = sum(bool(pred[lab == j].any()) for j in range(1, n + 1))
    return hits / n


def evaluate(net, test, dev="cpu"):
    net.eval()
    rows = []
    with torch.no_grad():
        for r in test:
            s = load(r)
            x = torch.from_numpy(norm(s["ct"]))[None, None]
            logit = sliding_window_inference(x, (P, P, P), 4, net, overlap=0.25, mode="gaussian")
            pred = (torch.sigmoid(logit)[0, 0].numpy() > 0.5)
            ref = s["staple"].astype(bool)
            v_pred, v_ref = pred.sum() * VOX_MM3, ref.sum() * VOX_MM3
            rows.append({
                "id": r["id"],
                "dice": dice(pred, ref),
                "dice_vs_annotators": float(np.mean([dice(pred, s[k]) for k in "abc"])),
                "vol_pred_mm3": float(v_pred), "vol_ref_mm3": float(v_ref),
                "vol_err_pct": float(abs(v_pred - v_ref) / max(v_ref, 1e-6) * 100),
                "lesion_recall": lesion_recall(pred, ref),
                "fp_components": int(ndimage.label(pred & ~ndimage.binary_dilation(ref, iterations=2))[1]),
            })
    net.train()
    agg = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
           for k in ["dice", "dice_vs_annotators", "vol_err_pct", "lesion_recall", "fp_components"]}
    agg["median_vol_err_pct"] = float(np.median([r["vol_err_pct"] for r in rows]))
    return agg, rows


def make_net():
    return UNet(3, 1, 1, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="0 = all training scans")
    ap.add_argument("--arm", choices=["real", "naive", "fauxgraft"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--max_test", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    name = f"{a.arm}_k{a.k if a.k else 'all'}_s{a.seed}"
    out = os.path.join(a.runs, name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "result.json")) or os.path.exists(os.path.join(out, "SKIP")):
        print("done already or skipped", name)
        return

    idx = load_index()
    train, test = split(idx)
    if a.max_test:
        test = test[: a.max_test]
    labeled = pick_labeled(train, a.k, a.seed) if a.k else train
    t0 = time.time()
    scans = [load(r) for r in labeled]
    sampler = Sampler(scans, a.arm, rng)
    t_setup = time.time() - t0

    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.iters, pct_start=0.1)
    loss_fn = DiceCELoss(sigmoid=True, batch=True)  # batch Dice: empty patches must not dominate
    losses = []
    t0 = time.time()
    for it in range(a.iters):
        x, y = sampler.batch(a.batch)
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (it + 1) % 250 == 0:
            print(f"{name} it {it+1} loss {np.mean(losses[-250:]):.4f} {time.time()-t0:.0f}s", flush=True)
    t_train = time.time() - t0

    t0 = time.time()
    agg, rows = evaluate(net, test)
    t_eval = time.time() - t0
    torch.save(net.state_dict(), os.path.join(out, "model.pt"))
    res = {"name": name, "arm": a.arm, "k": a.k, "seed": a.seed, "iters": a.iters,
           "labeled_ids": [r["id"] for r in labeled] if a.k else f"all {len(labeled)}",
           "n_test": len(test), "metrics": agg, "per_scan": rows,
           "loss_curve": [float(np.mean(losses[i:i+50])) for i in range(0, len(losses), 50)],
           "seconds": {"setup": t_setup, "train": t_train, "eval": t_eval}}
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(name, json.dumps(agg), f"train {t_train:.0f}s eval {t_eval:.0f}s", flush=True)


if __name__ == "__main__":
    main()
