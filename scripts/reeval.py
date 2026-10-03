"""Re-evaluate saved sweep models with a size filter on predicted blobs.

    python3 scripts/reeval.py --workers 4

Why: models trained with fake tumors find more real tumors but also fire on
small tumor-like specks. A standard fix in lesion segmentation is to drop
connected components below a minimum volume. We report both raw and filtered
metrics side by side. Min size is 10 mm^3: 565 of the dataset's 570 consensus
tumor components (99.1%) are larger, and most of the other 5 are sub-1 mm^3
annotation specks. The threshold was fixed from that dataset fact, not tuned
on test results.
Writes runs/<name>/reeval.json.
"""

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fauxgraft.train import P, VOX_MM3, dice, lesion_recall, load, load_index, make_net, norm, split  # noqa: E402

MIN_MM3 = 10.0


def filt(pred):
    lab, n = ndimage.label(pred)
    if n == 0:
        return pred
    sizes = ndimage.sum(pred, lab, range(1, n + 1)) * VOX_MM3
    return np.isin(lab, np.nonzero(sizes >= MIN_MM3)[0] + 1)


def metrics(pred, s):
    ref = s["staple"].astype(bool)
    vp, vr = pred.sum() * VOX_MM3, ref.sum() * VOX_MM3
    return {"dice": dice(pred, ref), "vol_err_pct": float(abs(vp - vr) / max(vr, 1e-6) * 100),
            "lesion_recall": lesion_recall(pred, ref),
            "fp_components": int(ndimage.label(pred & ~ndimage.binary_dilation(ref, iterations=2))[1])}


def run(run_dir):
    out = os.path.join(run_dir, "reeval.json")
    if os.path.exists(out) or not os.path.exists(os.path.join(run_dir, "model.pt")):
        return
    torch.set_num_threads(4)
    net = make_net()
    net.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    net.eval()
    _, test = split(load_index())
    rows = []
    with torch.no_grad():
        for r in test:
            s = load(r)
            x = torch.from_numpy(norm(s["ct"]))[None, None]
            prob = torch.sigmoid(sliding_window_inference(x, (P, P, P), 4, net, overlap=0.25, mode="gaussian"))[0, 0].numpy()
            raw = prob > 0.5
            rows.append({"id": r["id"], "raw": metrics(raw, s), "filtered": metrics(filt(raw), s)})
    agg = {}
    for v in ("raw", "filtered"):
        agg[v] = {k: float(np.mean([x[v][k] for x in rows if x[v][k] is not None]))
                  for k in ("dice", "vol_err_pct", "lesion_recall", "fp_components")}
        agg[v]["median_vol_err_pct"] = float(np.median([x[v]["vol_err_pct"] for x in rows]))
    with open(out, "w") as f:
        json.dump({"min_mm3": MIN_MM3, "metrics": agg, "per_scan": rows}, f)
    print(os.path.basename(run_dir), json.dumps(agg["filtered"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", default=os.environ.get("FAUXGRAFT_RUNS", os.path.join(ROOT, "runs")))
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.runs, "*")) if os.path.exists(os.path.join(d, "result.json")))
    with Pool(a.workers) as p:
        p.map(run, dirs, chunksize=1)
