"""Convert the TumSeg NIfTI database into compact 0.42 mm numpy volumes.

    python3 scripts/preprocess.py --src "/path/to/TumSeg database" --out data/

For every scan this writes data/<id>.npz with:
    ct      float16, HU clipped to [-400, 1000] (the dataset's own range), 2x downsampled
    staple  uint8, STAPLE consensus of the 3 annotators (thresholded at 0.5)
    a,b,c   uint8, each annotator's own mask
and data/index.json with one row per scan (id, dataset, mouse, timepoint, voxel
count, tumor volume in mm^3 at full resolution).

Why downsample: training runs on a laptop CPU. At 0.42 mm a typical tumor is
still ~2,000 voxels across ~14 voxels, enough for Dice and volume to be
meaningful, and an epoch is 8x cheaper.
"""

import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import nibabel as nib
import numpy as np

FULL_VOXEL_MM3 = 0.21 ** 3


def pool2(x):
    """2x2x2 mean pool, cropping odd edges."""
    s = [d - d % 2 for d in x.shape]
    x = x[: s[0], : s[1], : s[2]]
    return x.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean(axis=(1, 3, 5))


def one(args):
    ct_path, out_dir = args
    folder = os.path.dirname(ct_path)
    tag = os.path.basename(folder)                       # e.g. M39_0d
    dataset = int(re.search(r"Dataset (\d+)", ct_path).group(1))
    mouse, tp = tag.split("_", 1)
    sid = f"D{dataset:02d}_{tag}"
    ct = nib.load(ct_path).get_fdata(dtype=np.float32)
    rec = {"id": sid, "dataset": dataset, "mouse": f"D{dataset:02d}_{mouse}",
           "mouse_id": mouse, "timepoint": tp, "shape_full": list(ct.shape)}
    out = {"ct": pool2(ct).astype(np.float16)}
    for key, prefix in [("staple", "STAPLE"), ("a", "Annotator_A"),
                        ("b", "Annotator_B"), ("c", "Annotator_C")]:
        p = os.path.join(folder, f"{prefix}_{tag}.nii.gz")
        if not os.path.exists(p):
            return None
        m = nib.load(p).get_fdata(dtype=np.float32) > 0.5
        rec[f"vol_mm3_{key}"] = float(m.sum() * FULL_VOXEL_MM3)
        out[key] = (pool2(m.astype(np.float32)) > 0.5).astype(np.uint8)
    rec["shape"] = list(out["ct"].shape)
    np.savez_compressed(os.path.join(out_dir, sid + ".npz"), **out)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cts = sorted(glob.glob(os.path.join(a.src, "*", "*", "*", "CT_*.nii.gz")))
    with Pool(a.workers) as p:
        recs = [r for r in p.map(one, [(c, a.out) for c in cts]) if r]
    recs.sort(key=lambda r: r["id"])
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(recs, f, indent=1)
    print(f"{len(recs)} scans from {len(cts)} CT files, {len({r['mouse'] for r in recs})} mice")


if __name__ == "__main__":
    main()
