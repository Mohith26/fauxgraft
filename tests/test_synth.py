"""Run: python3 tests/test_synth.py. Synthetic phantom only, no dataset needed."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fauxgraft.synth import VOXEL_MM, make_fauxgraft, make_naive, paste, Inserter

rng = np.random.default_rng(0)
# 1) generated volume tracks the requested volume (lumps add noise, ~±25%)
for target in (50.0, 300.0, 1200.0):
    vols = []
    for _ in range(10):
        alpha, _ = make_fauxgraft(rng, target, {"mean": 60.0, "std": 40.0})
        vols.append((alpha > 0.5).sum() * VOXEL_MM ** 3)
    assert 0.75 < np.median(vols) / target < 1.25, (target, np.median(vols))
alpha, tex = make_naive(rng, 300.0)
assert abs((alpha > 0.5).sum() * VOXEL_MM ** 3 / 300.0 - 1) < 0.15

# 2) paste clips safely at volume edges and labels match alpha > 0.5
ct = np.full((40, 40, 40), -400.0, np.float32); lab = np.zeros(ct.shape, np.uint8)
alpha, tex = make_fauxgraft(rng, 400.0, {"mean": 60.0, "std": 40.0})
paste(ct, lab, np.array([0, 20, 39]), alpha, tex)            # center on a corner/edge
assert lab.sum() > 0 and ct.max() > -400

# 3) on a cylinder "mouse" phantom, fakes land on the surface and bulge outward
ph = np.full((64, 64, 120), -400.0, np.float32)
yy, xx = np.mgrid[:64, :64]
body = (yy - 32) ** 2 + (xx - 32) ** 2 < 18 ** 2
ph[body] = 40.0
real = np.zeros(ph.shape, np.uint8)
ins = Inserter(ph, real, {"mean": 60.0, "std": 30.0}, "fauxgraft", rng)
assert ins.ok()
outside = []
for _ in range(10):
    img, l = ph.copy(), np.zeros(ph.shape, np.uint8)
    ins.insert(img, l)
    m = l.astype(bool)
    outside.append((m & ~np.broadcast_to(body[..., None], m.shape)).sum() / max(m.sum(), 1))
assert np.median(outside) > 0.3, np.median(outside)   # a real fraction protrudes
print("synth tests passed; median protruding fraction %.2f" % np.median(outside))
