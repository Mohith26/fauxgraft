"""Fauxgraft: paint fake subcutaneous tumors into real mouse micro-CT scans.

Two generators, so we can test whether realism matters:

  naive      a perfect sphere of constant 50 HU, hard edge, dropped at a random
             point inside the body. The "just paste blobs" baseline.

  fauxgraft  built from what a subcutaneous xenograft actually looks like:
             - sits on the skin surface of the flank and bulges outward
             - lumpy ellipsoid (random axes, rotation, smooth radial noise)
             - soft-tissue intensity and texture matched to the real labeled
               tumors the model is already allowed to see
             - partial-volume edge blending instead of a hard cut

Everything here works on one small box around the tumor, so inserting a fake
tumor into a 64^3 training patch costs about a millisecond.
"""

import numpy as np
from scipy import ndimage

VOXEL_MM = 0.42
AIR_HU = -400.0
BODY_HU = -200.0  # body mask threshold


def body_mask(ct):
    m = ct > BODY_HU
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:  # keep the largest component (drops bed / tube fragments)
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def surface_candidates(ct, avoid=None, avoid_mm=4.0, z_range=(0.25, 0.75)):
    """Skin-surface voxels on the trunk, optionally away from known tumors.

    Returns (points [N,3], outward normals [N,3]).
    """
    body = body_mask(ct)
    surf = body & ~ndimage.binary_erosion(body, iterations=1)
    zs = np.nonzero(body.any(axis=(0, 1)))[0]
    z0, z1 = zs.min(), zs.max()
    lo, hi = z0 + z_range[0] * (z1 - z0), z0 + z_range[1] * (z1 - z0)
    surf[:, :, : int(lo)] = False
    surf[:, :, int(hi):] = False
    if avoid is not None and avoid.any():
        dist = ndimage.distance_transform_edt(~avoid.astype(bool)) * VOXEL_MM
        surf &= dist > avoid_mm
    pts = np.argwhere(surf)
    # outward normal = minus gradient of a smoothed body mask
    sm = ndimage.gaussian_filter(body.astype(np.float32), 2.0)
    g = np.stack([ndimage.sobel(sm, axis=a) for a in range(3)], -1)
    n = -g[pts[:, 0], pts[:, 1], pts[:, 2]]
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
    return pts, n, body


def interior_candidates(body, step=3):
    return np.argwhere(body[::step, ::step, ::step]) * step


def tumor_stats(ct, mask):
    """Mean/std HU and a noise-texture estimate of labeled tumors."""
    vals = ct[mask.astype(bool)]
    if vals.size < 50:
        return {"mean": 60.0, "std": 45.0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


def sample_volume_mm3(rng, lo=30.0, hi=1500.0):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def make_fauxgraft(rng, volume_mm3, stats, box=None):
    """Return (alpha, texture) arrays on a cubic box, tumor centered.

    alpha in [0,1] is the soft occupancy (partial-volume edge); texture is HU.
    """
    r_eq = (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3) / VOXEL_MM  # voxels
    axes = r_eq * rng.uniform(0.75, 1.3, size=3)
    axes *= r_eq / np.prod(axes) ** (1 / 3)                         # keep volume
    half = int(np.ceil(axes.max() * 1.45)) + 2 if box is None else box // 2
    g = np.mgrid[-half:half + 1, -half:half + 1, -half:half + 1].astype(np.float32)
    R = _random_rotation(rng)
    p = np.tensordot(R, g, axes=1)
    rho = np.sqrt(sum((p[i] / axes[i]) ** 2 for i in range(3)))   # 1 on surface
    lumps = ndimage.gaussian_filter(rng.normal(size=rho.shape).astype(np.float32),
                                    sigma=max(1.5, r_eq / 2.5))
    lumps /= lumps.std() + 1e-6
    rho = rho * (1 + 0.12 * lumps)
    sdf = (rho - 1.0) * r_eq                                        # ~voxels
    alpha = 1 / (1 + np.exp(sdf / 0.6))
    tex = ndimage.gaussian_filter(rng.normal(size=rho.shape).astype(np.float32), 0.8)
    tex = tex / (tex.std() + 1e-6) * stats["std"] * rng.uniform(0.55, 0.9)
    mean = stats["mean"] + rng.normal(0, 10)
    # slightly brighter rim + darker core is common in larger xenografts
    core = np.clip(1 - rho, 0, 1)
    texture = mean + tex - 25 * core * (r_eq > 6) * rng.uniform(0, 1)
    return alpha.astype(np.float32), texture.astype(np.float32)


def make_naive(rng, volume_mm3):
    r = (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3) / VOXEL_MM
    half = int(np.ceil(r)) + 2
    g = np.mgrid[-half:half + 1, -half:half + 1, -half:half + 1].astype(np.float32)
    alpha = (np.sqrt((g ** 2).sum(0)) <= r).astype(np.float32)
    return alpha, np.full_like(alpha, 50.0)


def paste(ct, label, center, alpha, texture, body=None):
    """Blend a tumor box into ct/label in place at integer center (clipped)."""
    h = alpha.shape[0] // 2
    sl_dst, sl_src = [], []
    for d in range(3):
        a0, a1 = center[d] - h, center[d] + h + 1
        s0, s1 = 0, alpha.shape[d]
        if a0 < 0:
            s0, a0 = -a0, 0
        if a1 > ct.shape[d]:
            s1 -= a1 - ct.shape[d]
            a1 = ct.shape[d]
        if a1 <= a0:
            return
        sl_dst.append(slice(a0, a1))
        sl_src.append(slice(s0, s1))
    sd, ss = tuple(sl_dst), tuple(sl_src)
    al = alpha[ss]
    ct[sd] = (1 - al) * ct[sd] + al * texture[ss]
    label[sd] = np.maximum(label[sd], (al > 0.5).astype(label.dtype))


class Inserter:
    """Per-scan cache of where a fake tumor may go, plus the insert operation."""

    def __init__(self, ct, real_mask, stats, mode, rng):
        self.mode, self.stats, self.rng = mode, stats, rng
        pts, normals, body = surface_candidates(ct, avoid=real_mask)
        self.pts, self.normals = pts, normals
        self.interior = interior_candidates(body & ~ndimage.binary_dilation(
            real_mask.astype(bool), iterations=8))

    def ok(self):
        return len(self.pts) > 0 and len(self.interior) > 0

    def sample(self):
        """Pick a location and build one fake tumor: (center, alpha, texture)."""
        rng = self.rng
        vol = sample_volume_mm3(rng)
        if self.mode == "naive":
            c = self.interior[rng.integers(len(self.interior))]
            alpha, tex = make_naive(rng, vol)
        else:
            i = rng.integers(len(self.pts))
            r_eq = (3 * vol / (4 * np.pi)) ** (1 / 3) / VOXEL_MM
            c = np.round(self.pts[i] + self.normals[i] * r_eq * rng.uniform(0.1, 0.6)).astype(int)
            alpha, tex = make_fauxgraft(rng, vol, self.stats)
        return c, alpha, tex

    def insert(self, ct, label):
        """Insert one fake tumor into full-scan copies; return its center."""
        rng = self.rng
        vol = sample_volume_mm3(rng)
        if self.mode == "naive":
            c = self.interior[rng.integers(len(self.interior))]
            alpha, tex = make_naive(rng, vol)
        else:
            i = rng.integers(len(self.pts))
            r_eq = (3 * vol / (4 * np.pi)) ** (1 / 3) / VOXEL_MM
            # push center outward so most of the tumor bulges out of the body
            c = np.round(self.pts[i] + self.normals[i] * r_eq * rng.uniform(0.1, 0.6)).astype(int)
            alpha, tex = make_fauxgraft(rng, vol, self.stats)
        paste(ct, label, c, alpha, tex)
        return c
