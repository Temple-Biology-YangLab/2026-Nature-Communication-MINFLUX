# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-camera registration from bead acquisitions, and CamT -> CamR mapping."""
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import maximum_filter, gaussian_filter

from .config import RegisterConfig


class SRImage:
    """Histogram image plus the nm<->pixel mapping."""

    def __init__(self, img, x0, y0, pixrec):
        self.img, self.x0, self.y0, self.pixrec = img, x0, y0, pixrec

    def px_to_nm(self, ix, iy):
        return self.x0 + np.asarray(ix) * self.pixrec, self.y0 + np.asarray(iy) * self.pixrec

    def nm_to_px(self, x, y):
        return (np.asarray(x) - self.x0) / self.pixrec, (np.asarray(y) - self.y0) / self.pixrec

    @property
    def shape(self):
        return self.img.shape


def render_hist(x, y, pixrec, bounds=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if bounds is None:
        x0, y0 = float(x.min()), float(y.min())
        x1, y1 = float(x.max()), float(y.max())
    else:
        x0, x1, y0, y1 = bounds
    nx = max(2, int(np.ceil((x1 - x0) / pixrec)) + 1)
    ny = max(2, int(np.ceil((y1 - y0) / pixrec)) + 1)
    H, _, _ = np.histogram2d(y, x, bins=[ny, nx],
                             range=[[y0, y0 + ny * pixrec], [x0, x0 + nx * pixrec]])
    return SRImage(H, x0, y0, pixrec)


class Transform:
    """Maps source (CamT) coordinates onto target (CamR) coordinates."""

    def __init__(self, kind, params, residual_nm=None, n_pairs=0, units='nm'):
        self.kind = kind
        self.params = params
        self.residual_nm = residual_nm
        self.n_pairs = n_pairs
        self.units = units

    def apply(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        p = self.params
        if self.kind == 'translation':
            return x + p[0], y + p[1]
        if self.kind == 'rigid':
            dx, dy, th, s = p
            c, sn = np.cos(th), np.sin(th)
            return s * (c * x - sn * y) + dx, s * (sn * x + c * y) + dy
        if self.kind == 'affine':
            a = np.asarray(p).reshape(2, 3)
            return (a[0, 0] * x + a[0, 1] * y + a[0, 2],
                    a[1, 0] * x + a[1, 1] * y + a[1, 2])
        if self.kind == 'poly2':
            a = np.asarray(p).reshape(2, 6)
            f = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y])
            return a[0] @ f, a[1] @ f
        raise ValueError(self.kind)

    def __repr__(self):
        r = 'n/a' if self.residual_nm is None else f'{self.residual_nm:.1f} nm'
        return f'<Transform {self.kind} pairs={self.n_pairs} residual={r}>'


def detect_beads(img, threshold_sigma=5.0, min_sep_px=5.0):
    """Local maxima well above background."""
    im = np.asarray(img, float)
    sm = gaussian_filter(im, 1.0)
    bg = np.median(sm)
    mad = np.median(np.abs(sm - bg)) * 1.4826
    thr = bg + threshold_sigma * (mad if mad > 0 else sm.std())
    size = int(max(3, round(min_sep_px) | 1))
    mx = maximum_filter(sm, size=size)
    iy, ix = np.nonzero((sm == mx) & (sm >= thr))
    return np.column_stack([ix, iy]).astype(float)


def fit_bead(img, cx, cy, roi=7):
    """2D Gaussian centroid refinement. Returns (x, y, amp) in pixels."""
    h = roi // 2
    y0, y1 = int(cy) - h, int(cy) + h + 1
    x0, x1 = int(cx) - h, int(cx) + h + 1
    if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
        return None
    sub = np.asarray(img[y0:y1, x0:x1], float)
    sub = sub - sub.min()
    if sub.sum() <= 0:
        return None
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]

    def resid(p):
        a, mx_, my_, s, b = p
        return (a * np.exp(-((xx - mx_) ** 2 + (yy - my_) ** 2) / (2 * s ** 2)) + b - sub).ravel()

    p0 = [sub.max(), h, h, 1.5, 0.0]
    try:
        r = least_squares(resid, p0, max_nfev=500)
    except Exception:
        return None
    a, mx_, my_, s, b = r.x
    if not (0 <= mx_ < sub.shape[1] and 0 <= my_ < sub.shape[0]):
        return None
    return x0 + mx_, y0 + my_, a


def beads_from_image(img, cfg):
    """Detect then fit every bead in an image. Returns Nx2 pixel coordinates."""
    cand = detect_beads(img, cfg.bead_threshold_sigma, cfg.bead_min_sep_px)
    out = []
    for cx, cy in cand:
        f = fit_bead(img, cx, cy, cfg.bead_fit_roi_px)
        if f is not None:
            out.append(f[:2])
    return np.array(out) if out else np.zeros((0, 2))


def _mirror(P, shape, mode):
    if mode in (None, 'none'):
        return P
    Q = P.copy()
    if mode == 'updown':
        Q[:, 1] = shape[0] - 1 - Q[:, 1]
    elif mode == 'leftright':
        Q[:, 0] = shape[1] - 1 - Q[:, 0]
    else:
        raise ValueError(mode)
    return Q


def pair_points(A, B, max_dist):
    """Mutual nearest-neighbour pairing — rejects the one-sided mismatches that
    inflate a naive nearest-neighbour residual.
    """
    from scipy.spatial import cKDTree
    if len(A) == 0 or len(B) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    ta, tb = cKDTree(A), cKDTree(B)
    da, ja = tb.query(A, k=1)
    db, jb = ta.query(B, k=1)
    keep = [i for i in range(len(A)) if da[i] <= max_dist and jb[ja[i]] == i]
    return A[keep], B[ja[keep]]


def solve_transform(src, dst, kind='affine'):
    """Least-squares transform from src onto dst. Both Nx2, same units."""
    x, y = src[:, 0], src[:, 1]
    if kind == 'translation':
        p = [np.mean(dst[:, 0] - x), np.mean(dst[:, 1] - y)]
    elif kind == 'rigid':
        def resid(p):
            dx, dy, th, s = p
            c, sn = np.cos(th), np.sin(th)
            return np.concatenate([s * (c * x - sn * y) + dx - dst[:, 0],
                                   s * (sn * x + c * y) + dy - dst[:, 1]])
        p = least_squares(resid, [0, 0, 0, 1]).x
    elif kind == 'affine':
        M = np.column_stack([x, y, np.ones_like(x)])
        cx, *_ = np.linalg.lstsq(M, dst[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(M, dst[:, 1], rcond=None)
        p = np.concatenate([cx, cy])
    elif kind == 'poly2':
        M = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
        cx, *_ = np.linalg.lstsq(M, dst[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(M, dst[:, 1], rcond=None)
        p = np.concatenate([cx, cy])
    else:
        raise ValueError(kind)
    T = Transform(kind, p)
    px, py = T.apply(x, y)
    res = np.hypot(px - dst[:, 0], py - dst[:, 1])
    T.residual_nm = float(np.median(res))
    T.n_pairs = len(src)
    T.residuals = res
    return T


def register_from_images(img_src, img_dst, cfg):
    """Bead-based registration between two camera images."""
    A = beads_from_image(img_src, cfg)
    B = beads_from_image(img_dst, cfg)
    A = _mirror(A, img_src.shape, cfg.mirror)
    if len(A) < cfg.min_beads or len(B) < cfg.min_beads:
        raise ValueError(f'too few beads: src={len(A)} dst={len(B)} '
                         f'(need {cfg.min_beads}). Lower bead_threshold_sigma?')
    px = cfg.cam_pixelsize_nm
    a, b = pair_points(A * px, B * px, cfg.bead_max_pair_nm)
    if len(a) < cfg.min_beads:
        raise ValueError(f'only {len(a)} mutual pairs within '
                         f'{cfg.bead_max_pair_nm} nm — check the mirror mode')
    T = solve_transform(a, b, cfg.model)
    if T.residual_nm > cfg.max_residual_nm:
        import warnings
        warnings.warn(f'registration residual {T.residual_nm:.1f} nm exceeds '
                      f'{cfg.max_residual_nm} nm — comparable to the structure '
                      f'being measured')
    return T


def register_auto(img_src, img_dst, cfg, thresholds=(8.0, 5.0, 3.0, 2.0), min_pairs=6):
    """Sweep the bead detection threshold and keep the LOWEST-RESIDUAL solution."""
    best = None
    for thr in thresholds:
        c = replace(cfg, bead_threshold_sigma=thr)
        try:
            T = register_from_images(img_src, img_dst, c)
        except Exception:
            continue
        if T.n_pairs >= min_pairs and (best is None or T.residual_nm < best.residual_nm):
            best = T
            best.threshold_sigma = thr
    if best is None:
        raise ValueError('no threshold produced a usable registration')
    return best


def check_mirror(img_src, img_dst, cfg):
    """Try every mirroring modality and report residuals."""
    out = {}
    for m in ('none', 'updown', 'leftright'):
        c = replace(cfg, mirror=m)
        try:
            out[m] = register_from_images(img_src, img_dst, c).residual_nm
        except Exception as e:
            out[m] = f'failed: {e}'
    return out


def fit_shared_linear(pair_sets):
    """Joint fit: ONE linear part shared by all acquisitions, a SEPARATE
    translation for each.
    """
    sets = [(np.asarray(s, float), np.asarray(d, float))
            for s, d in pair_sets if len(s) >= 3]
    if not sets:
        raise ValueError('no usable pair sets')
    k = len(sets)
    n = sum(len(s) for s, _ in sets)
    A = np.zeros((2 * n, 4 + 2 * k))
    b = np.zeros(2 * n)
    row = 0
    for j, (s, d) in enumerate(sets):
        for i in range(len(s)):
            x, y = s[i]
            A[row, 0:2] = (x, y)          # L[0,0], L[0,1]
            A[row, 4 + 2 * j] = 1.0       # t_k x
            b[row] = d[i, 0]
            A[row + 1, 2:4] = (x, y)      # L[1,0], L[1,1]
            A[row + 1, 5 + 2 * j] = 1.0   # t_k y
            b[row + 1] = d[i, 1]
            row += 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    L = sol[:4].reshape(2, 2)
    ts = [sol[4 + 2 * j: 6 + 2 * j] for j in range(k)]
    res = []
    for j, (s, d) in enumerate(sets):
        p = s @ L.T + ts[j]
        res += list(np.linalg.norm(p - d, axis=1))
    res = np.array(res)
    return dict(linear=L, translations=ts, n_pairs=int(n), n_sets=k,
                residual_nm=float(np.median(res)),
                residual_p90_nm=float(np.percentile(res, 90)))


def bead_pairs(img_src, img_dst, cfg):
    """Detected and paired bead coordinates in nm, for fit_shared_linear."""
    A = beads_from_image(img_src, cfg)
    B = beads_from_image(img_dst, cfg)
    A = _mirror(A, img_src.shape, cfg.mirror)
    if len(A) < cfg.min_beads or len(B) < cfg.min_beads:
        raise ValueError(f'too few beads: src={len(A)} dst={len(B)}')
    px = cfg.cam_pixelsize_nm
    return pair_points(A * px, B * px, cfg.bead_max_pair_nm)


def split_transform(T):
    """Separate a transform into its linear part and its translation."""
    if T.kind == 'affine':
        a = np.asarray(T.params).reshape(2, 3)
        lin = a[:, :2].copy()
        trans = a[:, 2].copy()
    elif T.kind == 'translation':
        lin = np.eye(2)
        trans = np.asarray(T.params, float)
    else:
        raise ValueError(f'split_transform not defined for {T.kind}')
    return lin, trans


def make_affine(lin, trans):
    """Rebuild an affine Transform from a linear part and a translation."""
    p = np.concatenate([np.asarray(lin)[0], [trans[0]],
                        np.asarray(lin)[1], [trans[1]]])
    p = np.array([lin[0][0], lin[0][1], trans[0],
                  lin[1][0], lin[1][1], trans[1]], float)
    return Transform('affine', p)


def residual_translation(src_x, src_y, dst_x, dst_y, pixrec_nm=40.0,
                         max_shift_nm=1500.0, bounds=None, refine_nm=None):
    """Measure the leftover translation between two point clouds, by
    cross-correlation. Returns (dx, dy) in nm to ADD to the source.
    """
    from .drift import _xcorr_shift
    src_x = np.asarray(src_x, float)
    src_y = np.asarray(src_y, float)
    dst_x = np.asarray(dst_x, float)
    dst_y = np.asarray(dst_y, float)
    if bounds is None:
        bounds = (min(src_x.min(), dst_x.min()), max(src_x.max(), dst_x.max()),
                  min(src_y.min(), dst_y.min()), max(src_y.max(), dst_y.max()))
    A = render_hist(dst_x, dst_y, pixrec_nm, bounds).img
    B = render_hist(src_x, src_y, pixrec_nm, bounds).img
    dx, dy = _xcorr_shift(A, B, max_shift_nm / pixrec_nm)
    tx, ty = float(dx * pixrec_nm), float(dy * pixrec_nm)

    if refine_nm and refine_nm < pixrec_nm:
        span = 1.5 * pixrec_nm
        A2 = render_hist(dst_x, dst_y, refine_nm, bounds).img
        B2 = render_hist(src_x + tx, src_y + ty, refine_nm, bounds).img
        ddx, ddy = _xcorr_shift(A2, B2, span / refine_nm)
        tx += float(ddx * refine_nm)
        ty += float(ddy * refine_nm)
    return tx, ty


def align_camt_to_camr(camt_x, camt_y, bead_transform, drift_curve=None,
                       frame=None, mode=None):
    """CamT -> CamR for ONE acquisition. Two corrections, two sources."""
    from .roi import UNREGISTERED

    x = np.asarray(camt_x, float)
    y = np.asarray(camt_y, float)

    if bead_transform is None:
        raise ValueError(
            'no bead transform for this acquisition — the camera offset is '
            'unmeasured. Registering without it would mean inventing the '
            'offset, so the acquisition is skipped instead.')

    lin, trans = split_transform(bead_transform)
    x, y = (lin[0, 0] * x + lin[0, 1] * y + trans[0],
            lin[1, 0] * x + lin[1, 1] * y + trans[1])

    drift_applied = False
    n_extrapolated = 0
    frame_bin = 1.0
    if drift_curve is not None and frame is not None:
        try:
            x, y, extrap = drift_curve.apply(x, y, frame, return_flags=True)
            n_extrapolated = int(np.count_nonzero(extrap))
        except TypeError:
            x, y = drift_curve.apply(x, y, frame)
        frame_bin = float(getattr(drift_curve, 'frame_bin', 1.0))
        drift_applied = True

    return x, y, dict(linear=lin,
                      bead_translation_nm=(float(trans[0]), float(trans[1])),
                      drift_applied=drift_applied,
                      n_drift_extrapolated=n_extrapolated,
                      drift_frame_bin=frame_bin,
                      mode_used=(mode or UNREGISTERED))


def channel_overlap_offset(camt_x, camt_y, camr_x, camr_y, pixrec_nm=40.0,
                           max_shift_nm=1500.0):
    """DIAGNOSTIC ONLY — never feed this back into the coordinates."""
    return residual_translation(camt_x, camt_y, camr_x, camr_y, pixrec_nm,
                                max_shift_nm, refine_nm=10.0)


def register_from_locs(src_locs, dst_locs, cfg):
    """Cross-correlation registration, two refinement passes."""
    from .drift import _xcorr_shift
    sx = np.asarray(src_locs['xnm'], float); sy = np.asarray(src_locs['ynm'], float)
    dx_ = np.asarray(dst_locs['xnm'], float); dy_ = np.asarray(dst_locs['ynm'], float)
    bounds = (min(sx.min(), dx_.min()), max(sx.max(), dx_.max()),
              min(sy.min(), dy_.min()), max(sy.max(), dy_.max()))
    tx = ty = 0.0
    for pixrec, maxsh in zip(cfg.cc_pixrec_nm, cfg.cc_max_shift_nm):
        a = render_hist(dx_, dy_, pixrec, bounds).img
        b = render_hist(sx + tx, sy + ty, pixrec, bounds).img
        ddx, ddy = _xcorr_shift(a, b, maxsh / pixrec)
        tx += ddx * pixrec
        ty += ddy * pixrec
    return Transform('translation', [tx, ty], residual_nm=None, n_pairs=0)


def read_tif(path, frame=0):
    """Read one frame of a tif. Prefers tifffile, falls back to PIL."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            return t.pages[frame].asarray().astype(float)
    except ImportError:
        from PIL import Image
        im = Image.open(path)
        im.seek(frame)
        return np.asarray(im, float)


def sum_projection(path, max_frames=200):
    """Sum projection — beads are static so this raises them well above noise."""
    try:
        import tifffile
    except ImportError:
        raise ImportError('tifffile required for sum_projection')
    acc = None
    with tifffile.TiffFile(path) as t:
        for i, pg in enumerate(t.pages):
            if i >= max_frames:
                break
            a = pg.asarray().astype(float)
            acc = a if acc is None else acc + a
    return acc
