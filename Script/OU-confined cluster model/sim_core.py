# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""OU-confined cluster model: simulation, calibration and analysis pipeline.
Lengths in nm, times in s.
"""
import math
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import lfilter
from sklearn.neighbors import NearestNeighbors
from skimage.feature import peak_local_max

RANDOM_SEED = 20261030
CONSTRUCTS = ['GFP-Nup58-HaloTag', 'HaloTag-Nup54-GFP', 'HaloTag-Nup58-GFP', 'HaloTag-Nup62-GFP']
SHORT = {'GFP-Nup58-HaloTag': 'Nup58 C', 'HaloTag-Nup54-GFP': 'Nup54 N',
         'HaloTag-Nup58-GFP': 'Nup58 N', 'HaloTag-Nup62-GFP': 'Nup62 N'}

CONSTRUCT = {
    'GFP-Nup58-HaloTag': dict(dt=0.000359, noise=(10.02, 9.37, 3.36), tau_xy=0.010, tau_z_old=0.002722,
                              z_a=18.29, z_tau_a=0.001272, z_b=11.26, z_tau_b=0.007045, target_sd=(15.30, 15.78, 20.14)),
    'HaloTag-Nup54-GFP': dict(dt=0.000365, noise=(9.53, 9.14, 2.18), tau_xy=0.006, tau_z_old=0.002792,
                              z_a=19.73, z_tau_a=0.001456, z_b=8.50, z_tau_b=0.018028, target_sd=(14.31, 14.07, 20.34)),
    'HaloTag-Nup58-GFP': dict(dt=0.000371, noise=(9.19, 8.72, 0.00), tau_xy=0.010, tau_z_old=0.004048,
                              z_a=21.42, z_tau_a=0.002048, z_b=10.73, z_tau_b=0.023942, target_sd=(14.16, 14.21, 22.14)),
    'HaloTag-Nup62-GFP': dict(dt=0.000360, noise=(9.30, 8.78, 0.00), tau_xy=0.004, tau_z_old=0.004117,
                              z_a=19.53, z_tau_a=0.001546, z_b=7.77, z_tau_b=0.015022, target_sd=(12.80, 12.90, 20.28)),
}

# Analysis settings (manuscript pipeline, unchanged)
ANALYSIS = dict(VOX_NM=1.0, GRID_MAX_NM=120.0, CENTER_NM=60.0, K_NN=3, BANDWIDTH_FIXED_NM=None,
                SLICE_SPACING_NM=10.0, SLICE_THICKNESS_NM=1.0, ZOOM_MIN_NM=30.0, ZOOM_MAX_NM=90.0,
                GLOBAL_NOISE_CUTOFF=0.20, SLICE_RELATIVE_THRESHOLD=0.50, MIN_PEAK_DISTANCE_VOX=2,
                SUBVOXEL_REFINEMENT=True, SUBVOXEL_WINDOW_XY_NM=2.0, SUBVOXEL_STEP_XY_NM=0.1, SUBVOXEL_STEP_Z_NM=0.25,
                MIN_PEAKS_FOR_FIT=3, ANNULUS_BAND_HALF_WIDTH_NM=1.0)


# Configuration and calibration
def make_config(construct, n_clusters, ring_radius_nm=10.0, dwell_s=0.025, n_localizations=2000):
    p = CONSTRUCT[construct]
    c = dict(ANALYSIS, CONSTRUCT=construct, CONSTRUCT_INDEX=CONSTRUCTS.index(construct), RANDOM_SEED=RANDOM_SEED,
             N_CLUSTERS=int(n_clusters), RING_RADIUS_NM=float(ring_radius_nm), DWELL_S=float(dwell_s),
             N_LOCALIZATIONS=int(n_localizations), DT_S=p['dt'], NOISE_SD_NM=p['noise'],
             TAU_S=(p['tau_xy'], p['tau_xy'], p['tau_z_old']), Z=p, POINT_HANDLING='keep_all')
    c['CLUSTER_SD_NM'], c['Z_SCALE'], c['ACHIEVED_SD_NM'] = calibrate(c)
    return c


def _observed_sd(c, cluster_sd, z_scale, seeds):
    cc = dict(c, CLUSTER_SD_NM=tuple(cluster_sd), Z_SCALE=z_scale, POINT_HANDLING='crop')
    return np.median([preprocess(simulate_trace(cc, np.random.default_rng(s)), cc).astype(float).std(0, ddof=1)
                      for s in seeds], axis=0)


def calibrate(c, n_traces=80, n_iter=10):
    """Bisection on cluster size (x, y) and axial amplitude so the median per-trace SD matches the experiment."""
    target = np.asarray(CONSTRUCT[c['CONSTRUCT']]['target_sd'], float)
    seeds = [np.random.SeedSequence([c['RANDOM_SEED'], 777, c['CONSTRUCT_INDEX'], i]) for i in range(n_traces)]
    lo, hi, zlo, zhi = np.zeros(3), np.full(3, 40.0), 0.0, 2.0
    for _ in range(n_iter):
        mid, zmid = (lo + hi) / 2, (zlo + zhi) / 2
        got = _observed_sd(c, mid, zmid, seeds)
        lo = np.where(got < target, mid, lo); hi = np.where(got < target, hi, mid)
        zlo, zhi = (zmid, zhi) if got[2] < target[2] else (zlo, zmid)
    sd = np.maximum((lo + hi) / 2, 1.0); zs = (zlo + zhi) / 2
    return tuple(sd.tolist()), zs, tuple(_observed_sd(c, sd, zs, seeds).round(3).tolist())


def trace_seed(c, rep):
    tag = int(round(c['RING_RADIUS_NM'] * 1000)) + 7919 * c['N_CLUSTERS'] + int(round(c['DWELL_S'] * 1e6))
    return np.random.SeedSequence([c['RANDOM_SEED'], c['CONSTRUCT_INDEX'], rep, tag])


# Simulation
def _ou_path(n, sd, kappa, rng):
    """Stationary OU path; kappa = relaxation time in steps."""
    a = math.exp(-1.0 / kappa); x = np.empty(n); x[0] = rng.normal() * sd
    e = rng.normal(size=n - 1) * sd * math.sqrt(1 - a * a)
    x[1:] = lfilter([1.0], [1.0, -a], e, zi=[a * x[0]])[0]
    return x


def simulate_trace(c, rng, return_truth=False):
    """Cluster centres on a ring; OU confinement around the current centre; Markov switching; localization error."""
    n, dt = c['N_LOCALIZATIONS'], c['DT_S']
    sd = np.asarray(c['CLUSTER_SD_NM']); tau = np.asarray(c['TAU_S'])
    a = np.exp(-dt / tau); s_in = sd * np.sqrt(1 - a ** 2)
    K, R = c['N_CLUSTERS'], c['RING_RADIUS_NM']
    phi = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(K) / K
    C = np.c_[R * np.cos(phi), R * np.sin(phi), np.zeros(K)]
    # which cluster the terminus is attached to at each step
    state = np.empty(n, int); state[0] = rng.integers(K)
    sw = rng.random(n - 1) < 1 - math.exp(-dt / c['DWELL_S'])
    step = rng.integers(1, K, size=n - 1)
    for i in range(1, n):
        state[i] = (state[i - 1] + step[i - 1]) % K if sw[i - 1] else state[i - 1]
    # lateral OU motion toward the current centre
    x = np.empty((n, 3)); x[0] = C[state[0]] + rng.normal(size=3) * sd
    z = rng.normal(size=(n - 1, 3)) * s_in
    for i in range(1, n):
        ci = C[state[i]]
        x[i] = ci + a * (x[i - 1] - ci) + z[i - 1]
    # axial motion: two OU components
    p, zs = c['Z'], c['Z_SCALE']
    x[:, 2] = C[state, 2] + _ou_path(n, p['z_a'] * zs, p['z_tau_a'] / dt, rng) + _ou_path(n, p['z_b'] * zs, p['z_tau_b'] / dt, rng)
    # localization error
    obs = x + rng.normal(size=(n, 3)) * np.asarray(c['NOISE_SD_NM'])
    if return_truth:
        return obs, dict(centres=C, state=state, latent=x)
    return obs


# Analysis (manuscript pipeline)
def preprocess(xyz, c):
    """Centre on the geometric mean (+CENTER_NM); optional crop to the 120 nm cube."""
    a = np.asfortranarray(np.asarray(xyz, dtype=np.float32))
    pts = a - np.nanmean(a, axis=0) + c['CENTER_NM']
    if c['POINT_HANDLING'] == 'crop':
        pts = pts[np.all((pts >= 0) & (pts <= c['GRID_MAX_NM']), axis=1)]
    return np.ascontiguousarray(pts, dtype=np.float32)


def grid_vals(c):
    return np.arange(0.0, c['GRID_MAX_NM'] + c['VOX_NM'] / 2, c['VOX_NM'], dtype=np.float32)


def knn_bandwidth(pts, c):
    if c['BANDWIDTH_FIXED_NM'] is not None:
        return float(c['BANDWIDTH_FIXED_NM'])
    k = c['K_NN']
    d, _ = NearestNeighbors(n_neighbors=k).fit(pts).kneighbors(pts)
    bw = float(np.median(d[:, k - 1]))
    return bw if (np.isfinite(bw) and bw >= 1e-6) else 0.1 * c['VOX_NM']


def kde_binned(pts, vals, bw, c):
    vox = c['VOX_NM']; nv = len(vals)
    if c['POINT_HANDLING'] == 'crop':
        ijk = np.clip(np.rint(pts / vox).astype(int), 0, nv - 1)
        counts = np.zeros((nv, nv, nv), np.float32)
        np.add.at(counts, (ijk[:, 2], ijk[:, 1], ijk[:, 0]), 1.0)
        dens = gaussian_filter(counts, sigma=max(bw / vox, 1e-6), mode='constant', cval=0.0, truncate=4.0)
    else:
        ijk = np.rint(pts / vox).astype(int)
        halo = int(np.ceil(4 * bw / vox)) + 1
        lo = np.minimum(ijk.min(0), 0) - halo
        hi = np.maximum(ijk.max(0), nv - 1) + halo
        counts = np.zeros(tuple((hi - lo + 1)[::-1]), np.float32)
        loc = ijk - lo
        np.add.at(counts, (loc[:, 2], loc[:, 1], loc[:, 0]), 1.0)
        full = gaussian_filter(counts, sigma=max(bw / vox, 1e-6), mode='constant', cval=0.0, truncate=4.0)
        x0, y0, z0 = (-lo).tolist()
        dens = full[z0:z0 + nv, y0:y0 + nv, x0:x0 + nv]
    return (dens / vox ** 3).astype(np.float32)


def proj_xy(dens, zc, c):
    half = c['SLICE_THICKNESS_NM'] / 2; vox = c['VOX_NM']
    lo = int(max(0.0, zc - half) / vox); hi = int(min(c['GRID_MAX_NM'], zc + half) / vox)
    lo = max(0, min(lo, dens.shape[0] - 1)); hi = max(0, min(hi, dens.shape[0] - 1))
    if lo > hi:
        return np.zeros(dens.shape[1:], dens.dtype)
    return dens[lo:hi + 1].mean(axis=0)


def _local_axis(center, window, step, c):
    return np.arange(max(0.0, center - window), min(c['GRID_MAX_NM'], center + window) + step / 2, step, dtype=np.float32)


def refine_peak(pts, bw, x0, y0, zc, c):
    h = max(bw, 1e-6)
    xf = _local_axis(x0, c['SUBVOXEL_WINDOW_XY_NM'], c['SUBVOXEL_STEP_XY_NM'], c)
    yf = _local_axis(y0, c['SUBVOXEL_WINDOW_XY_NM'], c['SUBVOXEL_STEP_XY_NM'], c)
    half = c['SLICE_THICKNESS_NM'] / 2
    zlo, zhi = max(0.0, zc - half), min(c['GRID_MAX_NM'], zc + half)
    zv = np.arange(zlo, zhi + c['SUBVOXEL_STEP_Z_NM'] / 2, c['SUBVOXEL_STEP_Z_NM'], dtype=np.float32) if zhi >= zlo else np.array([zc], np.float32)
    if zv.size == 0:
        zv = np.array([zc], np.float32)
    wx = np.exp(-0.5 * ((xf[:, None] - pts[None, :, 0]) / h) ** 2).astype(np.float32)
    wy = np.exp(-0.5 * ((yf[:, None] - pts[None, :, 1]) / h) ** 2).astype(np.float32)
    wz = np.exp(-0.5 * ((zv[:, None] - pts[None, :, 2]) / h) ** 2).astype(np.float32).mean(axis=0)
    d = (wy * wz[None, :]) @ wx.T
    iy, ix = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(xf[ix]), float(yf[iy])


def slice_peaks(norm, vals, pts, bw, c):
    zi = np.where((vals >= c['ZOOM_MIN_NM']) & (vals <= c['ZOOM_MAX_NM']))[0]
    s0, s1 = int(zi[0]), int(zi[-1]); zv = vals[s0:s1 + 1]
    rows = []
    for zc in np.arange(0.0, c['GRID_MAX_NM'] + 1e-9, c['SLICE_SPACING_NM']):
        sl = proj_xy(norm, float(zc), c)[s0:s1 + 1, s0:s1 + 1]
        if sl.size == 0:
            continue
        thr = max(float(sl.max()) * c['SLICE_RELATIVE_THRESHOLD'], c['GLOBAL_NOISE_CUTOFF'])
        for r, cc in peak_local_max(sl, min_distance=c['MIN_PEAK_DISTANCE_VOX'], threshold_abs=thr, exclude_border=False):
            x, y = float(zv[cc]), float(zv[r])
            xr, yr = refine_peak(pts, bw, x, y, float(zc), c) if c['SUBVOXEL_REFINEMENT'] else (x, y)
            rows.append(dict(z_nm=float(zc), x_nm=x, y_nm=y, x_refined_nm=xr, y_refined_nm=yr, density=float(sl[r, cc])))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['z_nm', 'density'], ascending=[True, False]).reset_index(drop=True)
    return df


def circle_fit(x, y, w):
    w = np.clip(np.asarray(w, float), 1e-12, None); sw = np.sqrt(w)
    A = np.c_[x, y, np.ones_like(x)] * sw[:, None]; b = -(x ** 2 + y ** 2) * sw
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(-D / 2), float(-E / 2), float(math.sqrt(max((D * D + E * E) / 4 - F, 0.0)))


def _bilinear(m, vals, x, y):
    step = float(vals[1] - vals[0]); fx = (np.asarray(x, float) - vals[0]) / step; fy = (np.asarray(y, float) - vals[0]) / step
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    out = np.full(fx.shape, np.nan)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < m.shape[1]) & (y0 + 1 < m.shape[0])
    dx, dy = fx[ok] - x0[ok], fy[ok] - y0[ok]; X0, Y0 = x0[ok], y0[ok]
    out[ok] = (m[Y0, X0] * (1 - dx) * (1 - dy) + m[Y0, X0 + 1] * dx * (1 - dy)
               + m[Y0 + 1, X0] * (1 - dx) * dy + m[Y0 + 1, X0 + 1] * dx * dy)
    return out


def circle_density_ratio(norm, vals, cx, cy, r, z_list, c):
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r) and r > 0):
        return np.nan
    X2, Y2 = np.meshgrid(vals, vals, indexing='xy')
    rr = np.hypot(X2 - cx, Y2 - cy); hw = c['ANNULUS_BAND_HALF_WIDTH_NM']
    band = (rr >= max(0.0, r - hw)) & (rr <= r + hw)
    centre, ring = [], []
    for zc in sorted(set(float(z) for z in z_list)):
        p = proj_xy(norm, zc, c)
        v = _bilinear(p, vals, np.array([cx]), np.array([cy]))[0]
        if np.isfinite(v):
            centre.append(v)
        if band.any():
            b = p[band]; ring.extend(b[np.isfinite(b)].tolist())
    if not centre or not ring:
        return np.nan
    cm, bm = float(np.mean(centre)), float(np.mean(ring))
    return bm / cm if cm > 0 else np.nan


def analyze_trace(pts, c):
    """3D kNN-KDE -> XY-slice maxima -> weighted circle fit. Coordinates in the analysis frame (0–120 nm)."""
    vals = grid_vals(c)
    bw = knn_bandwidth(pts, c)
    dens = kde_binned(pts, vals, bw, c)
    gmax = float(dens.max()); norm = dens / gmax if gmax > 0 else dens
    peaks = slice_peaks(norm, vals, pts, bw, c)
    sd = pts.astype(float).std(axis=0, ddof=1)
    res = dict(n_analyzed=len(pts), bandwidth_nm=bw, n_maxima=len(peaks), radius_nm=np.nan, centre_x_nm=np.nan,
               centre_y_nm=np.nan, annulus_center_ratio=np.nan, sd_x_nm=sd[0], sd_y_nm=sd[1], sd_z_nm=sd[2],
               sd_lateral_nm=float(np.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2)))
    if len(peaks) >= c['MIN_PEAKS_FOR_FIT']:
        cx, cy, r = circle_fit(peaks.x_refined_nm.to_numpy(), peaks.y_refined_nm.to_numpy(), peaks.density.to_numpy())
        ratio = circle_density_ratio(norm, vals, cx, cy, r, peaks.z_nm.to_numpy(), c)
        res.update(radius_nm=r, centre_x_nm=cx - c['CENTER_NM'], centre_y_nm=cy - c['CENTER_NM'], annulus_center_ratio=ratio)
    return res, peaks
