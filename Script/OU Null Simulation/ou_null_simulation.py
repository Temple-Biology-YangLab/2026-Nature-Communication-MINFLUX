#!/usr/bin/env python3
# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""OU null simulation.

Structureless confined (Ornstein-Uhlenbeck) MINFLUX traces with
construct-specific localization noise are analysed with the manuscript pipeline
and compared with the experimental radius distributions (Mann-Whitney U,
two-sample KS, bootstrap CI of the median difference). Edit SETTINGS and run.
Coordinates in nm, times in s.
"""
import os
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import beta, mannwhitneyu, ks_2samp
from sklearn.neighbors import NearestNeighbors, KernelDensity
from skimage.feature import peak_local_max

HERE = os.path.dirname(os.path.abspath(__file__))

#  SETTINGS
RUN_NAME        = "OU_null"            # output folder: outputs/<RUN_NAME>_<timestamp>
N_REPEATS       = 1000                 # simulated traces per construct
N_LOCALIZATIONS = 2000                 # localizations generated per trace
RANDOM_SEED     = 20261016
N_WORKERS       = 8

# Constructs simulated in one run
CONSTRUCTS = ['GFP-Nup58-HaloTag', 'HaloTag-Nup54-GFP', 'HaloTag-Nup58-GFP', 'HaloTag-Nup62-GFP']
DISPLAY_NAMES = {'GFP-Nup58-HaloTag': 'Nup58 C terminus', 'HaloTag-Nup54-GFP': 'Nup54 N terminus',
                 'HaloTag-Nup58-GFP': 'Nup58 N terminus', 'HaloTag-Nup62-GFP': 'Nup62 N terminus'}

# Spread of the molecule's motion (latent, per axis, before noise)
MOTION_SD_NM    = (9.1, 9.1, 20)           # x, y, z — applied to every construct
MOTION_SD_PER_CONSTRUCT = {}           # optional override, e.g. {'HaloTag-Nup62-GFP': (6, 6, 12)}

CONSTRUCT_PARAMS = {
    'GFP-Nup58-HaloTag': dict(NOISE_SD_NM=(9.47, 9.06, 8.79), DT_S=0.000359, MOTION_TAU_S=(0.009998, 0.011225, 0.002722)),
    'HaloTag-Nup54-GFP': dict(NOISE_SD_NM=(9.58, 9.14, 8.72), DT_S=0.000365, MOTION_TAU_S=(0.009615, 0.010367, 0.002792)),
    'HaloTag-Nup58-GFP': dict(NOISE_SD_NM=(9.35, 9.02, 8.42), DT_S=0.000371, MOTION_TAU_S=(0.015140, 0.014963, 0.004048)),
    'HaloTag-Nup62-GFP': dict(NOISE_SD_NM=(9.09, 8.77, 8.30), DT_S=0.000360, MOTION_TAU_S=(0.013633, 0.014758, 0.004117)),
}

POINT_HANDLING  = 'keep_all'

# KDE
K_NN            = 3
BANDWIDTH_FIXED_NM = None    # set a number (nm) to override the kNN bandwidth
KDE_METHOD      = 'binned'   # 'binned' or 'exact'
VOX_NM          = 1.0
GRID_MAX_NM     = 120.0
CENTER_NM       = 60.0

# Slice peaks and refinement
SLICE_SPACING_NM      = 10.0
SLICE_THICKNESS_NM    = 1.0
ZOOM_MIN_NM, ZOOM_MAX_NM = 30.0, 90.0
GLOBAL_NOISE_CUTOFF   = 0.20
SLICE_RELATIVE_THRESHOLD = 0.50
MIN_PEAK_DISTANCE_VOX = 2
SUBVOXEL_REFINEMENT   = True
SUBVOXEL_WINDOW_XY_NM = 2.0
SUBVOXEL_STEP_XY_NM   = 0.1
SUBVOXEL_STEP_Z_NM    = 0.25
MIN_PEAKS_FOR_FIT     = 3

# Fitted-circle density ratio
ANNULUS_BAND_HALF_WIDTH_NM = 1.0
DENSITY_Z_MODE        = 'peak_slices'
EVENT_RADIUS_NM       = (6.0, 14.0)
EVENT_RATIO           = 1.0
EVENT_RATIO_STRICT    = 1.2

# Statistics
N_BOOTSTRAP     = 5000       # bootstrap resamples for the CI of the median difference
ALPHA           = 0.05

SWEEP           = {}

# Experimental comparison
ANALYZE_EXPERIMENT       = True
EXPERIMENT_POINT_HANDLING = 'crop'   # experimental traces were cropped to the 120 nm cube
EXPERIMENT_TRACES_NPZ    = os.path.join(HERE, 'experimental_traces.npz')
EXPERIMENT_REFERENCE_CSV = os.path.join(HERE, 'experimental_reference_radii.csv')

# Outputs
OUTPUT_ROOT       = os.path.join(HERE, 'outputs')
PLOT_FORMATS      = ('png',)        # add 'pdf' / 'svg' if needed
RADIUS_XLIM_NM    = (0, 25)
SHOW_STATS_ON_PLOT = True           # medians, Mann–Whitney and KS p on each panel
SAVE_COORDINATES  = False
SAVE_PEAKS        = False


#  Configuration
def build_config(construct, overrides=None):
    g = globals()
    keys = ['RUN_NAME', 'N_REPEATS', 'N_LOCALIZATIONS', 'RANDOM_SEED', 'MOTION_SD_NM', 'POINT_HANDLING',
            'K_NN', 'BANDWIDTH_FIXED_NM', 'KDE_METHOD', 'VOX_NM', 'GRID_MAX_NM', 'CENTER_NM',
            'SLICE_SPACING_NM', 'SLICE_THICKNESS_NM', 'ZOOM_MIN_NM', 'ZOOM_MAX_NM', 'GLOBAL_NOISE_CUTOFF',
            'SLICE_RELATIVE_THRESHOLD', 'MIN_PEAK_DISTANCE_VOX', 'SUBVOXEL_REFINEMENT', 'SUBVOXEL_WINDOW_XY_NM',
            'SUBVOXEL_STEP_XY_NM', 'SUBVOXEL_STEP_Z_NM', 'MIN_PEAKS_FOR_FIT', 'ANNULUS_BAND_HALF_WIDTH_NM',
            'DENSITY_Z_MODE', 'EVENT_RADIUS_NM', 'EVENT_RATIO', 'EVENT_RATIO_STRICT', 'SAVE_PEAKS']
    c = {k: g[k] for k in keys}
    c.update(CONSTRUCT_PARAMS[construct]); c['CONSTRUCT'] = construct
    if construct in MOTION_SD_PER_CONSTRUCT:
        c['MOTION_SD_NM'] = MOTION_SD_PER_CONSTRUCT[construct]
    if overrides:
        c.update(overrides)
    for k in ('MOTION_SD_NM', 'MOTION_TAU_S', 'NOISE_SD_NM'):
        c[k] = tuple(float(v) for v in c[k])
    c['N_LOCALIZATIONS'] = int(c['N_LOCALIZATIONS']); c['N_REPEATS'] = int(c['N_REPEATS']); c['K_NN'] = int(c['K_NN'])
    c['CONSTRUCT_INDEX'] = CONSTRUCTS.index(construct) if construct in CONSTRUCTS else 0
    return c


#  Simulation
def simulate_trace(c, rng):
    """Stationary 3D OU trajectory (exact discretization) + independent Gaussian noise."""
    n, dt = c['N_LOCALIZATIONS'], c['DT_S']
    sd = np.asarray(c['MOTION_SD_NM']); tau = np.asarray(c['MOTION_TAU_S'])
    a = np.exp(-dt / tau); s_in = sd * np.sqrt(1 - a ** 2)
    x = np.empty((n, 3)); x[0] = rng.normal(size=3) * sd
    z = rng.normal(size=(n - 1, 3)) * s_in
    for i in range(1, n):
        x[i] = a * x[i - 1] + z[i - 1]
    return x + rng.normal(size=(n, 3)) * np.asarray(c['NOISE_SD_NM'])


def preprocess(xyz, c):
    """Pre-centre on the geometric mean (+CENTER_NM), float32 as in the archived pipeline."""
    a = np.asfortranarray(np.asarray(xyz, dtype=np.float32))
    pts = a - np.nanmean(a, axis=0) + c['CENTER_NM']
    if c['POINT_HANDLING'] == 'crop':
        pts = pts[np.all((pts >= 0) & (pts <= c['GRID_MAX_NM']), axis=1)]
    elif c['POINT_HANDLING'] != 'keep_all':
        raise ValueError("POINT_HANDLING must be 'keep_all' or 'crop'")
    return np.ascontiguousarray(pts, dtype=np.float32)


#  Analysis (manuscript pipeline)
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


def kde_exact(pts, vals, bw, c):
    Z, Y, X = np.meshgrid(vals, vals, vals, indexing='ij')
    grid = np.c_[X.ravel(), Y.ravel(), Z.ravel()].astype(np.float32)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw, algorithm='kd_tree', breadth_first=True, rtol=1e-4).fit(pts)
    out = np.empty(len(grid), np.float32)
    for s in range(0, len(grid), 250_000):
        out[s:s + 250_000] = np.exp(kde.score_samples(grid[s:s + 250_000]))
    nv = len(vals)
    return (out.reshape(nv, nv, nv) * len(pts) / c['VOX_NM'] ** 3).astype(np.float32)


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
    vals = grid_vals(c)
    bw = knn_bandwidth(pts, c)
    dens = kde_binned(pts, vals, bw, c) if c['KDE_METHOD'] == 'binned' else kde_exact(pts, vals, bw, c)
    gmax = float(dens.max()); norm = dens / gmax if gmax > 0 else dens
    peaks = slice_peaks(norm, vals, pts, bw, c)
    sd = pts.astype(float).std(axis=0, ddof=1)
    res = dict(n_analyzed=len(pts), bandwidth_nm=bw, n_peaks=len(peaks), fit_success=False,
               radius_nm=np.nan, centre_x_nm=np.nan, centre_y_nm=np.nan, annulus_center_ratio=np.nan,
               sd_x_nm=sd[0], sd_y_nm=sd[1], sd_z_nm=sd[2], sd_lateral_nm=float(np.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2)))
    v = peaks.dropna(subset=['x_refined_nm', 'y_refined_nm']) if len(peaks) else peaks
    if len(v) >= c['MIN_PEAKS_FOR_FIT']:
        cx, cy, r = circle_fit(v.x_refined_nm.to_numpy(), v.y_refined_nm.to_numpy(), v.density.to_numpy())
        zl = v.z_nm.to_numpy() if c['DENSITY_Z_MODE'] == 'peak_slices' else np.arange(0.0, c['GRID_MAX_NM'] + 1e-9, c['SLICE_SPACING_NM'])
        ratio = circle_density_ratio(norm, vals, cx, cy, r, zl, c)
        res.update(radius_nm=r, centre_x_nm=cx - c['CENTER_NM'], centre_y_nm=cy - c['CENTER_NM'],
                   annulus_center_ratio=ratio, fit_success=bool(np.isfinite(r) and np.isfinite(ratio)))
    lo, hi = c['EVENT_RADIUS_NM']
    inwin = res['fit_success'] and lo <= res['radius_nm'] <= hi
    res['annulus_event'] = bool(inwin and res['annulus_center_ratio'] > c['EVENT_RATIO'])
    res['annulus_event_strict'] = bool(inwin and res['annulus_center_ratio'] > c['EVENT_RATIO_STRICT'])
    return res, peaks


#  Simulation driver
def _one(args):
    c, rep = args
    rng = np.random.default_rng(np.random.SeedSequence([c['RANDOM_SEED'], c['CONSTRUCT_INDEX'], rep]))
    xyz = simulate_trace(c, rng)
    res, peaks = analyze_trace(preprocess(xyz, c), c)
    res.update(replicate=rep, n_generated=len(xyz))
    if c['SAVE_PEAKS'] and len(peaks):
        peaks.insert(0, 'replicate', rep)
    return res, (peaks if c['SAVE_PEAKS'] else None), (xyz if SAVE_COORDINATES else None)


def simulate_construct(c, label, out_dir):
    jobs = [(c, r) for r in range(c['N_REPEATS'])]
    if N_WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            out = list(ex.map(_one, jobs, chunksize=4))
    else:
        out = []
        for j in jobs:
            out.append(_one(j))
            if (j[1] + 1) % 100 == 0:
                print(f'    {j[1] + 1}/{c["N_REPEATS"]}', flush=True)
    df = pd.DataFrame([o[0] for o in out])
    df.insert(0, 'construct', c['CONSTRUCT']); df.insert(0, 'condition', label)
    tag = f'{label}_{c["CONSTRUCT"]}'
    df.to_csv(os.path.join(out_dir, f'{tag}_null_per_trace.csv'), index=False)
    if c['SAVE_PEAKS']:
        pk = [o[1] for o in out if o[1] is not None and len(o[1])]
        if pk:
            pd.concat(pk).to_csv(os.path.join(out_dir, f'{tag}_null_peaks.csv'), index=False)
    if SAVE_COORDINATES:
        np.savez_compressed(os.path.join(out_dir, f'{tag}_null_coordinates.npz'), **{f'rep{r:05d}': o[2] for r, o in enumerate(out)})
    return df


def experiment_construct(c, label, out_dir):
    ref = pd.read_csv(EXPERIMENT_REFERENCE_CSV)
    ref = ref[ref.construct == c['CONSTRUCT']]
    if not ANALYZE_EXPERIMENT:
        return ref
    ce = dict(c, POINT_HANDLING=EXPERIMENT_POINT_HANDLING)
    Z = np.load(EXPERIMENT_TRACES_NPZ); rows = []
    for r in ref.itertuples():
        res, _ = analyze_trace(preprocess(Z[r.trace_id], ce), ce)
        res.update(trace_id=r.trace_id, construct=r.construct, source=f'{r.source_file}/{r.source_sheet}')
        rows.append(res)
    e = pd.DataFrame(rows); e.insert(0, 'condition', label)
    e.to_csv(os.path.join(out_dir, f'{label}_{c["CONSTRUCT"]}_experiment_per_trace.csv'), index=False)
    return e


#  Statistics
def binom_ci(k, n):
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def compare(null, exp, construct, label, c, rng):
    """Null vs experiment radius distributions."""
    a = null.loc[null.fit_success, 'radius_nm'].to_numpy(float)
    b = exp.loc[exp.fit_success.astype(bool), 'radius_nm'].to_numpy(float)
    mw2 = mannwhitneyu(b, a, alternative='two-sided')
    mwg = mannwhitneyu(b, a, alternative='greater')
    mwl = mannwhitneyu(b, a, alternative='less')
    ks = ks_2samp(b, a)
    boot = np.array([np.median(rng.choice(b, b.size)) - np.median(rng.choice(a, a.size)) for _ in range(N_BOOTSTRAP)])
    k = int(null.annulus_event.sum()); lo, hi = binom_ci(k, len(null))
    q = lambda s, p: float(np.quantile(s, p)) if len(s) else np.nan
    return dict(condition=label, construct=construct, display_name=DISPLAY_NAMES.get(construct, construct),
                motion_sd_x_nm=c['MOTION_SD_NM'][0], motion_sd_y_nm=c['MOTION_SD_NM'][1], motion_sd_z_nm=c['MOTION_SD_NM'][2],
                n_localizations=c['N_LOCALIZATIONS'], k_nn=c['K_NN'],
                n_null=len(null), n_null_fit=a.size, n_experiment=len(exp), n_experiment_fit=b.size,
                null_radius_median_nm=q(a, .5), null_radius_p025_nm=q(a, .025), null_radius_p975_nm=q(a, .975),
                exp_radius_median_nm=q(b, .5), exp_radius_p025_nm=q(b, .025), exp_radius_p975_nm=q(b, .975),
                median_difference_exp_minus_null_nm=float(np.median(b) - np.median(a)),
                median_difference_ci95_lo=q(boot, .025), median_difference_ci95_hi=q(boot, .975),
                prob_exp_greater_than_null=float(mwg.statistic / (a.size * b.size)),   # probability of superiority
                mannwhitney_p_two_sided=float(mw2.pvalue), mannwhitney_p_exp_greater=float(mwg.pvalue),
                mannwhitney_p_exp_less=float(mwl.pvalue),
                ks_statistic=float(ks.statistic), ks_p=float(ks.pvalue),
                significant_two_sided=bool(mw2.pvalue < ALPHA and ks.pvalue < ALPHA),
                null_ratio_median=float(null.annulus_center_ratio.median()),
                exp_ratio_median=float(exp.annulus_center_ratio.median()),
                null_bandwidth_median_nm=float(null.bandwidth_nm.median()),
                null_annulus_events=k, null_annulus_event_ci95=f'{lo:.4f}-{hi:.4f}')


def fmt_p(p):
    return 'p < 1e-300' if p == 0 else (f'p = {p:.2g}' if p >= 1e-3 else f'p = {p:.1e}')


#  Plots (radius CDF only)
def draw_panel(ax, null, exp, st):
    a = np.sort(null.loc[null.fit_success, 'radius_nm'].to_numpy(float))
    b = np.sort(exp.loc[exp.fit_success.astype(bool), 'radius_nm'].to_numpy(float))
    ax.step(a, np.arange(1, a.size + 1) / a.size, where='post', c='#3d4a57', lw=1.6, label=f'OU null (n = {a.size})')
    ax.step(b, np.arange(1, b.size + 1) / b.size, where='post', c='#c0392b', lw=1.9, label=f'Experiment (n = {b.size})')
    ax.set_xlim(*RADIUS_XLIM_NM); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Fitted circle radius (nm)'); ax.set_ylabel('Cumulative fraction')
    ax.set_title(st['display_name'], fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    lines = []
    if SHOW_STATS_ON_PLOT:
        lines += [f'Median: null {st["null_radius_median_nm"]:.2f}, exp {st["exp_radius_median_nm"]:.2f} nm',
                  f'Mann–Whitney {fmt_p(st["mannwhitney_p_two_sided"])}',
                  f'KS D = {st["ks_statistic"]:.2f}, {fmt_p(st["ks_p"])}']
    if lines:
        ax.text(0.98, 0.04, '\n'.join(lines), transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color='0.25')


def make_plots(results, label, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(results); ncol = 2 if n > 1 else 1; nrow = int(math.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow), squeeze=False)
    for ax, (null, exp, st) in zip(axs.ravel(), results):
        draw_panel(ax, null, exp, st)
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout()
    for f in PLOT_FORMATS:
        fig.savefig(os.path.join(out_dir, f'{label}_radius_cdf_all.{f}'), dpi=300)
    plt.close(fig)
    for null, exp, st in results:
        fig, ax = plt.subplots(figsize=(5.2, 4.2)); draw_panel(ax, null, exp, st); fig.tight_layout()
        for f in PLOT_FORMATS:
            fig.savefig(os.path.join(out_dir, f'{label}_radius_cdf_{st["construct"]}.{f}'), dpi=300)
        plt.close(fig)


#  Main
def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUTPUT_ROOT, f'{RUN_NAME}_{stamp}'); os.makedirs(out_dir, exist_ok=True)
    if SWEEP:
        if len(SWEEP) != 1:
            raise ValueError('SWEEP takes one parameter at a time')
        (key, values), = SWEEP.items()
        variants = [(''.join(ch if ch.isalnum() or ch in '=._-' else '_' for ch in f'{key}={v}'), {key: v}) for v in values]
    else:
        variants = [('base', None)]
    rng = np.random.default_rng(RANDOM_SEED + 1)
    settings, stats_rows = {}, []
    for label, ov in variants:
        results = []
        for construct in CONSTRUCTS:
            c = build_config(construct, ov); settings[f'{label}/{construct}'] = c
            t0 = time.perf_counter()
            print(f'[{label}] {construct}: N={c["N_LOCALIZATIONS"]}, repeats={c["N_REPEATS"]}, k={c["K_NN"]}, '
                  f'motion SD={list(c["MOTION_SD_NM"])} nm, noise={list(c["NOISE_SD_NM"])} nm', flush=True)
            null = simulate_construct(c, label, out_dir)
            exp = experiment_construct(c, label, out_dir)
            st = compare(null, exp, construct, label, c, rng); st['seconds'] = round(time.perf_counter() - t0, 1)
            stats_rows.append(st); results.append((null, exp, st))
        make_plots(results, label, out_dir)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(os.path.join(out_dir, 'statistics.csv'), index=False)
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump(dict(settings=settings, sweep=SWEEP, analyze_experiment=ANALYZE_EXPERIMENT,
                       experiment_point_handling=EXPERIMENT_POINT_HANDLING, n_bootstrap=N_BOOTSTRAP,
                       created=stamp), f, indent=2, default=str)
    show = ['condition', 'display_name', 'n_null_fit', 'n_experiment_fit', 'null_radius_median_nm', 'exp_radius_median_nm',
            'median_difference_exp_minus_null_nm', 'prob_exp_greater_than_null', 'mannwhitney_p_two_sided', 'ks_p']
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.4g}'.format):
        print(stats[show].to_string(index=False))
    print(f'\nOutputs: {out_dir}')
    return out_dir


if __name__ == '__main__':
    main()
