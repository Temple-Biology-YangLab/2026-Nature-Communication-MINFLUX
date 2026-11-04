# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""KDE bandwidth sensitivity of the MINFLUX annular circle fit.

Sweep A varies the kNN order k of the bandwidth estimator, sweep B rescales the
k = 3 bandwidth, and sweep C samples each trace's window between its
nearest-neighbour floor and its critical (mode-merging) bandwidth. All other
steps follow the manuscript pipeline.
"""
from __future__ import annotations

import os
# single-threaded numeric libraries per worker; must precede the numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# USER CONFIG
N_WORKERS     = 8            # parallel worker processes
PARALLEL_MODE = "process"    # "process" | "thread" | "serial"
LIMIT         = 0            # 0 = all traces

# Sweep A: kNN order k of the bandwidth estimator
K_LIST = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
# Sweep B: multiplicative rescaling of the k = 3 bandwidth
M_LIST = [0.50, 0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
# Sweep C: position u within each trace's [floor, ceiling] bandwidth window
U_LIST = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]

BASELINE_K = 3               # manuscript K_BAND; anchors both sweeps

COMPUTE_LSCV    = True       # least-squares cross-validation bandwidth per trace
COMPUTE_PEAK_DIAGNOSTICS = True   # radial distribution of detected peaks (2x KDE cost)
LSCV_MAX_POINTS = 1500       # subsample cap for the O(n^2) LSCV distance matrix

FLOOR_K = 2                  # floor = median nearest-neighbour distance (column 0 is self)

# Silverman-style critical bandwidth (upper bound on smoothing)
COMPUTE_CRITICAL_BW = True
CRIT_BW_GRID_NM = None            # None -> np.geomspace(1.5, 15.0, 18)
CRIT_PROFILE_R_MAX_NM = 25.0      # radial range of the azimuthal profile
CRIT_PROFILE_DR_NM    = 1.0       # radial bin width
CRIT_MIN_PEAK_RADIUS_NM = 3.0     # profile peak must sit at least this far out
CRIT_CENTRAL_DIP_FRAC = 0.90      # annular iff g(0) < this fraction of the peak
INPUT_ROOT  = "Extracted Trace"
OUTPUT_ROOT = "bandwidth_sweep_output"
KEEP_MIN_LOCALIZATIONS = 800     # matches the manuscript / bootstrap filter
X_COL, Y_COL, Z_COL = "X (nm)", "Y (nm)", "Z (nm)"

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_BULK_DIR = os.path.abspath(os.path.join(_HERE, "..", "Bootstrap Stability of Radius"))
if _BULK_DIR not in sys.path:
    sys.path.insert(0, _BULK_DIR)

import bulk_bootstrap_all_traces as bulk   # noqa: E402
import minflux_null_simulation_v5_fitted_circle_density as core  # noqa: E402

core.KDE_EVALUATION_METHOD = "binned_gaussian"
core.ENABLE_SUBVOXEL_2D_REFINEMENT = True

INPUT_ROOT = os.path.join(_HERE, INPUT_ROOT)
OUTPUT_ROOT = os.path.join(_HERE, OUTPUT_ROOT)

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:                                    # pragma: no cover
    _HAVE_TQDM = False

    class tqdm:                                        # noqa: N801
        """Fallback: prints a percentage line every PRINT_EVERY updates."""
        PRINT_EVERY = 10

        def __init__(self, total=None, desc="", unit="it", **_):
            self.total, self.desc, self.unit = total, desc, unit
            self.n, self.t0, self.postfix = 0, time.time(), ""
            print(f"{desc} (install tqdm for a live bar)", flush=True)

        def update(self, n=1):
            self.n += n
            if self.n % self.PRINT_EVERY and self.n != self.total:
                return
            el = time.time() - self.t0
            rate = el / max(self.n, 1)
            eta = rate * max((self.total or 0) - self.n, 0)
            print(f"  {self.n}/{self.total} {self.unit} "
                  f"[{el/60:.1f} min elapsed, ~{eta/60:.1f} min left, "
                  f"{rate:.1f} s/{self.unit}] {self.postfix}", flush=True)

        def set_postfix_str(self, s):
            self.postfix = s

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()


def normal_reference_bandwidths(pts: np.ndarray) -> Dict[str, float]:
    """Silverman and Scott normal-reference bandwidths for a 3D point cloud."""
    n, d = len(pts), 3
    sigma = float(np.mean(np.std(pts, axis=0, ddof=1)))
    return dict(
        bw_silverman_nm=sigma * (4.0 / (n * (d + 2))) ** (1.0 / (d + 4)),
        bw_scott_nm=sigma * n ** (-1.0 / (d + 4)),
    )


def lscv_bandwidth(pts: np.ndarray, h_grid: np.ndarray | None = None,
                   max_points: int = LSCV_MAX_POINTS, seed: int = 0) -> Dict[str, float]:
    """Least-squares cross-validation bandwidth for an isotropic 3D Gaussian KDE."""
    rng = np.random.default_rng(seed)
    p = pts if len(pts) <= max_points else pts[rng.choice(len(pts), max_points, replace=False)]
    n = len(p)
    nan = dict(bw_lscv_nm=np.nan, lscv_interior_min=False)
    if n < 50:
        return nan
    if h_grid is None:
        h_grid = np.geomspace(0.3, 15.0, 60)

    diff = p[:, None, :] - p[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    del diff

    scores = np.empty(len(h_grid))
    for i, h in enumerate(h_grid):
        h2 = h * h
        # integral of f-hat squared (includes the i == j diagonal)
        t1 = ((4.0 * np.pi * h2) ** -1.5) * float(np.exp(-d2 / (4.0 * h2)).sum()) / (n * n)
        # leave-one-out term; subtract n to drop the diagonal (exp(0) = 1)
        s2 = float(np.exp(-d2 / (2.0 * h2)).sum()) - n
        t2 = 2.0 * ((2.0 * np.pi * h2) ** -1.5) * s2 / (n * (n - 1))
        scores[i] = t1 - t2

    j = int(np.argmin(scores))
    return dict(bw_lscv_nm=float(h_grid[j]),
                lscv_interior_min=bool(0 < j < len(h_grid) - 1))


def _azimuthal_profile(density_norm_3d: np.ndarray, vals: np.ndarray,
                       z_center: float, cx: float = None, cy: float = None) -> tuple:
    """Azimuthally averaged areal density g(r) about (cx, cy)."""
    if cx is None:
        cx = core.CENTER_NM
    if cy is None:
        cy = core.CENTER_NM
    proj = core.proj_xy(density_norm_3d, z_center, core.NEW_2D_PROJ_THICKNESS, core.VOX_NM)
    zi = np.where((vals >= core.ZOOM_MIN_NM) & (vals <= core.ZOOM_MAX_NM))[0]
    if len(zi) == 0:
        return np.array([]), np.array([]), np.array([])
    lo, hi = int(zi[0]), int(zi[-1])
    sub = proj[lo:hi + 1, lo:hi + 1]
    zv = vals[lo:hi + 1]
    # proj is indexed [row = y, col = x]
    Xg, Yg = np.meshgrid(zv, zv)
    R = np.sqrt((Xg - cx) ** 2 + (Yg - cy) ** 2).ravel()
    v = np.asarray(sub, float).ravel()

    bins = np.arange(0.0, CRIT_PROFILE_R_MAX_NM + CRIT_PROFILE_DR_NM, CRIT_PROFILE_DR_NM)
    nb = len(bins) - 1
    idx = np.digitize(R, bins) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(v)
    counts = np.bincount(idx[ok], minlength=nb).astype(float)
    sums = np.bincount(idx[ok], weights=v[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(counts > 0, sums / counts, np.nan)
    return 0.5 * (bins[:-1] + bins[1:]), g, counts


def _profile_is_annular(r: np.ndarray, g: np.ndarray) -> tuple:
    """Interior maximum beyond CRIT_MIN_PEAK_RADIUS_NM with a depleted centre."""
    if g.size == 0 or not np.any(np.isfinite(g)):
        return False, np.nan, np.nan
    finite = np.isfinite(g)
    outer = finite & (r >= CRIT_MIN_PEAK_RADIUS_NM) & (r <= CRIT_PROFILE_R_MAX_NM - CRIT_PROFILE_DR_NM)
    if not np.any(outer):
        return False, np.nan, np.nan
    j = int(np.flatnonzero(outer)[np.argmax(g[outer])])
    g_peak, r_peak = float(g[j]), float(r[j])
    if 0 < j < len(g) - 1 and np.isfinite(g[j - 1]) and np.isfinite(g[j + 1]):
        denom = g[j - 1] - 2.0 * g[j] + g[j + 1]
        if denom != 0:
            shift = 0.5 * (g[j - 1] - g[j + 1]) / denom
            if abs(shift) <= 1.0:
                r_peak = float(r[j] + shift * (r[1] - r[0]))
    inner = finite & (r < CRIT_MIN_PEAK_RADIUS_NM)
    if not np.any(inner) or not (g_peak > 0):
        return False, r_peak, np.nan
    g_centre = float(np.nanmax(g[inner]))
    dip = g_centre / g_peak
    # interior maximum: must not sit at the outer edge of the tested range
    interior = j < (len(r) - 1)
    return bool(interior and dip < CRIT_CENTRAL_DIP_FRAC), r_peak, dip


def critical_bandwidth(pts: np.ndarray) -> Dict:
    """Smallest bandwidth in the grid at which the annular profile is lost."""
    out = dict(critical_bw_nm=np.nan, critical_bw_monotone=False,
               annular_at_k3=False, crit_r_peak_at_k3_nm=np.nan,
               crit_central_dip_at_k3=np.nan)
    grid = (np.geomspace(1.5, 15.0, 18) if CRIT_BW_GRID_NM is None
            else np.asarray(CRIT_BW_GRID_NM, float))
    bw_k3 = core.calculate_global_kth_nn_bandwidth(pts, BASELINE_K)
    vals, grid_pts = bulk._ensure_grid()

    def annular_at(bw):
        density = core.kde_density_on_grid(pts, grid_pts, vals, bw, core.VOX_NM,
                                           core.KDE_GRID_CHUNK_SIZE)
        gmax = float(np.max(density)) if density.size else np.nan
        dn = density / gmax if (gmax and np.isfinite(gmax) and gmax > 0) else density
        best = (False, np.nan, np.inf)
        for zc in core.NEW_2D_PROJ_CENTRES:
            r, g, _ = _azimuthal_profile(dn, vals, float(zc))
            ok, r_peak, dip = _profile_is_annular(r, g)
            if np.isfinite(dip) and dip < best[2]:
                best = (ok, r_peak, dip)
        return best

    try:
        ok3, rp3, dip3 = annular_at(bw_k3)
        out.update(annular_at_k3=bool(ok3), crit_r_peak_at_k3_nm=rp3,
                   crit_central_dip_at_k3=dip3)
        flags = []
        for h in grid:
            ok, _, _ = annular_at(float(h))
            flags.append(bool(ok))
            # early exit: annularity lost and stays lost for two consecutive steps
            if len(flags) >= 2 and not flags[-1] and not flags[-2]:
                break
        flags = np.asarray(flags, bool)
        lost = np.flatnonzero(~flags)
        if lost.size:
            first = int(lost[0])
            out["critical_bw_nm"] = float(grid[first])
            out["critical_bw_monotone"] = bool(not flags[first:].any())
        else:
            # never lost within the grid -> ceiling is above the tested range
            out["critical_bw_nm"] = float(grid[len(flags) - 1])
            out["critical_bw_monotone"] = True
    except Exception as exc:
        warnings.warn(f"critical bandwidth failed: {exc}")
    return out


def selector_row(pts: np.ndarray) -> Dict:
    """All objective bandwidth estimates for one trace, plus the k = 3 value."""
    bw_k3 = core.calculate_global_kth_nn_bandwidth(pts, BASELINE_K)
    row = dict(bw_knn_k3_nm=float(bw_k3), n_locs=int(len(pts)))
    row.update(normal_reference_bandwidths(pts))
    if COMPUTE_LSCV:
        row.update(lscv_bandwidth(pts))
    else:
        row.update(bw_lscv_nm=np.nan, lscv_interior_min=False)
    bw_floor = core.calculate_global_kth_nn_bandwidth(pts, FLOOR_K)
    row["bw_floor_nm"] = float(bw_floor)
    row["k3_over_floor"] = float(bw_k3 / bw_floor) if bw_floor > 0 else np.nan

    n = len(pts)
    sd = np.std(pts, axis=0, ddof=1)
    vol_ellip = (4.0 / 3.0) * np.pi * float(np.prod(2 * sd))
    row["spacing_ellipsoid_nm"] = float((vol_ellip / n) ** (1.0 / 3.0))
    try:
        from scipy.spatial import ConvexHull
        row["spacing_hull_nm"] = float((ConvexHull(pts).volume / n) ** (1.0 / 3.0))
    except Exception:
        row["spacing_hull_nm"] = np.nan
    for nm in ("ellipsoid", "hull"):
        sp = row[f"spacing_{nm}_nm"]
        row[f"k3_over_spacing_{nm}"] = float(bw_k3 / sp) if (np.isfinite(sp) and sp > 0) else np.nan

    if COMPUTE_CRITICAL_BW:
        crit = critical_bandwidth(pts)
        row.update(crit)
        cb = crit["critical_bw_nm"]
        row["critical_bw_over_k3"] = float(cb / bw_k3) if (bw_k3 > 0 and np.isfinite(cb)) else np.nan
        row["k3_within_bounds"] = bool(
            np.isfinite(cb) and bw_floor <= bw_k3 <= cb)
        row["window_width_nm"] = float(cb - bw_floor) if np.isfinite(cb) else np.nan
        # where k = 3 sits inside the window, on the same 0-1 scale as sweep C
        row["u_k3"] = (float((bw_k3 - bw_floor) / (cb - bw_floor))
                       if (np.isfinite(cb) and cb > bw_floor) else np.nan)

    for name in ("silverman", "scott", "lscv"):
        v = row.get(f"bw_{name}_nm", np.nan)
        row[f"ratio_{name}_to_k3"] = float(v / bw_k3) if (bw_k3 > 0 and np.isfinite(v)) else np.nan
    return row


# per-trace sweep
def sweep_one_trace(pts: np.ndarray, bw_floor: float = np.nan,
                    bw_ceiling: float = np.nan) -> List[Dict]:
    """Run both bandwidth sweeps on a single centred, cube-filtered trace."""
    bulk._ensure_grid()
    rows: List[Dict] = []

    bw_baseline = core.calculate_global_kth_nn_bandwidth(pts, BASELINE_K)
    bounds = dict(bw_floor_nm=float(bw_floor), bw_ceiling_nm=float(bw_ceiling),
                  n_locs_for_ratio=int(len(pts)))

    # Sweep A - vary the kNN order k.
    for k in K_LIST:
        bw = core.calculate_global_kth_nn_bandwidth(pts, k)
        rows.append(_one_point(pts, bw, sweep="k_band", setting=float(k),
                               bw_baseline=bw_baseline, **bounds))

    # Sweep B - rescale the k = 3 bandwidth.
    for m in M_LIST:
        rows.append(_one_point(pts, float(m) * bw_baseline, sweep="multiplier",
                               setting=float(m), bw_baseline=bw_baseline, **bounds))

    # Sweep C - even positions within this trace's own [floor, ceiling] window.
    if (np.isfinite(bw_floor) and np.isfinite(bw_ceiling) and bw_ceiling > bw_floor):
        span = bw_ceiling - bw_floor
        for u in U_LIST:
            rows.append(_one_point(pts, bw_floor + float(u) * span, sweep="window_frac",
                                   setting=float(u), bw_baseline=bw_baseline, **bounds))
    return rows


def _peak_diagnostics(pts: np.ndarray, bw: float,
                      cx: float = None, cy: float = None) -> Dict:
    """Radial distribution of the detected 2D slice peaks at one bandwidth."""
    out = dict(peak_median_radius_nm=np.nan, peak_q25_radius_nm=np.nan,
               peak_q75_radius_nm=np.nan, frac_peaks_in_annulus_window=np.nan,
               n_peaks_in_annulus=0, n_peaks_central=0, ring_to_centre_peak_ratio=np.nan,
               profile_peak_radius_nm=np.nan, profile_central_dip=np.nan,
               profile_annular=False)
    try:
        vals, grid_pts = bulk._ensure_grid()
        density = core.kde_density_on_grid(pts, grid_pts, vals, bw, core.VOX_NM,
                                           core.KDE_GRID_CHUNK_SIZE)
        gmax = float(np.max(density)) if density.size else np.nan
        dn = density / gmax if (gmax and np.isfinite(gmax) and gmax > 0) else density
        peaks = core.find_2d_slice_peaks(dn, vals, pts, bw, int(len(pts)), 0)
    except Exception as exc:
        warnings.warn(f"peak diagnostics failed at bw={bw:.3f}: {exc}")
        return out

    best_r, best_dip, best_ok = np.nan, np.inf, False
    for zc in core.NEW_2D_PROJ_CENTRES:
        rr, gg, _ = _azimuthal_profile(dn, vals, float(zc), cx=cx, cy=cy)
        okp, r_pk, dip = _profile_is_annular(rr, gg)
        if np.isfinite(dip) and dip < best_dip:
            best_r, best_dip, best_ok = r_pk, dip, okp
    out.update(profile_peak_radius_nm=float(best_r) if np.isfinite(best_r) else np.nan,
               profile_central_dip=float(best_dip) if np.isfinite(best_dip) else np.nan,
               profile_annular=bool(best_ok))

    if len(peaks) == 0:
        return out

    r = peaks["Radius_XY_Refined_From_True_Center_nm"].to_numpy(float)
    r = np.where(np.isfinite(r), r, peaks["Radius_XY_From_True_Center_nm"].to_numpy(float))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return out
    in_win = (r >= core.ANNULUS_RADIUS_MIN_NM) & (r <= core.ANNULUS_RADIUS_MAX_NM)
    central = r < core.ANNULUS_RADIUS_MIN_NM
    out.update(
        peak_median_radius_nm=float(np.median(r)),
        peak_q25_radius_nm=float(np.percentile(r, 25)),
        peak_q75_radius_nm=float(np.percentile(r, 75)),
        frac_peaks_in_annulus_window=float(in_win.mean()),
        n_peaks_in_annulus=int(in_win.sum()),
        n_peaks_central=int(central.sum()),
        ring_to_centre_peak_ratio=(float(in_win.sum() / central.sum())
                                   if central.sum() else np.inf),
    )
    return out


def _one_point(pts: np.ndarray, bw: float, sweep: str, setting: float,
               bw_baseline: float, bw_floor_nm: float = np.nan,
               bw_ceiling_nm: float = np.nan, n_locs_for_ratio: int = 0) -> Dict:
    """Evaluate the manuscript pipeline at one bandwidth."""
    try:
        res = bulk.analyze(pts, refine=True, bandwidth=bw)
    except Exception as exc:                      # keep the sweep going
        warnings.warn(f"analyze failed at {sweep}={setting}: {exc}")
        res = dict(refined_radius_nm=np.nan, refined_diameter_nm=np.nan,
                   refined_center_x_nm=np.nan, refined_center_y_nm=np.nan,
                   refined_adj_pseudo_r2=np.nan, coarse_radius_nm=np.nan,
                   fit_success=False, n_peaks=0, kde_bandwidth_nm=float(bw))
    _cx = res.get("refined_center_x_nm", np.nan)
    _cy = res.get("refined_center_y_nm", np.nan)
    if not (np.isfinite(_cx) and np.isfinite(_cy)):
        _cx = _cy = None
    diag = (_peak_diagnostics(pts, bw, cx=_cx, cy=_cy)
            if COMPUTE_PEAK_DIAGNOSTICS else {})
    below = bool(np.isfinite(bw_floor_nm) and bw < bw_floor_nm)
    above = bool(np.isfinite(bw_ceiling_nm) and bw > bw_ceiling_nm)
    npk = res.get("n_peaks", 0)
    return dict(
        **diag,
        bw_floor_nm=float(bw_floor_nm),
        bw_ceiling_nm=float(bw_ceiling_nm),
        peaks_per_loc=(float(npk) / n_locs_for_ratio) if n_locs_for_ratio else np.nan,
        bw_below_sampling_floor=below,
        bw_above_critical=above,
        bw_degenerate=bool(below or above),
        sweep=sweep,
        setting=setting,
        kde_bandwidth_nm=float(bw),
        bw_baseline_k3_nm=float(bw_baseline),
        bw_ratio_to_k3=float(bw / bw_baseline) if bw_baseline > 0 else np.nan,
        radius_nm=res["refined_radius_nm"],
        diameter_nm=res["refined_diameter_nm"],
        center_x_nm=res["refined_center_x_nm"],
        center_y_nm=res["refined_center_y_nm"],
        adj_pseudo_r2=res["refined_adj_pseudo_r2"],
        coarse_radius_nm=res["coarse_radius_nm"],
        n_peaks=res["n_peaks"],
        fit_success=bool(res["fit_success"]),
    )


def worker(args):
    fpath, construct, sheet = args
    try:
        df = pd.read_excel(fpath, sheet_name=sheet)
        if not {X_COL, Y_COL, Z_COL}.issubset(df.columns):
            return None
        pts = bulk.center_and_filter(df[[X_COL, Y_COL, Z_COL]].to_numpy(float))
        if len(pts) < KEEP_MIN_LOCALIZATIONS:
            return None
        ident = dict(construct=construct, source_file=os.path.basename(fpath),
                     sheet=sheet, n_locs=int(len(pts)))
        sel = selector_row(pts)
        sel.update(ident)
        rows = sweep_one_trace(pts, bw_floor=sel.get("bw_floor_nm", np.nan),
                               bw_ceiling=sel.get("critical_bw_nm", np.nan))
        for r in rows:
            r.update(ident)
        return dict(rows=rows, selector=sel)
    except Exception as exc:
        warnings.warn(f"{os.path.basename(fpath)}|{sheet} failed: {exc}")
        return None


def _dispatch(tasks, mode, workers):
    if mode == "serial" or workers <= 1:
        for t in tasks:
            yield worker(t)
    elif mode == "thread":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for f in as_completed(futs):
                yield f.result()
    else:
        from multiprocessing import Pool
        with Pool(processes=workers) as pool:
            for r in pool.imap_unordered(worker, tasks):
                yield r


# paired statistics and aggregation
def add_paired_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Express each bandwidth point relative to the same trace's k = 3 result."""
    key = ["source_file", "sheet"]
    base = (df[(df["sweep"] == "k_band") & (df["setting"] == float(BASELINE_K))]
            .set_index(key)["radius_nm"].rename("radius_k3_nm"))
    df = df.join(base, on=key)
    df["radius_delta_nm"] = df["radius_nm"] - df["radius_k3_nm"]
    df["radius_pct_change"] = 100.0 * df["radius_delta_nm"] / df["radius_k3_nm"]

    if "profile_peak_radius_nm" in df.columns:
        pbase = (df[(df["sweep"] == "k_band") & (df["setting"] == float(BASELINE_K))]
                 .set_index(key)["profile_peak_radius_nm"].rename("profile_radius_k3_nm"))
        df = df.join(pbase, on=key)
        df["profile_radius_pct_change"] = (
            100.0 * (df["profile_peak_radius_nm"] - df["profile_radius_k3_nm"])
            / df["profile_radius_k3_nm"])
    return df


def build_aggregate(df: pd.DataFrame, group_extra: List[str] | None = None) -> pd.DataFrame:
    """Summarize across traces at each bandwidth point."""
    keys = ["sweep", "setting"] + (group_extra or [])
    rows = []
    for gk, sub in df.groupby(keys, dropna=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        ok = sub[sub["fit_success"] & np.isfinite(sub["radius_nm"])]
        d = ok["radius_pct_change"].dropna()
        rows.append({
            **dict(zip(keys, gk)),
            "n_traces_attempted": int(len(sub)),
            "n_traces_fit": int(len(ok)),
            "frac_fit_success": float(len(ok) / len(sub)) if len(sub) else np.nan,
            "median_bandwidth_nm": float(ok["kde_bandwidth_nm"].median()) if len(ok) else np.nan,
            "median_bw_ratio_to_k3": float(ok["bw_ratio_to_k3"].median()) if len(ok) else np.nan,
            "median_radius_nm": float(ok["radius_nm"].median()) if len(ok) else np.nan,
            "q25_radius_nm": float(ok["radius_nm"].quantile(0.25)) if len(ok) else np.nan,
            "q75_radius_nm": float(ok["radius_nm"].quantile(0.75)) if len(ok) else np.nan,
            "median_diameter_nm": float(ok["diameter_nm"].median()) if len(ok) else np.nan,
            # paired change vs the same trace at k = 3
            "median_radius_pct_change": float(d.median()) if len(d) else np.nan,
            "iqr_radius_pct_change": float(d.quantile(0.75) - d.quantile(0.25)) if len(d) else np.nan,
            "median_abs_radius_pct_change": float(d.abs().median()) if len(d) else np.nan,
            "frac_within_10pct_of_k3": float((d.abs() <= 10).mean()) if len(d) else np.nan,
            "median_adj_pseudo_r2": float(ok["adj_pseudo_r2"].median()) if len(ok) else np.nan,
            "median_n_peaks": float(ok["n_peaks"].median()) if len(ok) else np.nan,
            # peak radial diagnostics: do the detected peaks still lie on the ring?
            "median_peak_radius_nm": float(ok["peak_median_radius_nm"].median()) if "peak_median_radius_nm" in ok else np.nan,
            "median_frac_peaks_in_annulus_window": float(ok["frac_peaks_in_annulus_window"].median()) if "frac_peaks_in_annulus_window" in ok else np.nan,
            "median_n_peaks_central": float(ok["n_peaks_central"].median()) if "n_peaks_central" in ok else np.nan,
            "frac_bw_degenerate": float(sub["bw_degenerate"].mean()) if "bw_degenerate" in sub else np.nan,
            "frac_below_sampling_floor": float(sub["bw_below_sampling_floor"].mean()) if "bw_below_sampling_floor" in sub else np.nan,
            "frac_above_critical": float(sub["bw_above_critical"].mean()) if "bw_above_critical" in sub else np.nan,
            "median_peaks_per_loc": float(ok["peaks_per_loc"].median()) if "peaks_per_loc" in ok else np.nan,
            # threshold-free comparison metric
            "median_profile_radius_nm": float(ok["profile_peak_radius_nm"].median()) if "profile_peak_radius_nm" in ok else np.nan,
            "median_profile_radius_pct_change": float(ok["profile_radius_pct_change"].median()) if "profile_radius_pct_change" in ok else np.nan,
            "iqr_profile_radius_pct_change": float(ok["profile_radius_pct_change"].quantile(.75) - ok["profile_radius_pct_change"].quantile(.25)) if "profile_radius_pct_change" in ok else np.nan,
            "frac_profile_annular": float(sub["profile_annular"].mean()) if "profile_annular" in sub else np.nan,
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def main():
    out_root = Path(OUTPUT_ROOT)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)

    bulk._ensure_grid()
    traces = bulk.enumerate_traces(INPUT_ROOT)
    if LIMIT:
        traces = traces[:LIMIT]
    n_bw = len(K_LIST) + len(M_LIST)
    print(f"Enumerated {len(traces)} candidate sheets; {n_bw} bandwidth points each "
          f"({len(K_LIST)} k_band + {len(M_LIST)} multiplier).", flush=True)

    all_rows, sel_rows, kept = [], [], 0
    bar = tqdm(total=len(traces), desc="Bandwidth sweep", unit="trace")
    for res in _dispatch(traces, PARALLEL_MODE, max(1, N_WORKERS)):
        if res:
            all_rows.extend(res["rows"])
            sel_rows.append(res["selector"])
            kept += 1
        bar.update(1)
        bar.set_postfix_str(f"kept {kept} (>= {KEEP_MIN_LOCALIZATIONS} locs), "
                            f"{len(all_rows)} fits")
    bar.close()

    if not all_rows:
        print("No traces passed the localization filter.")
        return None

    df = pd.DataFrame(all_rows)
    df = add_paired_deviations(df)
    df = df.sort_values(["construct", "source_file", "sheet", "sweep", "setting"]).reset_index(drop=True)
    df.to_csv(out_root / "tables" / "bandwidth_sweep_long.csv", index=False)

    agg = build_aggregate(df)
    agg.to_csv(out_root / "tables" / "bandwidth_sweep_aggregate.csv", index=False)
    build_aggregate(df, ["construct"]).to_csv(
        out_root / "tables" / "bandwidth_sweep_by_construct.csv", index=False)

    sel = pd.DataFrame(sel_rows).sort_values(["construct", "source_file", "sheet"]).reset_index(drop=True)
    sel.to_csv(out_root / "tables" / "bandwidth_selectors_per_trace.csv", index=False)
    sel_sum = sel.groupby("construct").agg(
        n_traces=("bw_knn_k3_nm", "size"),
        median_bw_knn_k3_nm=("bw_knn_k3_nm", "median"),
        median_bw_silverman_nm=("bw_silverman_nm", "median"),
        median_bw_scott_nm=("bw_scott_nm", "median"),
        median_bw_lscv_nm=("bw_lscv_nm", "median"),
        median_ratio_silverman_to_k3=("ratio_silverman_to_k3", "median"),
        median_ratio_lscv_to_k3=("ratio_lscv_to_k3", "median"),
        # measured floor (nearest-neighbour spacing)
        median_bw_floor_nm=("bw_floor_nm", "median"),
        median_k3_over_floor=("k3_over_floor", "median"),
        median_window_width_nm=("window_width_nm", "median"),
        # sampling-interval context
        median_k3_over_spacing_ellipsoid=("k3_over_spacing_ellipsoid", "median"),
        median_k3_over_spacing_hull=("k3_over_spacing_hull", "median"),
        # mode-merging ceiling
        median_critical_bw_nm=("critical_bw_nm", "median"),
        median_critical_bw_over_k3=("critical_bw_over_k3", "median"),
        frac_annular_at_k3=("annular_at_k3", "mean"),
        frac_k3_within_bounds=("k3_within_bounds", "mean"),
        frac_critical_bw_monotone=("critical_bw_monotone", "mean"),
    ).reset_index()
    sel_sum.to_csv(out_root / "tables" / "bandwidth_selectors_summary.csv", index=False)

    with open(out_root / "analysis_parameters.json", "w") as fh:
        json.dump(dict(
            keep_min_localizations=KEEP_MIN_LOCALIZATIONS,
            kde_evaluation_method=core.KDE_EVALUATION_METHOD,
            subvoxel_refinement=bool(core.ENABLE_SUBVOXEL_2D_REFINEMENT),
            measured_metric="circle_fit_refined_radius_nm",
            baseline_k_band=BASELINE_K,
            k_list=K_LIST, multiplier_list=M_LIST,
            vox_nm=core.VOX_NM, kernel=core.KERNEL,
            bootstrap="none (deterministic sweep); CIs reported by Bulk Bootstrapping",
            bandwidth_window="measured per trace; no external precision constant",
            floor_definition=f"median nearest-neighbour distance (FLOOR_K={FLOOR_K})",
            ceiling_definition="Silverman-style critical bandwidth (annular profile merges)",
            objective_selectors=["knn_k3", "silverman", "scott"] + (["lscv"] if COMPUTE_LSCV else []),
            lscv_max_points=LSCV_MAX_POINTS,
            peak_diagnostics=bool(COMPUTE_PEAK_DIAGNOSTICS),
            critical_bandwidth=dict(
                computed=bool(COMPUTE_CRITICAL_BW),
                grid_nm=("geomspace(1.5,15,18)" if CRIT_BW_GRID_NM is None
                         else list(map(float, CRIT_BW_GRID_NM))),
                profile_r_max_nm=CRIT_PROFILE_R_MAX_NM, profile_dr_nm=CRIT_PROFILE_DR_NM,
                min_peak_radius_nm=CRIT_MIN_PEAK_RADIUS_NM,
                central_dip_fraction=CRIT_CENTRAL_DIP_FRAC,
                reference="Silverman 1981 JRSS-B 43:97 (critical bandwidth for mode merging)",
                caveat="monotonicity proved for 1D KDE of raw data; g(r) here is an "
                       "azimuthal average of a projected 3D KDE, so monotonicity is "
                       "expected but not guaranteed (see critical_bw_monotone)"),
            annulus_radius_window_nm=[core.ANNULUS_RADIUS_MIN_NM, core.ANNULUS_RADIUS_MAX_NM],
            n_traces_kept=int(kept), parallel_mode=PARALLEL_MODE, n_workers=N_WORKERS,
        ), fh, indent=2)

    print(f"\nKept {kept} traces -> {out_root.resolve()}")
    print("\nObjective bandwidth selectors (median per construct, nm):")
    print(sel_sum.to_string(index=False))
    print("\nAggregate (all constructs):")
    cols = ["sweep", "setting", "median_bandwidth_nm", "median_radius_nm",
            "median_radius_pct_change", "iqr_radius_pct_change",
            "median_profile_radius_nm", "median_profile_radius_pct_change",
            "iqr_profile_radius_pct_change", "frac_profile_annular",
            "frac_fit_success", "frac_bw_degenerate"]
    agg_w = agg[agg["sweep"] == "window_frac"]
    if len(agg_w):
        print("\nWindow-position sweep (u = 0 at floor, 1 at critical bandwidth):")
        print(agg_w[[c for c in cols if c in agg.columns]].to_string(index=False))
        print("\nAll sweeps:")
    print(agg[[c for c in cols if c in agg.columns]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
