# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bootstrap of all experimental MINFLUX traces for the stability of the refined
circle-fit radius.

Analysis functions are imported from
../Null Simulation/minflux_null_simulation_v5_fitted_circle_density.py.
Input: Extracted Trace/<folder>/*.xlsx, one sheet per trace.
"""
from __future__ import annotations

import os
# single-threaded numeric libraries per worker; must precede the numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import glob
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Null Simulation")) \
    if "__file__" in globals() else os.path.abspath(os.path.join(os.getcwd(), "..", "Null Simulation"))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
import minflux_null_simulation_v5_fitted_circle_density as core

core.KDE_EVALUATION_METHOD = "binned_gaussian"   # manuscript default
core.ENABLE_SUBVOXEL_2D_REFINEMENT = True        # set once (thread-safe: never mutated per call)

N_BOOTSTRAPS   = 1000        # bootstrap replicates per trace
N_WORKERS      = 8           # parallel workers
PARALLEL_MODE  = "process"   # "process" | "thread" | "serial"
RUN_BOTH_MODES = True        # re-estimated and fixed bandwidth, separate outputs
FIXED_BANDWIDTH = True       # used only when RUN_BOTH_MODES = False
LIMIT          = 0           # 0 = all traces

INPUT_ROOT = "Extracted Trace"
OUTPUT_ROOT = "bulk_bootstrap_output"
KEEP_MIN_LOCALIZATIONS = 800
RANDOM_SEED = 12345
X_COL, Y_COL, Z_COL = "X (nm)", "Y (nm)", "Z (nm)"

# grid built once per process
_VALS = None
_GRID_PTS = None


def _ensure_grid():
    global _VALS, _GRID_PTS
    if _VALS is None:
        vals, Xg, Yg, Zg, grid_pts = core.build_grid(core.VOX_NM)
        _VALS, _GRID_PTS = vals, grid_pts
    return _VALS, _GRID_PTS


def center_and_filter(pts_raw: np.ndarray) -> np.ndarray:
    pts_raw = pts_raw[np.all(np.isfinite(pts_raw), axis=1)]
    if pts_raw.shape[0] == 0:
        return pts_raw
    pts = pts_raw - pts_raw.mean(axis=0)
    pts += core.CENTER_NM
    inside = np.all((pts >= core.GRID_MIN_NM) & (pts <= core.GRID_MAX_NM), axis=1)
    return pts[inside]


def analyze(pts: np.ndarray, refine: bool, bandwidth: float = None) -> Dict:
    """Run the manuscript per-trace pipeline."""
    # Sub-voxel refinement is enabled globally once at import (thread-safe).
    vals, grid_pts = _ensure_grid()
    bw = core.calculate_global_kth_nn_bandwidth(pts, core.K_BAND) if bandwidth is None else float(bandwidth)
    density = core.kde_density_on_grid(pts, grid_pts, vals, bw, core.VOX_NM, core.KDE_GRID_CHUNK_SIZE)
    gmax = float(np.max(density)) if density.size else np.nan
    density_norm = density / gmax if (gmax and np.isfinite(gmax) and gmax > 0) else density
    peaks = core.find_2d_slice_peaks(density_norm, vals, pts, bw, int(len(pts)), 0)

    out = dict(coarse_radius_nm=np.nan, coarse_diameter_nm=np.nan,
               coarse_center_x_nm=np.nan, coarse_center_y_nm=np.nan, coarse_adj_pseudo_r2=np.nan,
               refined_radius_nm=np.nan, refined_diameter_nm=np.nan,
               refined_center_x_nm=np.nan, refined_center_y_nm=np.nan, refined_adj_pseudo_r2=np.nan,
               fit_success=False, n_peaks=int(len(peaks)), kde_bandwidth_nm=float(bw))
    if len(peaks) < core.MIN_PEAKS_FOR_CIRCLE_FIT:
        return out

    w = peaks["Normalized_Density_at_Peak"].to_numpy(float)
    cx, cy, r = core.weighted_circle_fit_xy(peaks["X_nm"].to_numpy(float),
                                            peaks["Y_nm"].to_numpy(float), weights=w)
    qc = core.circle_fit_quality_metrics(peaks["X_nm"].to_numpy(float),
                                         peaks["Y_nm"].to_numpy(float), cx, cy, r, weights=w)
    out.update(coarse_radius_nm=r, coarse_diameter_nm=2 * r, coarse_center_x_nm=cx,
               coarse_center_y_nm=cy, coarse_adj_pseudo_r2=qc["adjusted_pseudo_r2"], fit_success=True)

    if refine:
        refined = peaks.dropna(subset=["X_refined_nm", "Y_refined_nm"])
        if len(refined) >= core.MIN_PEAKS_FOR_CIRCLE_FIT:
            wr = refined["Normalized_Density_at_Peak"].to_numpy(float)
            cxr, cyr, rr = core.weighted_circle_fit_xy(refined["X_refined_nm"].to_numpy(float),
                                                       refined["Y_refined_nm"].to_numpy(float), weights=wr)
            qr = core.circle_fit_quality_metrics(refined["X_refined_nm"].to_numpy(float),
                                                 refined["Y_refined_nm"].to_numpy(float),
                                                 cxr, cyr, rr, weights=wr)
            out.update(refined_radius_nm=rr, refined_diameter_nm=2 * rr, refined_center_x_nm=cxr,
                       refined_center_y_nm=cyr, refined_adj_pseudo_r2=qr["adjusted_pseudo_r2"])
        else:
            out.update(refined_radius_nm=r, refined_diameter_nm=2 * r, refined_center_x_nm=cx,
                       refined_center_y_nm=cy, refined_adj_pseudo_r2=qc["adjusted_pseudo_r2"])
    return out


def _ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(median=np.nan, mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan, n=0)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), n=int(v.size))


def _offset_outside(meas, lo, hi):
    if not (np.isfinite(meas) and np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    if meas < lo:
        return meas - lo
    if meas > hi:
        return meas - hi
    return 0.0


def _bc_ci(boot_vals, theta_hat, alpha=0.05):
    """Bias-corrected (BC) percentile bootstrap CI."""
    v = np.asarray(boot_vals, float)
    v = v[np.isfinite(v)]
    B = v.size
    if B < 20 or not np.isfinite(theta_hat):
        return dict(lo=np.nan, hi=np.nan, z0=np.nan)
    frac = float(np.mean(v < theta_hat))
    frac = min(max(frac, 0.5 / B), 1 - 0.5 / B)   # clamp to avoid +/-inf
    z0 = float(norm.ppf(frac))
    zl, zu = norm.ppf(alpha / 2.0), norm.ppf(1 - alpha / 2.0)
    a1 = float(norm.cdf(2 * z0 + zl))
    a2 = float(norm.cdf(2 * z0 + zu))
    return dict(lo=float(np.percentile(v, 100 * a1)),
                hi=float(np.percentile(v, 100 * a2)), z0=z0)


def bootstrap_trace(pts: np.ndarray, seed: int, n_boot: int, fixed_bandwidth: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    ref = analyze(pts, refine=True)              # manuscript reference (refined)
    ref_bw = ref["kde_bandwidth_nm"]
    bw_for_boot = ref_bw if fixed_bandwidth else None

    rad, dia, cx, cy = [], [], [], []
    n_ok = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(pts), size=len(pts))
        pb = pts[idx]
        pb = pb - pb.mean(axis=0) + core.CENTER_NM
        pb = pb[np.all((pb >= core.GRID_MIN_NM) & (pb <= core.GRID_MAX_NM), axis=1)]
        if len(pb) < 10:
            continue
        try:
            fb = analyze(pb, refine=True, bandwidth=bw_for_boot)   # full refined method
        except Exception:
            continue
        if fb["fit_success"] and np.isfinite(fb["refined_radius_nm"]):
            n_ok += 1
            rad.append(fb["refined_radius_nm"])
            dia.append(fb["refined_diameter_nm"])
            cx.append(fb["refined_center_x_nm"])
            cy.append(fb["refined_center_y_nm"])

    r_ci, d_ci, cx_ci, cy_ci = _ci(rad), _ci(dia), _ci(cx), _ci(cy)
    mr, md = ref["refined_radius_nm"], ref["refined_diameter_nm"]
    r_bc, d_bc = _bc_ci(rad, mr), _bc_ci(dia, md)

    row = dict(
        measured_radius_nm=mr, measured_diameter_nm=md,
        measured_adj_pseudo_r2=ref["refined_adj_pseudo_r2"],
        measured_fit_success=ref["fit_success"],
        n_peaks_measured=ref["n_peaks"], kde_bandwidth_nm=ref_bw,
        fixed_bandwidth=bool(fixed_bandwidth),
        # primary: bootstrap standard error on the radius
        boot_radius_se_nm=r_ci["sd"],
        boot_radius_rel_se_pct=(100 * r_ci["sd"] / mr) if np.isfinite(mr) and mr > 0 else np.nan,
        boot_radius_mean_nm=r_ci["mean"], boot_radius_median_nm=r_ci["median"],
        # bias-corrected point estimate = 2*theta_hat - mean(boot)
        radius_bias_corrected_nm=(2 * mr - r_ci["mean"]) if np.isfinite(mr) else np.nan,
        # percentile CI (naive)
        boot_radius_ci_low_nm=r_ci["lo"], boot_radius_ci_high_nm=r_ci["hi"],
        radius_within_ci=bool(np.isfinite(mr) and r_ci["lo"] <= mr <= r_ci["hi"]),
        radius_bias_nm=(mr - r_ci["median"]) if np.isfinite(mr) else np.nan,
        radius_ci_width_nm=(r_ci["hi"] - r_ci["lo"]) if np.isfinite(r_ci["lo"]) else np.nan,
        radius_offset_outside_ci_nm=_offset_outside(mr, r_ci["lo"], r_ci["hi"]),
        # bias-corrected (BC) CI
        boot_radius_bc_ci_low_nm=r_bc["lo"], boot_radius_bc_ci_high_nm=r_bc["hi"],
        radius_bc_z0=r_bc["z0"],
        radius_bc_within_ci=bool(np.isfinite(mr) and r_bc["lo"] <= mr <= r_bc["hi"]),
        radius_bc_ci_width_nm=(r_bc["hi"] - r_bc["lo"]) if np.isfinite(r_bc["lo"]) else np.nan,
        # diameter (SE + both CIs)
        boot_diameter_se_nm=d_ci["sd"], boot_diameter_median_nm=d_ci["median"],
        boot_diameter_ci_low_nm=d_ci["lo"], boot_diameter_ci_high_nm=d_ci["hi"],
        diameter_within_ci=bool(np.isfinite(md) and d_ci["lo"] <= md <= d_ci["hi"]),
        boot_diameter_bc_ci_low_nm=d_bc["lo"], boot_diameter_bc_ci_high_nm=d_bc["hi"],
        diameter_bc_within_ci=bool(np.isfinite(md) and d_bc["lo"] <= md <= d_bc["hi"]),
        # center CIs
        boot_center_x_ci_low_nm=cx_ci["lo"], boot_center_x_ci_high_nm=cx_ci["hi"],
        boot_center_y_ci_low_nm=cy_ci["lo"], boot_center_y_ci_high_nm=cy_ci["hi"],
        n_boot_success=n_ok, n_boot_requested=n_boot,
        # distributions (popped before CSV; saved to npz)
        _boot_radius=np.asarray(rad, dtype=np.float32),
        _boot_diameter=np.asarray(dia, dtype=np.float32),
    )
    return row


def enumerate_traces(input_root: str):
    import openpyxl
    out = []
    for f in sorted(glob.glob(os.path.join(input_root, "*", "*.xlsx"))):
        construct = os.path.basename(os.path.dirname(f))
        wb = openpyxl.load_workbook(f, read_only=True)
        for s in wb.sheetnames:
            out.append((f, construct, s))
        wb.close()
    return out


def worker(args):
    fpath, construct, sheet, seed, n_boot, fixed_bw = args
    try:
        df = pd.read_excel(fpath, sheet_name=sheet)
        if not {X_COL, Y_COL, Z_COL}.issubset(df.columns):
            return None
        pts = center_and_filter(df[[X_COL, Y_COL, Z_COL]].to_numpy(float))
        if len(pts) < KEEP_MIN_LOCALIZATIONS:
            return None
        row = bootstrap_trace(pts, seed, n_boot, fixed_bandwidth=fixed_bw)
        row.update(construct=construct, source_file=os.path.basename(fpath),
                   sheet=sheet, n_locs=int(len(pts)))
        return row
    except Exception as exc:
        warnings.warn(f"{os.path.basename(fpath)}|{sheet} failed: {exc}")
        return None


def build_aggregate(df, out_root):
    ok = df[df["measured_fit_success"]].copy()

    def s(sub, label):
        return dict(group=label, n_traces=int(len(sub)),
                    median_measured_radius_nm=float(sub["measured_radius_nm"].median()),
                    # primary: bootstrap standard error / precision
                    median_boot_radius_se_nm=float(sub["boot_radius_se_nm"].median()),
                    median_boot_radius_rel_se_pct=float(sub["boot_radius_rel_se_pct"].median()),
                    # coverage: naive percentile vs bias-corrected
                    frac_radius_within_ci=float(sub["radius_within_ci"].mean()) if len(sub) else np.nan,
                    frac_radius_within_BC_ci=float(sub["radius_bc_within_ci"].mean()) if len(sub) else np.nan,
                    median_radius_bias_nm=float(sub["radius_bias_nm"].median()),
                    median_percentile_ci_width_nm=float(sub["radius_ci_width_nm"].median()),
                    median_BC_ci_width_nm=float(sub["radius_bc_ci_width_nm"].median()))
    rows = [s(ok, "ALL")] + [s(sub, con) for con, sub in ok.groupby("construct")]
    agg = pd.DataFrame(rows)
    agg.to_csv(out_root / "tables" / "bulk_bootstrap_aggregate.csv", index=False)
    print("\nAggregate summary:")
    print(agg.to_string(index=False))


def make_figures(df, out_root):
    ok = df[df["measured_fit_success"] & np.isfinite(df["measured_radius_nm"])].copy()
    ok = ok.sort_values("measured_radius_nm").reset_index(drop=True)
    fd = out_root / "figures"
    x = np.arange(len(ok))

    # (1) measured radius +/- bootstrap SE  [PRIMARY]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.errorbar(x, ok["measured_radius_nm"], yerr=ok["boot_radius_se_nm"],
                fmt=".", ms=3, elinewidth=0.6, capsize=0, alpha=0.8, label="Measured radius +/- bootstrap SE")
    ax.set_xlabel("Trace (sorted by measured radius)"); ax.set_ylabel("Ring radius (nm)")
    ax.set_title("Per-trace ring radius with bootstrap standard error"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "bulk_radius_with_SE.png", transparent=True); plt.close(fig)

    # (2) measured vs naive percentile CI vs BC CI
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.fill_between(x, ok["boot_radius_ci_low_nm"], ok["boot_radius_ci_high_nm"],
                    alpha=0.25, label="Percentile 95% CI (naive)")
    ax.fill_between(x, ok["boot_radius_bc_ci_low_nm"], ok["boot_radius_bc_ci_high_nm"],
                    alpha=0.35, label="Bias-corrected 95% CI")
    ax.plot(x, ok["measured_radius_nm"], ".", ms=4, color="k", label="Measured radius")
    ax.set_xlabel("Trace (sorted by measured radius)"); ax.set_ylabel("Ring radius (nm)")
    ax.set_title("Measured radius vs percentile and bias-corrected 95% CI")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "bulk_measured_vs_ci_radius.png", transparent=True); plt.close(fig)

    # (3) histograms: relative SE, bias, BC CI width
    for col, fn, xl in [("boot_radius_rel_se_pct", "bulk_relSE_hist.png", "Bootstrap relative SE, radius (%)"),
                        ("radius_bias_nm", "bulk_bias_hist.png", "Radius bias: measured - bootstrap median (nm)"),
                        ("radius_bc_ci_width_nm", "bulk_ci_width_hist.png", "Bias-corrected 95% CI width, radius (nm)")]:
        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
        ax.hist(ok[col].dropna(), bins="auto", alpha=0.8)
        if col == "radius_bias_nm":
            ax.axvline(0, ls="--", lw=1)
        ax.set_xlabel(xl); ax.set_ylabel("Trace count")
        fig.tight_layout(); fig.savefig(fd / fn, transparent=True); plt.close(fig)


def _dispatch(tasks, mode, workers):
    """Yield worker results as they complete, for the chosen parallel backend."""
    if mode == "serial" or workers <= 1:
        for t in tasks:
            yield worker(t)
    elif mode == "thread":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for f in as_completed(futs):
                yield f.result()
    else:  # "process"
        from multiprocessing import Pool
        with Pool(processes=workers) as pool:
            for r in pool.imap_unordered(worker, tasks):
                yield r


def run_one_mode(traces, n_boot, fixed_bw, out_root, mode, workers):
    out_root = Path(out_root)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    per_trace_csv = out_root / "tables" / "bulk_bootstrap_per_trace.csv"
    if per_trace_csv.exists():
        per_trace_csv.unlink()

    tasks = [(f, c, s, RANDOM_SEED + i, n_boot, fixed_bw) for i, (f, c, s) in enumerate(traces)]
    label = "fixed-bandwidth" if fixed_bw else "re-estimated-bandwidth"
    print(f"\n=== Bootstrap mode: {label}  |  {len(tasks)} traces  |  "
          f"{n_boot} boots  |  {mode} x{workers}  ->  {out_root} ===", flush=True)

    results, dists, header_written, done, t0 = [], {}, False, 0, time.time()
    for row in _dispatch(tasks, mode, workers):
        done += 1
        if row is not None:
            key = f"{row['source_file']}|{row['sheet']}"
            dists[key + "|radius"] = row.pop("_boot_radius", np.array([], dtype=np.float32))
            dists[key + "|diameter"] = row.pop("_boot_diameter", np.array([], dtype=np.float32))
            results.append(row)
            pd.DataFrame([row]).to_csv(per_trace_csv, mode="a", header=not header_written, index=False)
            header_written = True
        if done % 10 == 0 or done == len(tasks):
            el = time.time() - t0
            print(f"  {done}/{len(tasks)} processed (kept {len(results)})  "
                  f"[{el/60:.1f} min, {el/max(done,1):.1f} s/trace]", flush=True)

    if not results:
        print("  No traces passed the localization filter."); return None
    df = pd.DataFrame(results).sort_values(["construct", "source_file", "sheet"]).reset_index(drop=True)
    df.to_csv(per_trace_csv, index=False)
    # full bootstrap distributions (for reproducible BC/BCa or re-analysis)
    np.savez_compressed(out_root / "tables" / "bootstrap_distributions.npz", **dists)
    build_aggregate(df, out_root)
    make_figures(df, out_root)
    with open(out_root / "analysis_parameters.json", "w") as fh:
        json.dump(dict(keep_min_localizations=KEEP_MIN_LOCALIZATIONS, n_bootstraps=n_boot,
                       random_seed=RANDOM_SEED, kde_evaluation_method=core.KDE_EVALUATION_METHOD,
                       measured_metric="circle_fit_refined_radius_nm", k_band=core.K_BAND,
                       fixed_bandwidth=bool(fixed_bw), parallel_mode=mode, n_workers=workers,
                       bootstrap_method="full_refined_pipeline_per_replicate",
                       ci_methods=["bootstrap_SE", "percentile_95", "bias_corrected_BC_95"],
                       bca_note="acceleration term omitted (localization jackknife intractable); BC used"),
                  fh, indent=2)
    print(f"  Done. Kept {len(df)} traces -> {out_root.resolve()}", flush=True)
    return df


def main():
    _ensure_grid()  # pre-build shared grid before any threads start
    traces = enumerate_traces(INPUT_ROOT)
    if LIMIT:
        traces = traces[:LIMIT]
    print(f"Enumerated {len(traces)} candidate sheets/traces.", flush=True)

    if RUN_BOTH_MODES:
        modes = [(False, OUTPUT_ROOT + "_reest"), (True, OUTPUT_ROOT + "_fixedbw")]
    else:
        modes = [(FIXED_BANDWIDTH, OUTPUT_ROOT + ("_fixedbw" if FIXED_BANDWIDTH else "_reest"))]

    for fixed_bw, out_dir in modes:
        run_one_mode(traces, N_BOOTSTRAPS, fixed_bw, out_dir, PARALLEL_MODE, max(1, N_WORKERS))


# Spyder runs the file as __main__, so this executes on %runfile.
if __name__ == "__main__":
    main()
