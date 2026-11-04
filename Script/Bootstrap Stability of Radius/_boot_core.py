# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bootstrap of a single MINFLUX trace: uncertainty of KDE XY peak positions and of
the density-weighted circle-fit centre, radius and diameter.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity, NearestNeighbors
from skimage.feature import peak_local_max
from tqdm import tqdm


# USER PARAMETERS

@dataclass
class Config:
    # Input / output
    excel_file: str = "Nup58C M0101Tid4156.xlsx"
    output_dir: str = "bootstrap_real_trace_xy_ring_output"
    sheet_name: Union[int, str] = 0

    # Coordinate columns in the Excel file
    x_col: str = "X (nm)"
    y_col: str = "Y (nm)"
    z_col: str = "Z (nm)"

    # KDE / grid parameters
    cube_min_nm: float = 0.0
    cube_max_nm: float = 120.0
    cube_center_nm: float = 60.0
    vox_nm: float = 1.0
    k_band: int = 3
    min_bandwidth_nm: float = 0.1

    z_centers_nm: Tuple[float, ...] = (30, 40, 50, 60, 70, 80, 90)
    slice_thickness_nm: float = 1.0
    zoom_min_nm: float = 30.0
    zoom_max_nm: float = 90.0

    # 2D peak detection parameters copied from the existing analyzer logic.
    min_dist_2d_peaks_vox: int = 2
    normalized_global_noise_cutoff: float = 0.20
    slice_relative_threshold_fraction: float = 0.50
    exclude_border: bool = False

    min_peaks_for_circle_fit: int = 3

    # Bootstrap parameters
    n_bootstraps: int = 10000
    random_seed: int = 12345
    bootstrap_mode: str = "point"   # "point" or "block"
    block_length: int = 25           # used only for bootstrap_mode="block"

    max_peak_match_distance_nm: float = 5.0

    circle_weight_mode: str = "density"

    # Diagnostics / plots
    make_plots: bool = True
    dpi: int = 300


CFG = Config()


# HELPER FUNCTIONS

def parse_prefix_from_filename(filename: str) -> str:
    """Parse M####Tid#### from filenames like 'Nup58C M0101Tid4156.xlsx'."""
    base = os.path.basename(filename)
    match = re.search(r"\sM(\d{2})(\d{2})Tid(\d+)\.xlsx$", base)
    if match:
        meas_idx, sub_idx, tid_val = match.groups()
        return f"M{meas_idx}{sub_idx}Tid{tid_val}"
    return Path(filename).stem.replace(" ", "_")


def ensure_output_dirs(output_dir: str) -> Dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "debug": root / "debug",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_and_center_trace(cfg: Config) -> np.ndarray:
    """Load raw coordinates, geometric-recenter them, shift to the 120 nm cube, and filter inside cube."""
    df = pd.read_excel(cfg.excel_file, sheet_name=cfg.sheet_name)
    required = {cfg.x_col, cfg.y_col, cfg.z_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required coordinate columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    pts_raw = df[[cfg.x_col, cfg.y_col, cfg.z_col]].to_numpy(dtype=np.float64)
    pts_raw = pts_raw[np.all(np.isfinite(pts_raw), axis=1)]
    if pts_raw.shape[0] == 0:
        raise ValueError("No finite localization coordinates found.")

    pts = pts_raw - pts_raw.mean(axis=0)
    pts += cfg.cube_center_nm
    inside = np.all((pts >= cfg.cube_min_nm) & (pts <= cfg.cube_max_nm), axis=1)
    pts = pts[inside]
    if pts.shape[0] < 10:
        raise ValueError(f"Only {pts.shape[0]} points remain inside the cube after recentering/filtering.")
    return pts


def estimate_global_kde_bandwidth(pts: np.ndarray, k_band: int, min_bandwidth_nm: float) -> float:
    """Estimate trace-specific global KDE bandwidth from median k-NN distance."""
    k = min(max(1, int(k_band)), len(pts))
    nn = NearestNeighbors(n_neighbors=k).fit(pts)
    distances, _ = nn.kneighbors(pts)
    bw = float(np.median(distances[:, k - 1]))
    if (not np.isfinite(bw)) or bw < min_bandwidth_nm:
        bw = float(min_bandwidth_nm)
    return bw


def bootstrap_points(pts: np.ndarray, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Resample localizations with replacement. Supports point or moving-block bootstrap."""
    n = len(pts)
    mode = cfg.bootstrap_mode.lower().strip()
    if mode == "point":
        idx = rng.integers(0, n, size=n)
        return pts[idx].copy()
    if mode == "block":
        block_len = max(1, int(cfg.block_length))
        starts = rng.integers(0, n, size=int(math.ceil(n / block_len)))
        chunks = []
        for s in starts:
            # Circular moving block so all starts are valid.
            block_idx = (np.arange(s, s + block_len) % n).astype(int)
            chunks.append(pts[block_idx])
        return np.vstack(chunks)[:n].copy()
    raise ValueError("bootstrap_mode must be 'point' or 'block'.")


def recenter_bootstrap_sample(pts_boot: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply the same geometric recentering to each bootstrap sample."""
    pts = pts_boot - pts_boot.mean(axis=0)
    pts += cfg.cube_center_nm
    inside = np.all((pts >= cfg.cube_min_nm) & (pts <= cfg.cube_max_nm), axis=1)
    pts = pts[inside]
    return pts


def values_for_zoom_grid(cfg: Config) -> np.ndarray:
    """Return 1D coordinate values for the zoomed XY grid."""
    # Include endpoints. For vox=1, this gives 30,31,...,90.
    return np.arange(cfg.zoom_min_nm, cfg.zoom_max_nm + 0.5 * cfg.vox_nm, cfg.vox_nm, dtype=np.float64)


def z_values_for_slice(center_z: float, cfg: Config) -> np.ndarray:
    """Return Z planes included in a slice/projection."""
    half = cfg.slice_thickness_nm / 2.0
    vals = np.arange(cfg.cube_min_nm, cfg.cube_max_nm + 0.5 * cfg.vox_nm, cfg.vox_nm, dtype=np.float64)
    use = vals[(vals >= center_z - half - 1e-9) & (vals <= center_z + half + 1e-9)]
    if len(use) == 0:
        use = np.array([float(center_z)], dtype=np.float64)
    return use


def kde_xy_slices_zoom(pts: np.ndarray, cfg: Config) -> Tuple[Dict[float, np.ndarray], float, float]:
    """Compute KDE only on the XY zoom grid and requested Z slices."""
    bw = estimate_global_kde_bandwidth(pts, cfg.k_band, cfg.min_bandwidth_nm)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(pts)

    xy_vals = values_for_zoom_grid(cfg)
    X, Y = np.meshgrid(xy_vals, xy_vals, indexing="xy")
    xy_flat = np.column_stack([X.ravel(), Y.ravel()])

    projections: Dict[float, np.ndarray] = {}
    max_density = 0.0
    voxel_volume = cfg.vox_nm ** 3

    for zc in cfg.z_centers_nm:
        zvals = z_values_for_slice(float(zc), cfg)
        mats = []
        for zv in zvals:
            grid_pts = np.column_stack([xy_flat, np.full(len(xy_flat), zv, dtype=np.float64)])
            density = np.exp(kde.score_samples(grid_pts)).reshape(len(xy_vals), len(xy_vals))
            density = density * len(pts) / voxel_volume  # loc/nm^3, matching existing script convention
            mats.append(density)
        proj = np.mean(mats, axis=0)
        projections[float(zc)] = proj
        if proj.size:
            max_density = max(max_density, float(np.nanmax(proj)))

    if max_density <= 0 or not np.isfinite(max_density):
        max_density = 1.0
    return projections, bw, max_density


def quadratic_subpixel_refine_2d(matrix: np.ndarray, row: int, col: int, pixel_size_nm: float) -> Tuple[float, float, float]:
    """Refine a 2D peak using a local 3x3 quadratic surface."""
    nrows, ncols = matrix.shape
    if row <= 0 or row >= nrows - 1 or col <= 0 or col >= ncols - 1:
        return 0.0, 0.0, float(matrix[row, col])

    patch = matrix[row - 1:row + 2, col - 1:col + 2].astype(float)
    if not np.all(np.isfinite(patch)):
        return 0.0, 0.0, float(matrix[row, col])

    xs, ys = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], indexing="xy")
    A = np.column_stack([
        xs.ravel() ** 2,
        ys.ravel() ** 2,
        xs.ravel() * ys.ravel(),
        xs.ravel(),
        ys.ravel(),
        np.ones(9),
    ])
    b = patch.ravel()

    try:
        a, b2, c, d, e, f = np.linalg.lstsq(A, b, rcond=None)[0]
        H = np.array([[2 * a, c], [c, 2 * b2]], dtype=float)
        g = np.array([d, e], dtype=float)
        vertex = -np.linalg.solve(H, g)  # [x, y] in pixel units
        if not np.all(np.isfinite(vertex)):
            raise ValueError("Non-finite vertex")
        if abs(vertex[0]) > 1.0 or abs(vertex[1]) > 1.0:
            raise ValueError("Vertex outside 3x3 neighborhood")
        refined_val = float(np.array([vertex[0] ** 2, vertex[1] ** 2, vertex[0] * vertex[1], vertex[0], vertex[1], 1.0]) @ np.array([a, b2, c, d, e, f]))
        if refined_val < matrix[row, col] * 0.90:
            raise ValueError("Refined value unexpectedly lower")
        return float(vertex[0] * pixel_size_nm), float(vertex[1] * pixel_size_nm), refined_val
    except Exception:
        return 0.0, 0.0, float(matrix[row, col])


def detect_peaks_in_projections(
    projections_physical: Dict[float, np.ndarray],
    global_max_density: float,
    cfg: Config,
    prefix: str,
) -> pd.DataFrame:
    """Detect and sub-voxel refine local maxima in each XY projection."""
    xy_vals = values_for_zoom_grid(cfg)
    rows = []
    peak_id = 1

    for zc, physical in projections_physical.items():
        norm = physical / global_max_density
        if norm.size == 0 or not np.any(np.isfinite(norm)):
            continue

        slice_max_norm = float(np.nanmax(norm))
        threshold_abs = max(
            slice_max_norm * cfg.slice_relative_threshold_fraction,
            cfg.normalized_global_noise_cutoff,
        )

        peak_pixels = peak_local_max(
            norm,
            min_distance=int(cfg.min_dist_2d_peaks_vox),
            threshold_abs=threshold_abs,
            exclude_border=cfg.exclude_border,
        )

        # Sort high to low density for stable output.
        if len(peak_pixels):
            values = np.array([norm[r, c] for r, c in peak_pixels])
            peak_pixels = peak_pixels[np.argsort(values)[::-1]]

        for rank, (r, c) in enumerate(peak_pixels, start=1):
            dx, dy, refined_norm_val = quadratic_subpixel_refine_2d(norm, int(r), int(c), cfg.vox_nm)
            x_nm = float(xy_vals[c] + dx)
            y_nm = float(xy_vals[r] + dy)
            refined_physical_val = float(refined_norm_val * global_max_density)

            rows.append({
                "prefix": prefix,
                "peak_id": peak_id,
                "z_center_nm": float(zc),
                "rank_in_slice": rank,
                "x_nm": x_nm,
                "y_nm": y_nm,
                "x_integer_nm": float(xy_vals[c]),
                "y_integer_nm": float(xy_vals[r]),
                "dx_subvoxel_nm": float(dx),
                "dy_subvoxel_nm": float(dy),
                "density_physical": refined_physical_val,
                "density_normalized_to_requested_slices": float(refined_norm_val),
                "threshold_normalized": float(threshold_abs),
                "slice_max_normalized": float(slice_max_norm),
            })
            peak_id += 1

    return pd.DataFrame(rows)


def initial_circle_guess(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
    """Weighted algebraic circle initial guess."""
    w = np.asarray(w, dtype=float)
    if np.sum(w) <= 0:
        w = np.ones_like(x)
    w = w / np.sum(w)
    x0 = float(np.sum(w * x))
    y0 = float(np.sum(w * y))
    r0 = float(np.sum(w * np.sqrt((x - x0) ** 2 + (y - y0) ** 2)))
    if not np.isfinite(r0) or r0 <= 0:
        r0 = float(np.std(np.concatenate([x, y])))
    return x0, y0, max(r0, 1e-6)


def weighted_circle_fit(peaks_df: pd.DataFrame, cfg: Config) -> Dict[str, float | bool | str]:
    """Weighted nonlinear least-squares circle fit to XY peak coordinates."""
    if peaks_df is None or len(peaks_df) < cfg.min_peaks_for_circle_fit:
        return {
            "fit_success": False,
            "fit_message": f"Need at least {cfg.min_peaks_for_circle_fit} peaks; got {0 if peaks_df is None else len(peaks_df)}",
            "circle_center_x_nm": np.nan,
            "circle_center_y_nm": np.nan,
            "circle_radius_nm": np.nan,
            "circle_diameter_nm": np.nan,
            "circle_rmse_nm": np.nan,
            "circle_weighted_rmse_nm": np.nan,
            "circle_r2": np.nan,
            "circle_adj_r2": np.nan,
            "n_peaks_fit": 0 if peaks_df is None else int(len(peaks_df)),
        }

    x = peaks_df["x_nm"].to_numpy(float)
    y = peaks_df["y_nm"].to_numpy(float)

    if cfg.circle_weight_mode.lower() == "density" and "density_physical" in peaks_df.columns:
        w = peaks_df["density_physical"].to_numpy(float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if np.sum(w) <= 0:
            w = np.ones_like(x)
    else:
        w = np.ones_like(x)

    w = w / np.nanmax(w)
    w = np.clip(w, 1e-6, None)
    sqrtw = np.sqrt(w)
    x0, y0, r0 = initial_circle_guess(x, y, w)

    def residual(params: np.ndarray) -> np.ndarray:
        cx, cy, r = params
        radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return sqrtw * (radii - r)

    try:
        res = least_squares(
            residual,
            x0=np.array([x0, y0, r0], dtype=float),
            bounds=([cfg.zoom_min_nm - 30, cfg.zoom_min_nm - 30, 0.0], [cfg.zoom_max_nm + 30, cfg.zoom_max_nm + 30, 100.0]),
            max_nfev=5000,
        )
        cx, cy, r = [float(v) for v in res.x]
        radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        resid_unweighted = radii - r
        rmse = float(np.sqrt(np.mean(resid_unweighted ** 2)))
        wrmse = float(np.sqrt(np.sum(w * resid_unweighted ** 2) / np.sum(w)))

        # R2 on radial constancy around the fitted circle, not on the full density map.
        ybar_w = float(np.sum(w * radii) / np.sum(w))
        ss_res = float(np.sum(w * (radii - r) ** 2))
        ss_tot = float(np.sum(w * (radii - ybar_w) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        n = len(x)
        p = 3
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)) if np.isfinite(r2) else np.nan

        return {
            "fit_success": bool(res.success),
            "fit_message": str(res.message),
            "circle_center_x_nm": cx,
            "circle_center_y_nm": cy,
            "circle_radius_nm": r,
            "circle_diameter_nm": 2.0 * r,
            "circle_rmse_nm": rmse,
            "circle_weighted_rmse_nm": wrmse,
            "circle_r2": r2,
            "circle_adj_r2": adj_r2,
            "n_peaks_fit": int(len(x)),
        }
    except Exception as exc:
        return {
            "fit_success": False,
            "fit_message": f"Circle fit failed: {exc}",
            "circle_center_x_nm": np.nan,
            "circle_center_y_nm": np.nan,
            "circle_radius_nm": np.nan,
            "circle_diameter_nm": np.nan,
            "circle_rmse_nm": np.nan,
            "circle_weighted_rmse_nm": np.nan,
            "circle_r2": np.nan,
            "circle_adj_r2": np.nan,
            "n_peaks_fit": int(len(peaks_df)),
        }


def match_bootstrap_peaks_to_reference(
    boot_peaks: pd.DataFrame,
    ref_peaks: pd.DataFrame,
    cfg: Config,
    bootstrap_id: int,
) -> pd.DataFrame:
    """Nearest-neighbor match bootstrap peaks to original/reference peaks within each Z slice."""
    rows = []
    if boot_peaks is None or boot_peaks.empty or ref_peaks is None or ref_peaks.empty:
        return pd.DataFrame(rows)

    for zc, ref_slice in ref_peaks.groupby("z_center_nm", sort=True):
        boot_slice = boot_peaks[np.isclose(boot_peaks["z_center_nm"].astype(float), float(zc))]
        if boot_slice.empty:
            for _, rr in ref_slice.iterrows():
                rows.append({
                    "bootstrap_id": bootstrap_id,
                    "ref_peak_id": int(rr["peak_id"]),
                    "z_center_nm": float(zc),
                    "matched": False,
                    "match_distance_nm": np.nan,
                    "x_nm": np.nan,
                    "y_nm": np.nan,
                    "density_physical": np.nan,
                    "ref_x_nm": float(rr["x_nm"]),
                    "ref_y_nm": float(rr["y_nm"]),
                })
            continue

        D = cdist(ref_slice[["x_nm", "y_nm"]].to_numpy(float), boot_slice[["x_nm", "y_nm"]].to_numpy(float))
        used_boot = set()
        # Greedy smallest-distance unique matching.
        candidate_pairs = [(float(D[i, j]), i, j) for i in range(D.shape[0]) for j in range(D.shape[1])]
        candidate_pairs.sort(key=lambda t: t[0])
        matched_ref = {}
        for dist, i, j in candidate_pairs:
            if dist > cfg.max_peak_match_distance_nm:
                break
            if i in matched_ref or j in used_boot:
                continue
            matched_ref[i] = (j, dist)
            used_boot.add(j)

        ref_records = ref_slice.reset_index(drop=True)
        boot_records = boot_slice.reset_index(drop=True)
        for i, rr in ref_records.iterrows():
            if i in matched_ref:
                j, dist = matched_ref[i]
                br = boot_records.iloc[j]
                rows.append({
                    "bootstrap_id": bootstrap_id,
                    "ref_peak_id": int(rr["peak_id"]),
                    "z_center_nm": float(zc),
                    "matched": True,
                    "match_distance_nm": float(dist),
                    "x_nm": float(br["x_nm"]),
                    "y_nm": float(br["y_nm"]),
                    "density_physical": float(br["density_physical"]),
                    "ref_x_nm": float(rr["x_nm"]),
                    "ref_y_nm": float(rr["y_nm"]),
                })
            else:
                rows.append({
                    "bootstrap_id": bootstrap_id,
                    "ref_peak_id": int(rr["peak_id"]),
                    "z_center_nm": float(zc),
                    "matched": False,
                    "match_distance_nm": np.nan,
                    "x_nm": np.nan,
                    "y_nm": np.nan,
                    "density_physical": np.nan,
                    "ref_x_nm": float(rr["x_nm"]),
                    "ref_y_nm": float(rr["y_nm"]),
                })
    return pd.DataFrame(rows)


def ci_summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "n": int(arr.size),
    }


def summarize_circle_bootstrap(circle_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = circle_df[circle_df["fit_success"].astype(bool)].copy()
    for col in [
        "kde_bandwidth_nm",
        "n_detected_peaks",
        "n_peaks_fit",
        "circle_center_x_nm",
        "circle_center_y_nm",
        "circle_radius_nm",
        "circle_diameter_nm",
        "circle_rmse_nm",
        "circle_weighted_rmse_nm",
        "circle_r2",
        "circle_adj_r2",
    ]:
        s = ci_summary(ok[col].to_numpy(float) if col in ok.columns else [])
        s["metric"] = col
        rows.append(s)
    return pd.DataFrame(rows)[["metric", "n", "mean", "median", "sd", "ci95_low", "ci95_high"]]


def summarize_peak_bootstrap(matched_df: pd.DataFrame, ref_peaks: pd.DataFrame, n_bootstraps: int) -> pd.DataFrame:
    rows = []
    if matched_df.empty:
        return pd.DataFrame(rows)
    for ref_id, grp in matched_df.groupby("ref_peak_id", sort=True):
        valid = grp[grp["matched"].astype(bool)]
        ref_row = ref_peaks.loc[ref_peaks["peak_id"] == ref_id].iloc[0]
        sx = ci_summary(valid["x_nm"].to_numpy(float))
        sy = ci_summary(valid["y_nm"].to_numpy(float))
        sr = ci_summary(np.sqrt((valid["x_nm"].to_numpy(float) - ref_row["x_nm"]) ** 2 + (valid["y_nm"].to_numpy(float) - ref_row["y_nm"]) ** 2))
        rows.append({
            "ref_peak_id": int(ref_id),
            "z_center_nm": float(ref_row["z_center_nm"]),
            "ref_x_nm": float(ref_row["x_nm"]),
            "ref_y_nm": float(ref_row["y_nm"]),
            "ref_density_physical": float(ref_row["density_physical"]),
            "n_bootstraps": int(n_bootstraps),
            "n_matched": int(len(valid)),
            "match_fraction": float(len(valid) / n_bootstraps),
            "x_mean_nm": sx["mean"],
            "x_median_nm": sx["median"],
            "x_sd_nm": sx["sd"],
            "x_ci95_low_nm": sx["ci95_low"],
            "x_ci95_high_nm": sx["ci95_high"],
            "y_mean_nm": sy["mean"],
            "y_median_nm": sy["median"],
            "y_sd_nm": sy["sd"],
            "y_ci95_low_nm": sy["ci95_low"],
            "y_ci95_high_nm": sy["ci95_high"],
            "radial_error_from_ref_mean_nm": sr["mean"],
            "radial_error_from_ref_ci95_low_nm": sr["ci95_low"],
            "radial_error_from_ref_ci95_high_nm": sr["ci95_high"],
        })
    return pd.DataFrame(rows)


def plot_reference_peaks_and_circle(ref_peaks: pd.DataFrame, fit: Dict, cfg: Config, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=cfg.dpi)
    sc = ax.scatter(ref_peaks["x_nm"], ref_peaks["y_nm"], c=ref_peaks["z_center_nm"], s=45, marker="x")
    if fit.get("fit_success") and np.isfinite(fit.get("circle_radius_nm", np.nan)):
        theta = np.linspace(0, 2 * np.pi, 400)
        cx = fit["circle_center_x_nm"]
        cy = fit["circle_center_y_nm"]
        r = fit["circle_radius_nm"]
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), linewidth=1.5)
        ax.scatter([cx], [cy], s=30, marker="+")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Z center (nm)")
    ax.set_xlim(cfg.zoom_min_nm, cfg.zoom_max_nm)
    ax.set_ylim(cfg.zoom_min_nm, cfg.zoom_max_nm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_title("Reference 2D KDE peaks and weighted circle fit")
    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def plot_circle_bootstrap(circle_df: pd.DataFrame, ref_fit: Dict, cfg: Config, out_dir: Path) -> None:
    ok = circle_df[circle_df["fit_success"].astype(bool)].copy()
    if ok.empty:
        return

    # Diameter histogram
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=cfg.dpi)
    ax.hist(ok["circle_diameter_nm"].dropna(), bins="auto", alpha=0.8)
    if ref_fit.get("fit_success") and np.isfinite(ref_fit.get("circle_diameter_nm", np.nan)):
        ax.axvline(ref_fit["circle_diameter_nm"], linestyle="--", linewidth=1.5, label="Original trace")
        ax.legend(frameon=False)
    ax.set_xlabel("Weighted circle-fit diameter (nm)")
    ax.set_ylabel("Bootstrap count")
    ax.set_title("Bootstrap annular diameter")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_diameter_histogram.png", transparent=True)
    plt.close(fig)

    # Center scatter
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=cfg.dpi)
    ax.scatter(ok["circle_center_x_nm"], ok["circle_center_y_nm"], s=8, alpha=0.35)
    if ref_fit.get("fit_success"):
        ax.scatter([ref_fit["circle_center_x_nm"]], [ref_fit["circle_center_y_nm"]], s=50, marker="+")
    ax.set_aspect("equal", adjustable="box")
    pad = 5
    vals = np.concatenate([ok["circle_center_x_nm"].dropna().to_numpy(), ok["circle_center_y_nm"].dropna().to_numpy()])
    if vals.size:
        lo, hi = np.percentile(vals, [1, 99])
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Circle center X (nm)")
    ax.set_ylabel("Circle center Y (nm)")
    ax.set_title("Bootstrap fitted centers")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_circle_center_scatter.png", transparent=True)
    plt.close(fig)


def plot_peak_bootstrap_cloud(matched_df: pd.DataFrame, ref_peaks: pd.DataFrame, ref_fit: Dict, cfg: Config, out_dir: Path) -> None:
    valid = matched_df[matched_df["matched"].astype(bool)].copy()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=cfg.dpi)
    ax.scatter(valid["x_nm"], valid["y_nm"], s=5, alpha=0.15, label="Bootstrap matched peaks")
    ax.scatter(ref_peaks["x_nm"], ref_peaks["y_nm"], s=50, marker="x", label="Original peaks")
    if ref_fit.get("fit_success"):
        theta = np.linspace(0, 2 * np.pi, 400)
        cx = ref_fit["circle_center_x_nm"]
        cy = ref_fit["circle_center_y_nm"]
        r = ref_fit["circle_radius_nm"]
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), linewidth=1.5, label="Original circle")
    ax.set_xlim(cfg.zoom_min_nm, cfg.zoom_max_nm)
    ax.set_ylim(cfg.zoom_min_nm, cfg.zoom_max_nm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_title("Bootstrap uncertainty of 2D peak positions")
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_peak_position_cloud.png", transparent=True)
    plt.close(fig)


# MAIN

def main(cfg: Config = CFG) -> None:
    dirs = ensure_output_dirs(cfg.output_dir)
    prefix = parse_prefix_from_filename(cfg.excel_file)

    with open(dirs["root"] / "analysis_parameters.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\nInput trace: {cfg.excel_file}")
    print(f"Output directory: {dirs['root'].resolve()}")
    print(f"Bootstrap mode: {cfg.bootstrap_mode}, n={cfg.n_bootstraps}")

    pts = load_and_center_trace(cfg)
    print(f"Loaded {len(pts)} localizations after geometric recentering and cube filtering.")

    # Original/reference analysis
    print("\nComputing reference KDE slices and 2D peaks from the original trace...")
    ref_proj, ref_bw, ref_global_max = kde_xy_slices_zoom(pts, cfg)
    ref_peaks = detect_peaks_in_projections(ref_proj, ref_global_max, cfg, prefix)
    ref_fit = weighted_circle_fit(ref_peaks, cfg)

    ref_peaks.to_csv(dirs["tables"] / "reference_2d_peaks.csv", index=False)
    pd.DataFrame([{**{"prefix": prefix, "n_locs": len(pts), "kde_bandwidth_nm": ref_bw, "n_detected_peaks": len(ref_peaks)}, **ref_fit}]).to_csv(
        dirs["tables"] / "reference_circle_fit.csv", index=False
    )

    if cfg.make_plots and len(ref_peaks):
        plot_reference_peaks_and_circle(ref_peaks, ref_fit, cfg, dirs["figures"] / "reference_peaks_circle_fit.png")

    print(f"Reference bandwidth: {ref_bw:.3f} nm")
    print(f"Reference detected peaks: {len(ref_peaks)}")
    if ref_fit.get("fit_success"):
        print(
            "Reference weighted circle fit: "
            f"center=({ref_fit['circle_center_x_nm']:.2f}, {ref_fit['circle_center_y_nm']:.2f}) nm, "
            f"radius={ref_fit['circle_radius_nm']:.2f} nm, "
            f"diameter={ref_fit['circle_diameter_nm']:.2f} nm, "
            f"adj_R2={ref_fit['circle_adj_r2']:.3f}"
        )
    else:
        print(f"Reference circle fit failed: {ref_fit.get('fit_message')}")

    # Bootstrap loop
    rng = np.random.default_rng(cfg.random_seed)
    circle_rows: List[Dict] = []
    matched_rows: List[pd.DataFrame] = []
    raw_peak_rows: List[pd.DataFrame] = []

    print("\nRunning bootstrap resampling...")
    for b in tqdm(range(1, cfg.n_bootstraps + 1), unit="boot"):
        pts_b0 = bootstrap_points(pts, rng, cfg)
        pts_b = recenter_bootstrap_sample(pts_b0, cfg)
        if len(pts_b) < 10:
            circle_rows.append({
                "bootstrap_id": b,
                "n_locs": len(pts_b),
                "kde_bandwidth_nm": np.nan,
                "n_detected_peaks": 0,
                **weighted_circle_fit(pd.DataFrame(), cfg),
            })
            continue

        try:
            proj_b, bw_b, gmax_b = kde_xy_slices_zoom(pts_b, cfg)
            peaks_b = detect_peaks_in_projections(proj_b, gmax_b, cfg, prefix)
            fit_b = weighted_circle_fit(peaks_b, cfg)
            circle_rows.append({
                "bootstrap_id": b,
                "n_locs": int(len(pts_b)),
                "kde_bandwidth_nm": float(bw_b),
                "n_detected_peaks": int(len(peaks_b)),
                **fit_b,
            })
            if not peaks_b.empty:
                peaks_b = peaks_b.copy()
                peaks_b.insert(0, "bootstrap_id", b)
                raw_peak_rows.append(peaks_b)
                matched_rows.append(match_bootstrap_peaks_to_reference(peaks_b, ref_peaks, cfg, b))
        except Exception as exc:
            warnings.warn(f"Bootstrap {b} failed: {exc}")
            circle_rows.append({
                "bootstrap_id": b,
                "n_locs": int(len(pts_b)),
                "kde_bandwidth_nm": np.nan,
                "n_detected_peaks": 0,
                **{
                    "fit_success": False,
                    "fit_message": f"Bootstrap failed: {exc}",
                    "circle_center_x_nm": np.nan,
                    "circle_center_y_nm": np.nan,
                    "circle_radius_nm": np.nan,
                    "circle_diameter_nm": np.nan,
                    "circle_rmse_nm": np.nan,
                    "circle_weighted_rmse_nm": np.nan,
                    "circle_r2": np.nan,
                    "circle_adj_r2": np.nan,
                    "n_peaks_fit": 0,
                },
            })

    circle_df = pd.DataFrame(circle_rows)
    circle_df.to_csv(dirs["tables"] / "bootstrap_circle_metrics.csv", index=False)

    if raw_peak_rows:
        raw_boot_peaks_df = pd.concat(raw_peak_rows, ignore_index=True)
    else:
        raw_boot_peaks_df = pd.DataFrame()
    raw_boot_peaks_df.to_csv(dirs["tables"] / "bootstrap_all_detected_2d_peaks.csv", index=False)

    if matched_rows:
        matched_df = pd.concat(matched_rows, ignore_index=True)
    else:
        matched_df = pd.DataFrame()
    matched_df.to_csv(dirs["tables"] / "bootstrap_matched_peak_positions.csv", index=False)

    # Summaries and CIs
    circle_summary = summarize_circle_bootstrap(circle_df)
    circle_summary.to_csv(dirs["tables"] / "bootstrap_circle_CI_summary.csv", index=False)

    peak_summary = summarize_peak_bootstrap(matched_df, ref_peaks, cfg.n_bootstraps)
    peak_summary.to_csv(dirs["tables"] / "bootstrap_peak_position_CI_summary.csv", index=False)

    # One compact summary table that is easy to paste into a figure/table.
    ref_row = pd.read_csv(dirs["tables"] / "reference_circle_fit.csv").iloc[0].to_dict()
    key_metrics = circle_summary[circle_summary["metric"].isin([
        "circle_center_x_nm",
        "circle_center_y_nm",
        "circle_radius_nm",
        "circle_diameter_nm",
        "circle_adj_r2",
    ])].copy()
    key_metrics.insert(0, "reference_value", key_metrics["metric"].map(lambda m: ref_row.get(m, np.nan)))
    key_metrics.to_csv(dirs["tables"] / "figure_ready_circle_CI_summary.csv", index=False)

    if cfg.make_plots:
        plot_circle_bootstrap(circle_df, ref_fit, cfg, dirs["figures"])
        if not matched_df.empty and len(ref_peaks):
            plot_peak_bootstrap_cloud(matched_df, ref_peaks, ref_fit, cfg, dirs["figures"])

    # Console summary
    ok = circle_df[circle_df["fit_success"].astype(bool)]
    print("\nDone.")
    print(f"Successful circle fits: {len(ok)} / {cfg.n_bootstraps}")
    print("\nKey bootstrap CI summary:")
    print(key_metrics.to_string(index=False))
    print(f"\nSaved outputs to: {dirs['root'].resolve()}")
    print("Most useful files:")
    print("  tables/reference_2d_peaks.csv")
    print("  tables/reference_circle_fit.csv")
    print("  tables/bootstrap_circle_metrics.csv")
    print("  tables/bootstrap_circle_CI_summary.csv")
    print("  tables/bootstrap_peak_position_CI_summary.csv")
    print("  tables/figure_ready_circle_CI_summary.csv")
    print("  figures/reference_peaks_circle_fit.png")
    print("  figures/bootstrap_diameter_histogram.png")
    print("  figures/bootstrap_peak_position_cloud.png")


if __name__ == "__main__":
    main(CFG)
