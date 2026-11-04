# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""MINFLUX KDE null simulation.

Localization-noise-only Gaussian traces centred at the origin are analysed with
the manuscript pipeline (120 nm cube, global kNN bandwidth, Gaussian KDE, 2D
slice peak detection with sub-voxel refinement, density-weighted circle fit) to
test whether noise alone produces an XY annulus. Localization precision is set
by SIGMA_*_NM.
"""

import os
import gc
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

from sklearn.neighbors import NearestNeighbors, KernelDensity
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy import stats


# User parameters

OUTPUT_ROOT = "null_kde_simulation_outputs_sigma2"

# Null localization precision (GFP-Nup58-HaloTag, Table S2: 9.47, 9.06, 8.79 nm)
SIGMA_X_NM = 9.47
SIGMA_Y_NM = 9.06
SIGMA_Z_NM = 8.79
SIGMA_AVG_3D_NM = 9

N_POINTS_LIST = [1000, 2000, 3000]

N_REPLICATES = 1000
RANDOM_SEED = 20260521

# Match current KDE setup
VOX_NM = 1.0
GRID_MIN_NM = 0.0
GRID_MAX_NM = 120.0
CENTER_NM = 60.0
K_BAND = 3
KERNEL = "gaussian"
KDE_GRID_CHUNK_SIZE = 250_000

KDE_EVALUATION_METHOD = "binned_gaussian"

# Match current 3D peak-finding setup
PEAK_THRESHOLD_REL_3D = 0.75

# Match current 2D peak projection setup
NEW_2D_PROJ_CENTRES = list(range(0, 121, 10))
NEW_2D_PROJ_THICKNESS = 1.0
MIN_DIST_2D_PEAKS_VOX = 2
ZOOM_MIN_NM = 30.0
ZOOM_MAX_NM = 90.0
NORMALIZED_GLOBAL_NOISE_CUTOFF = 0.20
SLICE_RELATIVE_THRESHOLD_FRACTION = 0.50

ENABLE_SUBVOXEL_2D_REFINEMENT = True
SUBVOXEL_FINE_STEP_XY_NM = 0.1
SUBVOXEL_LOCAL_WINDOW_XY_NM = 2.0
SUBVOXEL_FINE_STEP_Z_NM = 0.25

FITTED_CIRCLE_N_SAMPLE_POINTS = 720
FITTED_ANNULUS_BAND_HALF_WIDTH_NM = 1.0
FITTED_CIRCLE_DENSITY_Z_MODE = "peak_slices"  # "peak_slices" or "all_slices"

ENABLE_SUBVOXEL_DOMINANT_3D_REFINEMENT = False
SUBVOXEL_FINE_STEP_3D_NM = 0.1
SUBVOXEL_LOCAL_WINDOW_3D_NM = 2.0

ANNULUS_RADIUS_MIN_NM = 6.0
ANNULUS_RADIUS_MAX_NM = 13.0
MIN_PEAKS_FOR_CIRCLE_FIT = 3

FALSE_ANNULUS_DIAMETER_MIN_NM = 12.0
FALSE_ANNULUS_DIAMETER_MAX_NM = 28.0
MIN_ANNULUS_TO_CENTER_RATIO_FOR_FALSE_ANNULUS = 1.0
MIN_ANNULUS_TO_CENTER_RATIO_STRICT = 1.2

# Output controls
SAVE_EXAMPLE_PNGS = True
SAVE_EXAMPLE_NPZ = True
SAVE_ALL_NPZ = False  # Usually keep False; KDE volumes are large.
SAVE_2D_PEAKS_CSV = True
COLOR_MAP = colormaps["jet"]
PNG_DPI = 300


# Data containers

@dataclass
class NullReplicateResult:
    n_requested: int
    replicate: int
    n_after_cube_filter: int
    sigma_x_nm: float
    sigma_y_nm: float
    sigma_z_nm: float
    sigma_avg_3d_nm: float
    k_band: int
    kde_bandwidth_nm: float
    global_max_density: float

    # 3D peak metrics are retained only as sanity checks.
    n_3d_peaks: int
    dominant_3d_peak_x_nm: float
    dominant_3d_peak_y_nm: float
    dominant_3d_peak_z_nm: float
    dominant_3d_peak_xy_radius_from_center_nm: float
    dominant_3d_peak_3d_distance_from_center_nm: float
    dominant_3d_peak_refined_x_nm: float
    dominant_3d_peak_refined_y_nm: float
    dominant_3d_peak_refined_z_nm: float
    dominant_3d_peak_refined_xy_radius_from_center_nm: float
    dominant_3d_peak_refined_3d_distance_from_center_nm: float

    # 2D XY annulus-relevant metrics.
    n_2d_peaks_total: int
    n_2d_slices_with_peaks: int
    dominant_2d_peak_x_nm: float
    dominant_2d_peak_y_nm: float
    dominant_2d_peak_z_center_nm: float
    dominant_2d_peak_radius_xy_nm: float
    dominant_2d_peak_refined_x_nm: float
    dominant_2d_peak_refined_y_nm: float
    dominant_2d_peak_refined_radius_xy_nm: float
    median_2d_peak_radius_xy_nm: float
    max_2d_peak_radius_xy_nm: float
    median_2d_peak_radius_xy_refined_nm: float
    max_2d_peak_radius_xy_refined_nm: float
    n_2d_peaks_in_annulus_radius_window: int
    n_2d_peaks_in_annulus_radius_window_refined: int

    # Circle fitting should be interpreted as a null false-annulus diagnostic.
    circle_fit_used: bool
    circle_fit_n_peaks: int
    circle_fit_center_x_nm: float
    circle_fit_center_y_nm: float
    circle_fit_radius_nm: float
    circle_fit_diameter_nm: float
    circle_fit_center_distance_from_true_center_nm: float
    circle_fit_radius_in_annulus_window: bool
    circle_fit_diameter_in_experimental_window: bool
    circle_fit_rmse_nm: float
    circle_fit_mae_nm: float
    circle_fit_normalized_rmse: float
    circle_fit_pseudo_r2: float
    circle_fit_adjusted_pseudo_r2: float

    circle_fit_refined_center_x_nm: float
    circle_fit_refined_center_y_nm: float
    circle_fit_refined_radius_nm: float
    circle_fit_refined_diameter_nm: float
    circle_fit_refined_center_distance_from_true_center_nm: float
    circle_fit_refined_radius_in_annulus_window: bool
    circle_fit_refined_diameter_in_experimental_window: bool
    circle_fit_refined_rmse_nm: float
    circle_fit_refined_mae_nm: float
    circle_fit_refined_normalized_rmse: float
    circle_fit_refined_pseudo_r2: float
    circle_fit_refined_adjusted_pseudo_r2: float

    # Simple central-depletion diagnostic for the central XY slice.
    center_slice_center_density_norm: float
    center_slice_max_density_norm: float
    center_slice_center_to_max_ratio: float
    center_slice_annulus_mean_density_norm: float
    center_slice_annulus_to_center_ratio: float

    center_slice_center_to_annulus_ratio: float
    center_slice_annulus_minus_center_density_norm: float
    annular_enrichment_index: float
    central_depletion_index: float

    fitted_circle_density_z_count: int
    circle_fit_refined_line_mean_density_norm: float
    circle_fit_refined_line_max_density_norm: float
    circle_fit_refined_annulus_band_mean_density_norm: float
    circle_fit_refined_annulus_band_max_density_norm: float
    circle_fit_refined_center_mean_density_norm: float
    circle_fit_refined_center_max_density_norm: float
    circle_fit_refined_line_mean_to_center_mean_ratio: float
    circle_fit_refined_line_max_to_center_mean_ratio: float
    circle_fit_refined_annulus_band_mean_to_center_mean_ratio: float
    circle_fit_refined_annulus_band_max_to_center_mean_ratio: float
    circle_fit_refined_line_mean_to_center_max_ratio: float
    circle_fit_refined_line_max_to_center_max_ratio: float
    circle_fit_refined_annulus_band_mean_to_center_max_ratio: float
    circle_fit_refined_annulus_band_max_to_center_max_ratio: float

    center_depleted_binary: bool
    false_annulus_call: bool
    false_annulus_call_strict: bool


# Core functions matching the current workflow

def build_grid(vox_nm: float = VOX_NM) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the 0-120 nm grid in the same Z, Y, X matrix order used in the pipeline."""
    vals = np.arange(GRID_MIN_NM, GRID_MAX_NM + vox_nm / 2, vox_nm, dtype=np.float32)
    Zg, Yg, Xg = np.meshgrid(vals, vals, vals, indexing="ij")
    grid_pts = np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T.astype(np.float32)
    return vals, Xg, Yg, Zg, grid_pts


def simulate_null_trace(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a localization-noise-only trace centered at (0,0,0), then apply manuscript centering."""
    pts_raw = rng.normal(
        loc=(0.0, 0.0, 0.0),
        scale=(SIGMA_X_NM, SIGMA_Y_NM, SIGMA_Z_NM),
        size=(n_points, 3),
    ).astype(np.float32)

    pts = pts_raw - pts_raw.mean(axis=0)
    pts += CENTER_NM

    # Match current cube filtering.
    inside = np.all((pts >= GRID_MIN_NM) & (pts <= GRID_MAX_NM), axis=1)
    pts = pts[inside]
    return pts.astype(np.float32)


def calculate_global_kth_nn_bandwidth(pts: np.ndarray, k_band: int = K_BAND) -> float:
    """Current approach: one global bandwidth = median k-th NN distance."""
    nn = NearestNeighbors(n_neighbors=k_band).fit(pts)
    distances, _ = nn.kneighbors(pts)
    bw = float(np.median(distances[:, k_band - 1]))
    if bw == 0 or np.isnan(bw) or bw < 1e-6:
        bw = 0.1 * VOX_NM
    return bw


def kde_density_on_grid_exact_sklearn(
    pts: np.ndarray,
    grid_pts: np.ndarray,
    vals: np.ndarray,
    bandwidth_nm: float,
    vox_nm: float = VOX_NM,
    chunk_size: int = KDE_GRID_CHUNK_SIZE,
) -> np.ndarray:
    """Literal sklearn KDE evaluation. Accurate but slow for many null replicates."""
    kde = KernelDensity(
        kernel=KERNEL,
        bandwidth=bandwidth_nm,
        algorithm="kd_tree",
        breadth_first=True,
        rtol=1e-4,
    ).fit(pts)

    density_flat = np.empty(grid_pts.shape[0], dtype=np.float32)
    for start in range(0, grid_pts.shape[0], chunk_size):
        end = min(start + chunk_size, grid_pts.shape[0])
        log_density = kde.score_samples(grid_pts[start:end])
        density_flat[start:end] = np.exp(log_density).astype(np.float32)

    n_vox = len(vals)
    density = density_flat.reshape((n_vox, n_vox, n_vox))
    density = density * len(pts) / (vox_nm ** 3)  # loc/nm^3, matching current script
    return density.astype(np.float32)


def kde_density_on_grid_binned_gaussian(
    pts: np.ndarray,
    vals: np.ndarray,
    bandwidth_nm: float,
    vox_nm: float = VOX_NM,
) -> np.ndarray:
    """Fast grid-based Gaussian KDE."""
    n_vox = len(vals)

    # Convert X,Y,Z continuous positions to nearest Z,Y,X voxel indices.
    ijk = np.rint((pts - GRID_MIN_NM) / vox_nm).astype(int)
    ijk = np.clip(ijk, 0, n_vox - 1)

    counts = np.zeros((n_vox, n_vox, n_vox), dtype=np.float32)  # Z,Y,X
    np.add.at(counts, (ijk[:, 2], ijk[:, 1], ijk[:, 0]), 1.0)

    sigma_vox = max(float(bandwidth_nm / vox_nm), 1e-6)
    density = gaussian_filter(
        counts,
        sigma=sigma_vox,
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )

    density = density / (vox_nm ** 3)
    return density.astype(np.float32)


def kde_density_on_grid(
    pts: np.ndarray,
    grid_pts: np.ndarray,
    vals: np.ndarray,
    bandwidth_nm: float,
    vox_nm: float = VOX_NM,
    chunk_size: int = KDE_GRID_CHUNK_SIZE,
) -> np.ndarray:
    if KDE_EVALUATION_METHOD == "binned_gaussian":
        return kde_density_on_grid_binned_gaussian(pts, vals, bandwidth_nm, vox_nm)
    if KDE_EVALUATION_METHOD == "exact_sklearn":
        return kde_density_on_grid_exact_sklearn(pts, grid_pts, vals, bandwidth_nm, vox_nm, chunk_size)
    raise ValueError(f"Unknown KDE_EVALUATION_METHOD: {KDE_EVALUATION_METHOD}")


def normalize_density(density: np.ndarray) -> Tuple[np.ndarray, float]:
    global_max = float(np.max(density))
    if global_max > 0:
        return (density / global_max).astype(np.float32), global_max
    return density.astype(np.float32), global_max


def find_3d_peaks_from_density(
    density_matrix_3d: np.ndarray,
    grid_coords_1d: np.ndarray,
    current_grid_vox_size: float,
    min_dist_physical_nm_val: float,
    threshold_rel_val_param: float,
) -> List[Tuple[float, float, float, float]]:
    """Find 3D peaks. Returns x, y, z, density_at_peak."""
    min_dist_vox = max(1, int(round(min_dist_physical_nm_val / current_grid_vox_size)))
    peak_indices = peak_local_max(
        density_matrix_3d,
        min_distance=min_dist_vox,
        threshold_rel=threshold_rel_val_param,
        exclude_border=False,
    )

    peaks = []
    for z_idx, y_idx, x_idx in peak_indices:
        peaks.append((
            float(grid_coords_1d[x_idx]),
            float(grid_coords_1d[y_idx]),
            float(grid_coords_1d[z_idx]),
            float(density_matrix_3d[z_idx, y_idx, x_idx]),
        ))

    peaks.sort(key=lambda row: row[3], reverse=True)
    return peaks


def proj_xy(density_map: np.ndarray, z_center_nm: float, thick_nm: float, current_grid_vox_size: float) -> np.ndarray:
    """Match current XY projection: average voxel densities through a Z slab."""
    half_thick = thick_nm / 2
    lo_idx = int(max(GRID_MIN_NM, z_center_nm - half_thick) / current_grid_vox_size)
    hi_idx = int(min(GRID_MAX_NM, z_center_nm + half_thick) / current_grid_vox_size)
    lo_idx = max(0, min(lo_idx, density_map.shape[0] - 1))
    hi_idx = max(0, min(hi_idx, density_map.shape[0] - 1))
    if lo_idx > hi_idx:
        return np.zeros((density_map.shape[1], density_map.shape[2]), dtype=density_map.dtype)
    return density_map[lo_idx:hi_idx + 1].mean(axis=0)


# Continuous local KDE refinement

def _make_local_axis(center: float, window: float, step: float) -> np.ndarray:
    lo = max(GRID_MIN_NM, center - window)
    hi = min(GRID_MAX_NM, center + window)
    return np.arange(lo, hi + step / 2, step, dtype=np.float32)


def continuous_projected_kde_xy_local(
    pts: np.ndarray,
    bandwidth_nm: float,
    x0_nm: float,
    y0_nm: float,
    z_center_nm: float,
    slab_thickness_nm: float,
    xy_window_nm: float = SUBVOXEL_LOCAL_WINDOW_XY_NM,
    xy_step_nm: float = SUBVOXEL_FINE_STEP_XY_NM,
    z_step_nm: float = SUBVOXEL_FINE_STEP_Z_NM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the same continuous isotropic Gaussian KDE on a local fine XY grid."""
    h = float(max(bandwidth_nm, 1e-6))
    x_fine = _make_local_axis(x0_nm, xy_window_nm, xy_step_nm)
    y_fine = _make_local_axis(y0_nm, xy_window_nm, xy_step_nm)

    half = slab_thickness_nm / 2
    z_lo = max(GRID_MIN_NM, z_center_nm - half)
    z_hi = min(GRID_MAX_NM, z_center_nm + half)
    if z_hi < z_lo:
        z_vals = np.array([z_center_nm], dtype=np.float32)
    else:
        z_vals = np.arange(z_lo, z_hi + z_step_nm / 2, z_step_nm, dtype=np.float32)
        if z_vals.size == 0:
            z_vals = np.array([z_center_nm], dtype=np.float32)

    wx = np.exp(-0.5 * ((x_fine[:, None] - pts[None, :, 0]) / h) ** 2).astype(np.float32)  # nx, n
    wy = np.exp(-0.5 * ((y_fine[:, None] - pts[None, :, 1]) / h) ** 2).astype(np.float32)  # ny, n
    wz = np.exp(-0.5 * ((z_vals[:, None] - pts[None, :, 2]) / h) ** 2).astype(np.float32)   # nz, n
    wz_mean = wz.mean(axis=0).astype(np.float32)                                             # n,

    density_yx = (wy * wz_mean[None, :]) @ wx.T  # ny, nx
    return x_fine, y_fine, density_yx.astype(np.float32)


def refine_2d_peak_subvoxel(
    pts: np.ndarray,
    bandwidth_nm: float,
    coarse_x_nm: float,
    coarse_y_nm: float,
    z_center_nm: float,
) -> Tuple[float, float, float, float]:
    """Refine a coarse 2D peak by local continuous KDE evaluation."""
    x_fine, y_fine, dens = continuous_projected_kde_xy_local(
        pts=pts,
        bandwidth_nm=bandwidth_nm,
        x0_nm=coarse_x_nm,
        y0_nm=coarse_y_nm,
        z_center_nm=z_center_nm,
        slab_thickness_nm=NEW_2D_PROJ_THICKNESS,
    )
    flat_idx = int(np.argmax(dens))
    y_idx, x_idx = np.unravel_index(flat_idx, dens.shape)
    x_ref = float(x_fine[x_idx])
    y_ref = float(y_fine[y_idx])
    local_density = float(dens[y_idx, x_idx])
    radius_ref = math.sqrt((x_ref - CENTER_NM) ** 2 + (y_ref - CENTER_NM) ** 2)
    return x_ref, y_ref, radius_ref, local_density


def continuous_kde_3d_local_max(
    pts: np.ndarray,
    bandwidth_nm: float,
    x0_nm: float,
    y0_nm: float,
    z0_nm: float,
    window_nm: float = SUBVOXEL_LOCAL_WINDOW_3D_NM,
    step_nm: float = SUBVOXEL_FINE_STEP_3D_NM,
) -> Tuple[float, float, float, float]:
    """Optional local refinement of a 3D peak. Used only as a sanity check."""
    h = float(max(bandwidth_nm, 1e-6))
    x_fine = _make_local_axis(x0_nm, window_nm, step_nm)
    y_fine = _make_local_axis(y0_nm, window_nm, step_nm)
    z_fine = _make_local_axis(z0_nm, window_nm, step_nm)

    wx = np.exp(-0.5 * ((x_fine[:, None] - pts[None, :, 0]) / h) ** 2).astype(np.float32)
    wy = np.exp(-0.5 * ((y_fine[:, None] - pts[None, :, 1]) / h) ** 2).astype(np.float32)

    best_density = -np.inf
    best_x = float(x0_nm)
    best_y = float(y0_nm)
    best_z = float(z0_nm)

    # Loop over z to avoid creating a full 3D density array in memory.
    for z in z_fine:
        wz = np.exp(-0.5 * ((float(z) - pts[:, 2]) / h) ** 2).astype(np.float32)
        dens_yx = (wy * wz[None, :]) @ wx.T
        idx = int(np.argmax(dens_yx))
        y_idx, x_idx = np.unravel_index(idx, dens_yx.shape)
        val = float(dens_yx[y_idx, x_idx])
        if val > best_density:
            best_density = val
            best_x = float(x_fine[x_idx])
            best_y = float(y_fine[y_idx])
            best_z = float(z)

    return best_x, best_y, best_z, float(best_density)


# 2D peak detection, circle fitting, central depletion diagnostics

def find_2d_slice_peaks(
    density_normalized_3d: np.ndarray,
    vals: np.ndarray,
    pts: np.ndarray,
    bandwidth_nm: float,
    n_requested: int,
    replicate: int,
) -> pd.DataFrame:
    """Run the same zoomed 2D slice peak detection currently used for single traces."""
    records = []
    zoom_indices = np.where((vals >= ZOOM_MIN_NM) & (vals <= ZOOM_MAX_NM))[0]
    if len(zoom_indices) == 0:
        return pd.DataFrame(records)

    start_idx, end_idx = int(zoom_indices[0]), int(zoom_indices[-1])
    zoomed_vals = vals[start_idx:end_idx + 1]

    for center_z in NEW_2D_PROJ_CENTRES:
        normalized_proj = proj_xy(density_normalized_3d, center_z, NEW_2D_PROJ_THICKNESS, VOX_NM)
        zoomed_norm = normalized_proj[start_idx:end_idx + 1, start_idx:end_idx + 1]

        if zoomed_norm.size == 0:
            continue

        slice_max = float(np.max(zoomed_norm))
        threshold_slice_relative = slice_max * SLICE_RELATIVE_THRESHOLD_FRACTION
        threshold_abs = max(threshold_slice_relative, NORMALIZED_GLOBAL_NOISE_CUTOFF)

        peaks_pixels = peak_local_max(
            zoomed_norm,
            min_distance=MIN_DIST_2D_PEAKS_VOX,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )

        for peak_rank, (r_idx, c_idx) in enumerate(peaks_pixels, start=1):
            x_nm = float(zoomed_vals[c_idx])
            y_nm = float(zoomed_vals[r_idx])
            norm_density = float(zoomed_norm[r_idx, c_idx])
            radius = math.sqrt((x_nm - CENTER_NM) ** 2 + (y_nm - CENTER_NM) ** 2)

            x_ref = y_ref = radius_ref = local_refined_density = np.nan
            if ENABLE_SUBVOXEL_2D_REFINEMENT:
                x_ref, y_ref, radius_ref, local_refined_density = refine_2d_peak_subvoxel(
                    pts=pts,
                    bandwidth_nm=bandwidth_nm,
                    coarse_x_nm=x_nm,
                    coarse_y_nm=y_nm,
                    z_center_nm=float(center_z),
                )

            records.append({
                "n_requested": int(n_requested),
                "replicate": int(replicate),
                "Z_Center_nm": float(center_z),
                "Thickness_nm": float(NEW_2D_PROJ_THICKNESS),
                "Peak_Rank_Within_Slice": int(peak_rank),
                "X_nm": x_nm,
                "Y_nm": y_nm,
                "Radius_XY_From_True_Center_nm": radius,
                "X_refined_nm": x_ref,
                "Y_refined_nm": y_ref,
                "Radius_XY_Refined_From_True_Center_nm": radius_ref,
                "Normalized_Density_at_Peak": norm_density,
                "Local_Continuous_KDE_Density_Refined": local_refined_density,
                "Threshold_Normalized_Used": float(threshold_abs),
                "In_Annulus_Radius_Window": ANNULUS_RADIUS_MIN_NM <= radius <= ANNULUS_RADIUS_MAX_NM,
                "In_Annulus_Radius_Window_Refined": ANNULUS_RADIUS_MIN_NM <= radius_ref <= ANNULUS_RADIUS_MAX_NM if not np.isnan(radius_ref) else False,
            })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.sort_values(["Z_Center_nm", "Normalized_Density_at_Peak"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def weighted_circle_fit_xy(x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Weighted algebraic circle fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(x, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 1e-12, None)

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    sw = np.sqrt(weights)
    Aw = A * sw[:, None]
    bw = b * sw
    D, E, F = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    cx = -D / 2
    cy = -E / 2
    r2 = (D ** 2 + E ** 2) / 4 - F
    radius = math.sqrt(max(r2, 0.0))
    return float(cx), float(cy), float(radius)


def circle_fit_quality_metrics(
    x: np.ndarray,
    y: np.ndarray,
    center_x_nm: float,
    center_y_nm: float,
    radius_nm: float,
    weights: Optional[np.ndarray] = None,
    n_parameters: int = 3,
) -> dict:
    """Calculate circle-fit goodness metrics from radial residuals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0 or not np.isfinite(center_x_nm) or not np.isfinite(center_y_nm) or not np.isfinite(radius_nm):
        return {
            "rmse_nm": np.nan,
            "mae_nm": np.nan,
            "normalized_rmse": np.nan,
            "pseudo_r2": np.nan,
            "adjusted_pseudo_r2": np.nan,
        }

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0.0, None)
        if np.sum(w) <= 0:
            w = np.ones(n, dtype=float)

    d = np.sqrt((x - center_x_nm) ** 2 + (y - center_y_nm) ** 2)
    residual = d - radius_nm
    rss = float(np.sum(w * residual ** 2))
    tss0 = float(np.sum(w * d ** 2))
    mean_abs = float(np.sum(w * np.abs(residual)) / np.sum(w))
    rmse = math.sqrt(rss / float(np.sum(w))) if np.sum(w) > 0 else np.nan
    normalized_rmse = rmse / radius_nm if radius_nm > 0 and np.isfinite(rmse) else np.nan

    if tss0 > 0:
        pseudo_r2 = 1.0 - rss / tss0
    else:
        pseudo_r2 = np.nan

    if np.isfinite(pseudo_r2) and n > (n_parameters + 1):
        adjusted_pseudo_r2 = 1.0 - (1.0 - pseudo_r2) * (n - 1) / (n - n_parameters - 1)
    else:
        adjusted_pseudo_r2 = np.nan

    return {
        "rmse_nm": float(rmse),
        "mae_nm": mean_abs,
        "normalized_rmse": float(normalized_rmse),
        "pseudo_r2": float(pseudo_r2),
        "adjusted_pseudo_r2": float(adjusted_pseudo_r2) if np.isfinite(adjusted_pseudo_r2) else np.nan,
    }


def central_slice_depletion_metrics(density_normalized_3d: np.ndarray, vals: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Measure central enrichment/depletion in the center XY slice."""
    center_proj = proj_xy(density_normalized_3d, CENTER_NM, NEW_2D_PROJ_THICKNESS, VOX_NM)
    center_idx = int(np.argmin(np.abs(vals - CENTER_NM)))
    center_val = float(center_proj[center_idx, center_idx])
    max_val = float(np.max(center_proj))
    center_to_max = center_val / max_val if max_val > 0 else np.nan

    X2, Y2 = np.meshgrid(vals, vals, indexing="xy")
    R = np.sqrt((X2 - CENTER_NM) ** 2 + (Y2 - CENTER_NM) ** 2)
    annulus_mask = (R >= ANNULUS_RADIUS_MIN_NM) & (R <= ANNULUS_RADIUS_MAX_NM)
    annulus_mean = float(np.mean(center_proj[annulus_mask])) if np.any(annulus_mask) else np.nan

    annulus_to_center = annulus_mean / center_val if center_val > 0 else np.nan
    center_to_annulus = center_val / annulus_mean if annulus_mean > 0 else np.nan
    annulus_minus_center = annulus_mean - center_val if np.isfinite(annulus_mean) and np.isfinite(center_val) else np.nan

    annular_enrichment_index = max(0.0, annulus_to_center - 1.0) if np.isfinite(annulus_to_center) else np.nan
    central_depletion_index = max(0.0, 1.0 - center_to_annulus) if np.isfinite(center_to_annulus) else np.nan

    return (
        center_val,
        max_val,
        center_to_max,
        annulus_mean,
        annulus_to_center,
        center_to_annulus,
        annulus_minus_center,
        annular_enrichment_index,
        central_depletion_index,
    )


def bilinear_sample_2d(matrix: np.ndarray, vals: np.ndarray, x_nm: np.ndarray, y_nm: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2D XY matrix defined on vals x vals coordinates."""
    x = np.asarray(x_nm, dtype=float)
    y = np.asarray(y_nm, dtype=float)
    out = np.full(np.broadcast_shapes(x.shape, y.shape), np.nan, dtype=float)
    xb = np.broadcast_to(x, out.shape).ravel()
    yb = np.broadcast_to(y, out.shape).ravel()
    flat_out = out.ravel()

    step = float(vals[1] - vals[0]) if len(vals) > 1 else 1.0
    fx = (xb - float(vals[0])) / step
    fy = (yb - float(vals[0])) / step

    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < matrix.shape[1]) & (y1 < matrix.shape[0])
    if not np.any(valid):
        return out

    dx = fx[valid] - x0[valid]
    dy = fy[valid] - y0[valid]
    m = matrix
    v00 = m[y0[valid], x0[valid]]
    v10 = m[y0[valid], x1[valid]]
    v01 = m[y1[valid], x0[valid]]
    v11 = m[y1[valid], x1[valid]]
    flat_out[valid] = (
        v00 * (1 - dx) * (1 - dy)
        + v10 * dx * (1 - dy)
        + v01 * (1 - dx) * dy
        + v11 * dx * dy
    )
    return out


def fitted_circle_density_metrics(
    density_normalized_3d: np.ndarray,
    vals: np.ndarray,
    center_x_nm: float,
    center_y_nm: float,
    radius_nm: float,
    z_centers_nm: Optional[np.ndarray] = None,
    slab_thickness_nm: float = NEW_2D_PROJ_THICKNESS,
    n_circle_points: int = FITTED_CIRCLE_N_SAMPLE_POINTS,
    band_half_width_nm: float = FITTED_ANNULUS_BAND_HALF_WIDTH_NM,
) -> dict:
    """Calculate density ratios using the fitted circle center/radius."""
    keys = [
        "fitted_circle_density_z_count",
        "circle_fit_refined_line_mean_density_norm",
        "circle_fit_refined_line_max_density_norm",
        "circle_fit_refined_annulus_band_mean_density_norm",
        "circle_fit_refined_annulus_band_max_density_norm",
        "circle_fit_refined_center_mean_density_norm",
        "circle_fit_refined_center_max_density_norm",
        "circle_fit_refined_line_mean_to_center_mean_ratio",
        "circle_fit_refined_line_max_to_center_mean_ratio",
        "circle_fit_refined_annulus_band_mean_to_center_mean_ratio",
        "circle_fit_refined_annulus_band_max_to_center_mean_ratio",
        "circle_fit_refined_line_mean_to_center_max_ratio",
        "circle_fit_refined_line_max_to_center_max_ratio",
        "circle_fit_refined_annulus_band_mean_to_center_max_ratio",
        "circle_fit_refined_annulus_band_max_to_center_max_ratio",
    ]
    nan_out = {k: np.nan for k in keys}
    nan_out["fitted_circle_density_z_count"] = 0

    if not (np.isfinite(center_x_nm) and np.isfinite(center_y_nm) and np.isfinite(radius_nm) and radius_nm > 0):
        return nan_out

    if z_centers_nm is None or len(z_centers_nm) == 0:
        z_centers = np.array(NEW_2D_PROJ_CENTRES, dtype=float)
    else:
        z_centers = np.array(sorted(set(np.asarray(z_centers_nm, dtype=float))), dtype=float)

    theta = np.linspace(0, 2 * np.pi, int(n_circle_points), endpoint=False)
    circle_x = center_x_nm + radius_nm * np.cos(theta)
    circle_y = center_y_nm + radius_nm * np.sin(theta)

    X2, Y2 = np.meshgrid(vals, vals, indexing="xy")
    radial_from_fit_center = np.sqrt((X2 - center_x_nm) ** 2 + (Y2 - center_y_nm) ** 2)
    band_mask = (radial_from_fit_center >= max(0.0, radius_nm - band_half_width_nm)) & (radial_from_fit_center <= radius_nm + band_half_width_nm)

    center_values = []
    line_values = []
    band_values = []

    for zc in z_centers:
        proj = proj_xy(density_normalized_3d, float(zc), slab_thickness_nm, VOX_NM)
        center_sample = bilinear_sample_2d(proj, vals, np.array([center_x_nm]), np.array([center_y_nm]))[0]
        if np.isfinite(center_sample):
            center_values.append(float(center_sample))

        line_sample = bilinear_sample_2d(proj, vals, circle_x, circle_y)
        line_sample = line_sample[np.isfinite(line_sample)]
        if line_sample.size > 0:
            line_values.extend(line_sample.astype(float).tolist())

        if np.any(band_mask):
            band_sample = proj[band_mask]
            band_sample = band_sample[np.isfinite(band_sample)]
            if band_sample.size > 0:
                band_values.extend(band_sample.astype(float).tolist())

    center_values = np.asarray(center_values, dtype=float)
    line_values = np.asarray(line_values, dtype=float)
    band_values = np.asarray(band_values, dtype=float)

    center_mean = float(np.mean(center_values)) if center_values.size > 0 else np.nan
    center_max = float(np.max(center_values)) if center_values.size > 0 else np.nan
    line_mean = float(np.mean(line_values)) if line_values.size > 0 else np.nan
    line_max = float(np.max(line_values)) if line_values.size > 0 else np.nan
    band_mean = float(np.mean(band_values)) if band_values.size > 0 else np.nan
    band_max = float(np.max(band_values)) if band_values.size > 0 else np.nan

    def safe_ratio(num, den):
        return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan

    return {
        "fitted_circle_density_z_count": int(len(z_centers)),
        "circle_fit_refined_line_mean_density_norm": line_mean,
        "circle_fit_refined_line_max_density_norm": line_max,
        "circle_fit_refined_annulus_band_mean_density_norm": band_mean,
        "circle_fit_refined_annulus_band_max_density_norm": band_max,
        "circle_fit_refined_center_mean_density_norm": center_mean,
        "circle_fit_refined_center_max_density_norm": center_max,
        "circle_fit_refined_line_mean_to_center_mean_ratio": safe_ratio(line_mean, center_mean),
        "circle_fit_refined_line_max_to_center_mean_ratio": safe_ratio(line_max, center_mean),
        "circle_fit_refined_annulus_band_mean_to_center_mean_ratio": safe_ratio(band_mean, center_mean),
        "circle_fit_refined_annulus_band_max_to_center_mean_ratio": safe_ratio(band_max, center_mean),
        "circle_fit_refined_line_mean_to_center_max_ratio": safe_ratio(line_mean, center_max),
        "circle_fit_refined_line_max_to_center_max_ratio": safe_ratio(line_max, center_max),
        "circle_fit_refined_annulus_band_mean_to_center_max_ratio": safe_ratio(band_mean, center_max),
        "circle_fit_refined_annulus_band_max_to_center_max_ratio": safe_ratio(band_max, center_max),
    }


# Plotting and summarization

def save_example_center_slice(
    density_normalized_3d: np.ndarray,
    vals: np.ndarray,
    peaks_2d_df: pd.DataFrame,
    n_points: int,
    replicate: int,
    output_dir: str,
) -> None:
    matrix = proj_xy(density_normalized_3d, CENTER_NM, NEW_2D_PROJ_THICKNESS, VOX_NM)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=PNG_DPI)
    im = ax.imshow(
        matrix,
        origin="lower",
        cmap=COLOR_MAP,
        vmin=0,
        vmax=1,
        extent=[GRID_MIN_NM, GRID_MAX_NM, GRID_MIN_NM, GRID_MAX_NM],
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, shrink=0.75, label="Normalized density")
    ax.scatter([CENTER_NM], [CENTER_NM], marker="+", s=80, linewidths=1.5, color="white", label="True center")

    # Mark peaks detected in the center slice only.
    if len(peaks_2d_df) > 0:
        center_peaks = peaks_2d_df[np.isclose(peaks_2d_df["Z_Center_nm"], CENTER_NM)]
        if len(center_peaks) > 0:
            ax.scatter(center_peaks["X_nm"], center_peaks["Y_nm"], marker="x", s=35, color="white", linewidths=1.0, label="Coarse peaks")
            if ENABLE_SUBVOXEL_2D_REFINEMENT:
                ax.scatter(center_peaks["X_refined_nm"], center_peaks["Y_refined_nm"], marker="o", s=18, facecolors="none", edgecolors="black", linewidths=0.8, label="Refined peaks")

    ax.set_title(f"Null KDE center slice\nN={n_points}, replicate={replicate}")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_xlim(30, 90)
    ax.set_ylim(30, 90)
    ax.set_xticks([30, 45, 60, 75, 90])
    ax.set_yticks([30, 45, 60, 75, 90])
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"null_N{n_points}_rep{replicate:03d}_Z060_center_slice.png"), transparent=True)
    plt.close(fig)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in df.groupby("n_requested"):
        row = {"n_requested": n, "n_replicates": len(g)}
        for col in [
            "kde_bandwidth_nm",
            "n_3d_peaks",
            "dominant_3d_peak_xy_radius_from_center_nm",
            "dominant_3d_peak_3d_distance_from_center_nm",
            "n_2d_peaks_total",
            "n_2d_slices_with_peaks",
            "dominant_2d_peak_radius_xy_nm",
            "dominant_2d_peak_refined_radius_xy_nm",
            "median_2d_peak_radius_xy_nm",
            "median_2d_peak_radius_xy_refined_nm",
            "max_2d_peak_radius_xy_nm",
            "max_2d_peak_radius_xy_refined_nm",
            "n_2d_peaks_in_annulus_radius_window_refined",
            "circle_fit_refined_radius_nm",
            "circle_fit_refined_diameter_nm",
            "circle_fit_refined_center_distance_from_true_center_nm",
            "circle_fit_refined_rmse_nm",
            "circle_fit_refined_mae_nm",
            "circle_fit_refined_normalized_rmse",
            "circle_fit_refined_pseudo_r2",
            "circle_fit_refined_adjusted_pseudo_r2",
            "circle_fit_rmse_nm",
            "circle_fit_mae_nm",
            "circle_fit_normalized_rmse",
            "circle_fit_pseudo_r2",
            "circle_fit_adjusted_pseudo_r2",
            "center_slice_center_to_max_ratio",
            "circle_fit_refined_line_mean_to_center_mean_ratio",
            "circle_fit_refined_line_max_to_center_mean_ratio",
            "circle_fit_refined_annulus_band_mean_to_center_mean_ratio",
            "circle_fit_refined_annulus_band_max_to_center_mean_ratio",
            "center_slice_annulus_to_center_ratio",
            "center_slice_center_to_annulus_ratio",
            "center_slice_annulus_minus_center_density_norm",
            "annular_enrichment_index",
            "central_depletion_index",
        ]:
            vals_col = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals_col) == 0:
                row[f"{col}_median"] = np.nan
                row[f"{col}_q025"] = np.nan
                row[f"{col}_q975"] = np.nan
                row[f"{col}_ci95_low"] = np.nan
                row[f"{col}_ci95_high"] = np.nan
                row[f"{col}_mean"] = np.nan
                row[f"{col}_mean_ci95_low"] = np.nan
                row[f"{col}_mean_ci95_high"] = np.nan
            else:
                row[f"{col}_median"] = float(vals_col.median())
                row[f"{col}_q025"] = float(vals_col.quantile(0.025))
                row[f"{col}_q975"] = float(vals_col.quantile(0.975))
                row[f"{col}_ci95_low"] = float(vals_col.quantile(0.025))
                row[f"{col}_ci95_high"] = float(vals_col.quantile(0.975))
                row[f"{col}_mean"] = float(vals_col.mean())
                if len(vals_col) > 1:
                    sem = float(stats.sem(vals_col, nan_policy="omit"))
                    tcrit = float(stats.t.ppf(0.975, len(vals_col) - 1))
                    row[f"{col}_mean_ci95_low"] = float(vals_col.mean() - tcrit * sem)
                    row[f"{col}_mean_ci95_high"] = float(vals_col.mean() + tcrit * sem)
                else:
                    row[f"{col}_mean_ci95_low"] = np.nan
                    row[f"{col}_mean_ci95_high"] = np.nan

        row["frac_replicates_with_more_than_one_3d_peak"] = float((g["n_3d_peaks"] > 1).mean())
        row["frac_replicates_with_any_refined_2d_peak_in_annulus_radius_window"] = float((g["n_2d_peaks_in_annulus_radius_window_refined"] > 0).mean())
        row["frac_replicates_with_refined_circle_fit_radius_in_annulus_window"] = float((g["circle_fit_refined_radius_in_annulus_window"] == True).mean())
        row["frac_replicates_with_refined_circle_fit_diameter_in_18_19_nm_window"] = float((g["circle_fit_refined_diameter_in_experimental_window"] == True).mean())
        row["frac_replicates_with_center_depletion_annulus_gt_center"] = float((g["center_depleted_binary"] == True).mean())
        row["frac_replicates_false_annulus_call"] = float((g["false_annulus_call"] == True).mean())
        row["frac_replicates_false_annulus_call_strict"] = float((g["false_annulus_call_strict"] == True).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_requested")


def make_formatted_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table with median and empirical 95% interval."""
    selected_metrics = [
        ("kde_bandwidth_nm", "KDE bandwidth (nm)"),
        ("n_3d_peaks", "3D peaks"),
        ("n_2d_peaks_total", "2D peaks total"),
        ("circle_fit_refined_diameter_nm", "Refined circle-fit diameter (nm)"),
        ("circle_fit_refined_adjusted_pseudo_r2", "Refined circle-fit adjusted pseudo-R²"),
        ("circle_fit_refined_normalized_rmse", "Refined circle-fit normalized RMSE"),
        ("circle_fit_refined_line_mean_to_center_mean_ratio", "Fitted-circle line mean/center mean density ratio"),
        ("circle_fit_refined_line_max_to_center_mean_ratio", "Fitted-circle line max/center mean density ratio"),
        ("circle_fit_refined_annulus_band_mean_to_center_mean_ratio", "Fitted-annulus band mean/center mean density ratio"),
        ("circle_fit_refined_annulus_band_max_to_center_mean_ratio", "Fitted-annulus band max/center mean density ratio"),
        ("center_slice_annulus_to_center_ratio", "Fixed-center-slice annulus/center density ratio (QC only)"),
        ("annular_enrichment_index", "Annular enrichment index"),
        ("central_depletion_index", "Central depletion index"),
    ]
    rows = []
    for _, row in summary_df.iterrows():
        out = {
            "n_requested": row.get("n_requested"),
            "n_replicates": row.get("n_replicates"),
        }
        for metric, label in selected_metrics:
            med = row.get(f"{metric}_median", np.nan)
            lo = row.get(f"{metric}_q025", np.nan)
            hi = row.get(f"{metric}_q975", np.nan)
            if np.isfinite(med) and np.isfinite(lo) and np.isfinite(hi):
                out[label] = f"{med:.4g} ({lo:.4g}-{hi:.4g})"
            else:
                out[label] = "NA"
        out["False-annulus call fraction"] = row.get("frac_replicates_false_annulus_call", np.nan)
        out["Strict false-annulus call fraction"] = row.get("frac_replicates_false_annulus_call_strict", np.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def save_summary_plots(df: pd.DataFrame, output_dir: str) -> None:
    plot_specs = [
        ("kde_bandwidth_nm", "KDE bandwidth (nm)", "null_kde_bandwidth_by_n.png"),
        ("n_3d_peaks", "Number of 3D peaks", "null_n_3d_peaks_by_n.png"),
        ("dominant_2d_peak_refined_radius_xy_nm", "Dominant 2D peak refined XY radius (nm)", "null_dominant_2d_refined_radius_by_n.png"),
        ("median_2d_peak_radius_xy_refined_nm", "Median refined 2D peak XY radius (nm)", "null_median_2d_refined_radius_by_n.png"),
        ("circle_fit_refined_diameter_nm", "Refined circle-fit diameter (nm)", "null_refined_circle_fit_diameter_by_n.png"),
        ("circle_fit_refined_adjusted_pseudo_r2", "Refined circle-fit adjusted pseudo-R²", "null_refined_circle_fit_adjusted_pseudo_r2_by_n.png"),
        ("circle_fit_refined_normalized_rmse", "Refined circle-fit normalized RMSE", "null_refined_circle_fit_normalized_rmse_by_n.png"),
        ("circle_fit_refined_line_mean_to_center_mean_ratio", "Fitted-circle line mean/center mean density ratio", "null_fitted_circle_line_mean_to_center_mean_by_n.png"),
        ("circle_fit_refined_line_max_to_center_mean_ratio", "Fitted-circle line max/center mean density ratio", "null_fitted_circle_line_max_to_center_mean_by_n.png"),
        ("circle_fit_refined_annulus_band_mean_to_center_mean_ratio", "Fitted-annulus band mean/center mean density ratio", "null_fitted_annulus_band_mean_to_center_mean_by_n.png"),
        ("circle_fit_refined_annulus_band_max_to_center_mean_ratio", "Fitted-annulus band max/center mean density ratio", "null_fitted_annulus_band_max_to_center_mean_by_n.png"),
        ("center_slice_annulus_to_center_ratio", "Fixed-center-slice annulus/center density ratio (QC only)", "null_fixed_center_slice_annulus_to_center_ratio_by_n.png"),
        ("annular_enrichment_index", "Annular enrichment index, max(0, annulus/center - 1)", "null_annular_enrichment_index_by_n.png"),
        ("central_depletion_index", "Central depletion index, max(0, 1 - center/annulus)", "null_central_depletion_index_by_n.png"),
    ]

    for col, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=PNG_DPI)
        data = []
        labels = []
        for n, g in df.groupby("n_requested"):
            vals = pd.to_numeric(g[col], errors="coerce").dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(str(n))
        if data:
            ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_xlabel("Simulated localizations per trace")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if col == "circle_fit_refined_diameter_nm":
            ax.axhspan(18, 19, alpha=0.15, label="Experimental ~18-19 nm")
            ax.legend(fontsize=8)
        if "to_center" in col or col == "center_slice_annulus_to_center_ratio":
            ax.axhline(1.0, linestyle="--", linewidth=1.0)
        if col in {"annular_enrichment_index", "central_depletion_index"}:
            ax.axhline(0.0, linestyle="--", linewidth=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), transparent=True)
        plt.close(fig)


# Main simulation

def run_null_simulation() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    examples_dir = os.path.join(OUTPUT_ROOT, "example_pngs")
    npz_dir = os.path.join(OUTPUT_ROOT, "example_npz")
    plots_dir = os.path.join(OUTPUT_ROOT, "summary_plots")
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    vals, Xg, Yg, Zg, grid_pts = build_grid(VOX_NM)
    rng = np.random.default_rng(RANDOM_SEED)
    results: List[NullReplicateResult] = []
    all_2d_peak_dfs: List[pd.DataFrame] = []

    print("--- MINFLUX KDE null simulation v5 ---")
    print(f"Output folder: {OUTPUT_ROOT}")
    print(f"Grid: {GRID_MIN_NM}-{GRID_MAX_NM} nm, voxel={VOX_NM} nm, total grid points={grid_pts.shape[0]:,}")
    print(f"KDE evaluation mode: {KDE_EVALUATION_METHOD}")
    print(f"Sub-voxel 2D refinement: {ENABLE_SUBVOXEL_2D_REFINEMENT}, step={SUBVOXEL_FINE_STEP_XY_NM} nm")
    print(f"Null sigmas: X={SIGMA_X_NM}, Y={SIGMA_Y_NM}, Z={SIGMA_Z_NM} nm")
    print(f"N values: {N_POINTS_LIST}; replicates per N: {N_REPLICATES}")

    for n_points in N_POINTS_LIST:
        for rep in range(1, N_REPLICATES + 1):
            print(f"\nProcessing null N={n_points}, replicate {rep}/{N_REPLICATES}")
            pts = simulate_null_trace(n_points, rng)
            bw = calculate_global_kth_nn_bandwidth(pts, K_BAND)
            density = kde_density_on_grid(pts, grid_pts, vals, bw, VOX_NM, KDE_GRID_CHUNK_SIZE)
            density_norm, global_max = normalize_density(density)

            # 3D peaks retained as a sanity check only.
            dynamic_min_peak_dist_3d_nm = max(VOX_NM, 1 * bw)
            peaks_3d = find_3d_peaks_from_density(
                density,
                vals,
                VOX_NM,
                dynamic_min_peak_dist_3d_nm,
                PEAK_THRESHOLD_REL_3D,
            )

            if len(peaks_3d) > 0:
                dom3_x, dom3_y, dom3_z, _ = peaks_3d[0]
                dom3_xy_r = math.sqrt((dom3_x - CENTER_NM) ** 2 + (dom3_y - CENTER_NM) ** 2)
                dom3_3d_d = math.sqrt((dom3_x - CENTER_NM) ** 2 + (dom3_y - CENTER_NM) ** 2 + (dom3_z - CENTER_NM) ** 2)
            else:
                dom3_x = dom3_y = dom3_z = dom3_xy_r = dom3_3d_d = np.nan

            dom3_x_ref = dom3_y_ref = dom3_z_ref = dom3_xy_r_ref = dom3_3d_d_ref = np.nan
            if ENABLE_SUBVOXEL_DOMINANT_3D_REFINEMENT and len(peaks_3d) > 0:
                dom3_x_ref, dom3_y_ref, dom3_z_ref, _ = continuous_kde_3d_local_max(
                    pts=pts,
                    bandwidth_nm=bw,
                    x0_nm=dom3_x,
                    y0_nm=dom3_y,
                    z0_nm=dom3_z,
                )
                dom3_xy_r_ref = math.sqrt((dom3_x_ref - CENTER_NM) ** 2 + (dom3_y_ref - CENTER_NM) ** 2)
                dom3_3d_d_ref = math.sqrt((dom3_x_ref - CENTER_NM) ** 2 + (dom3_y_ref - CENTER_NM) ** 2 + (dom3_z_ref - CENTER_NM) ** 2)

            # 2D peaks and local sub-voxel refinement.
            peaks_2d_df = find_2d_slice_peaks(
                density_normalized_3d=density_norm,
                vals=vals,
                pts=pts,
                bandwidth_nm=bw,
                n_requested=n_points,
                replicate=rep,
            )
            if len(peaks_2d_df) > 0:
                all_2d_peak_dfs.append(peaks_2d_df.copy())

                radii = peaks_2d_df["Radius_XY_From_True_Center_nm"].to_numpy(float)
                median_r = float(np.median(radii))
                max_r = float(np.max(radii))
                n_in_annulus_window = int(peaks_2d_df["In_Annulus_Radius_Window"].sum())
                n_slices_with_peaks = int(peaks_2d_df["Z_Center_nm"].nunique())

                refined_radii_series = pd.to_numeric(peaks_2d_df["Radius_XY_Refined_From_True_Center_nm"], errors="coerce").dropna()
                if len(refined_radii_series) > 0:
                    refined_radii = refined_radii_series.to_numpy(float)
                    median_r_ref = float(np.median(refined_radii))
                    max_r_ref = float(np.max(refined_radii))
                    n_in_annulus_window_ref = int(peaks_2d_df["In_Annulus_Radius_Window_Refined"].sum())
                else:
                    median_r_ref = np.nan
                    max_r_ref = np.nan
                    n_in_annulus_window_ref = 0

                dom2_idx = int(peaks_2d_df["Normalized_Density_at_Peak"].idxmax())
                dom2 = peaks_2d_df.loc[dom2_idx]
                dom2_x = float(dom2["X_nm"])
                dom2_y = float(dom2["Y_nm"])
                dom2_zc = float(dom2["Z_Center_nm"])
                dom2_r = float(dom2["Radius_XY_From_True_Center_nm"])
                dom2_x_ref = float(dom2["X_refined_nm"]) if not pd.isna(dom2["X_refined_nm"]) else np.nan
                dom2_y_ref = float(dom2["Y_refined_nm"]) if not pd.isna(dom2["Y_refined_nm"]) else np.nan
                dom2_r_ref = float(dom2["Radius_XY_Refined_From_True_Center_nm"]) if not pd.isna(dom2["Radius_XY_Refined_From_True_Center_nm"]) else np.nan
            else:
                median_r = max_r = median_r_ref = max_r_ref = np.nan
                n_in_annulus_window = n_in_annulus_window_ref = 0
                n_slices_with_peaks = 0
                dom2_x = dom2_y = dom2_zc = dom2_r = np.nan
                dom2_x_ref = dom2_y_ref = dom2_r_ref = np.nan

            # Circle fits: coarse and refined (refined is reported).
            circle_used = False
            circle_n = 0
            circle_cx = circle_cy = circle_r = circle_d = circle_center_dist = np.nan
            circle_r_in_window = False
            circle_d_in_exp_window = False
            circle_rmse = circle_mae = circle_norm_rmse = circle_pseudo_r2 = circle_adj_pseudo_r2 = np.nan
            circle_cx_ref = circle_cy_ref = circle_r_ref = circle_d_ref = circle_center_dist_ref = np.nan
            circle_r_in_window_ref = False
            circle_d_in_exp_window_ref = False
            circle_rmse_ref = circle_mae_ref = circle_norm_rmse_ref = circle_pseudo_r2_ref = circle_adj_pseudo_r2_ref = np.nan
            fitted_circle_density = fitted_circle_density_metrics(density_norm, vals, np.nan, np.nan, np.nan, np.array([]))

            if len(peaks_2d_df) >= MIN_PEAKS_FOR_CIRCLE_FIT:
                circle_used = True
                circle_n = len(peaks_2d_df)
                weights = peaks_2d_df["Normalized_Density_at_Peak"].to_numpy(float)

                circle_cx, circle_cy, circle_r = weighted_circle_fit_xy(
                    peaks_2d_df["X_nm"].to_numpy(float),
                    peaks_2d_df["Y_nm"].to_numpy(float),
                    weights=weights,
                )
                circle_d = 2 * circle_r
                circle_center_dist = math.sqrt((circle_cx - CENTER_NM) ** 2 + (circle_cy - CENTER_NM) ** 2)
                circle_r_in_window = ANNULUS_RADIUS_MIN_NM <= circle_r <= ANNULUS_RADIUS_MAX_NM
                circle_d_in_exp_window = 18.0 <= circle_d <= 19.0
                q = circle_fit_quality_metrics(
                    peaks_2d_df["X_nm"].to_numpy(float),
                    peaks_2d_df["Y_nm"].to_numpy(float),
                    circle_cx,
                    circle_cy,
                    circle_r,
                    weights=weights,
                )
                circle_rmse = q["rmse_nm"]
                circle_mae = q["mae_nm"]
                circle_norm_rmse = q["normalized_rmse"]
                circle_pseudo_r2 = q["pseudo_r2"]
                circle_adj_pseudo_r2 = q["adjusted_pseudo_r2"]

                refined_valid = peaks_2d_df.dropna(subset=["X_refined_nm", "Y_refined_nm"])
                if len(refined_valid) >= MIN_PEAKS_FOR_CIRCLE_FIT:
                    weights_ref = refined_valid["Normalized_Density_at_Peak"].to_numpy(float)
                    circle_cx_ref, circle_cy_ref, circle_r_ref = weighted_circle_fit_xy(
                        refined_valid["X_refined_nm"].to_numpy(float),
                        refined_valid["Y_refined_nm"].to_numpy(float),
                        weights=weights_ref,
                    )
                    circle_d_ref = 2 * circle_r_ref
                    circle_center_dist_ref = math.sqrt((circle_cx_ref - CENTER_NM) ** 2 + (circle_cy_ref - CENTER_NM) ** 2)
                    circle_r_in_window_ref = ANNULUS_RADIUS_MIN_NM <= circle_r_ref <= ANNULUS_RADIUS_MAX_NM
                    circle_d_in_exp_window_ref = 18.0 <= circle_d_ref <= 19.0
                    q_ref = circle_fit_quality_metrics(
                        refined_valid["X_refined_nm"].to_numpy(float),
                        refined_valid["Y_refined_nm"].to_numpy(float),
                        circle_cx_ref,
                        circle_cy_ref,
                        circle_r_ref,
                        weights=weights_ref,
                    )
                    circle_rmse_ref = q_ref["rmse_nm"]
                    circle_mae_ref = q_ref["mae_nm"]
                    circle_norm_rmse_ref = q_ref["normalized_rmse"]
                    circle_pseudo_r2_ref = q_ref["pseudo_r2"]
                    circle_adj_pseudo_r2_ref = q_ref["adjusted_pseudo_r2"]

                    if FITTED_CIRCLE_DENSITY_Z_MODE == "peak_slices":
                        z_for_density = refined_valid["Z_Center_nm"].to_numpy(float)
                    else:
                        z_for_density = np.array(NEW_2D_PROJ_CENTRES, dtype=float)
                    fitted_circle_density = fitted_circle_density_metrics(
                        density_normalized_3d=density_norm,
                        vals=vals,
                        center_x_nm=circle_cx_ref,
                        center_y_nm=circle_cy_ref,
                        radius_nm=circle_r_ref,
                        z_centers_nm=z_for_density,
                    )

            center_val, max_val, center_to_max, ann_mean, ann_to_center, center_to_annulus, ann_minus_center, annular_enrichment_idx, central_depletion_idx = central_slice_depletion_metrics(density_norm, vals)
            center_depleted = bool(np.isfinite(ann_to_center) and ann_to_center > MIN_ANNULUS_TO_CENTER_RATIO_FOR_FALSE_ANNULUS)
            false_annulus_call = bool(
                circle_used
                and np.isfinite(circle_d_ref)
                and FALSE_ANNULUS_DIAMETER_MIN_NM <= circle_d_ref <= FALSE_ANNULUS_DIAMETER_MAX_NM
                and np.isfinite(fitted_circle_density.get("circle_fit_refined_annulus_band_mean_to_center_mean_ratio", np.nan))
                and fitted_circle_density.get("circle_fit_refined_annulus_band_mean_to_center_mean_ratio", np.nan) > MIN_ANNULUS_TO_CENTER_RATIO_FOR_FALSE_ANNULUS
            )
            false_annulus_call_strict = bool(
                circle_used
                and np.isfinite(circle_d_ref)
                and FALSE_ANNULUS_DIAMETER_MIN_NM <= circle_d_ref <= FALSE_ANNULUS_DIAMETER_MAX_NM
                and np.isfinite(fitted_circle_density.get("circle_fit_refined_annulus_band_mean_to_center_mean_ratio", np.nan))
                and fitted_circle_density.get("circle_fit_refined_annulus_band_mean_to_center_mean_ratio", np.nan) > MIN_ANNULUS_TO_CENTER_RATIO_STRICT
            )

            results.append(NullReplicateResult(
                n_requested=n_points,
                replicate=rep,
                n_after_cube_filter=len(pts),
                sigma_x_nm=SIGMA_X_NM,
                sigma_y_nm=SIGMA_Y_NM,
                sigma_z_nm=SIGMA_Z_NM,
                sigma_avg_3d_nm=SIGMA_AVG_3D_NM,
                k_band=K_BAND,
                kde_bandwidth_nm=bw,
                global_max_density=global_max,
                n_3d_peaks=len(peaks_3d),
                dominant_3d_peak_x_nm=dom3_x,
                dominant_3d_peak_y_nm=dom3_y,
                dominant_3d_peak_z_nm=dom3_z,
                dominant_3d_peak_xy_radius_from_center_nm=dom3_xy_r,
                dominant_3d_peak_3d_distance_from_center_nm=dom3_3d_d,
                dominant_3d_peak_refined_x_nm=dom3_x_ref,
                dominant_3d_peak_refined_y_nm=dom3_y_ref,
                dominant_3d_peak_refined_z_nm=dom3_z_ref,
                dominant_3d_peak_refined_xy_radius_from_center_nm=dom3_xy_r_ref,
                dominant_3d_peak_refined_3d_distance_from_center_nm=dom3_3d_d_ref,
                n_2d_peaks_total=len(peaks_2d_df),
                n_2d_slices_with_peaks=n_slices_with_peaks,
                dominant_2d_peak_x_nm=dom2_x,
                dominant_2d_peak_y_nm=dom2_y,
                dominant_2d_peak_z_center_nm=dom2_zc,
                dominant_2d_peak_radius_xy_nm=dom2_r,
                dominant_2d_peak_refined_x_nm=dom2_x_ref,
                dominant_2d_peak_refined_y_nm=dom2_y_ref,
                dominant_2d_peak_refined_radius_xy_nm=dom2_r_ref,
                median_2d_peak_radius_xy_nm=median_r,
                max_2d_peak_radius_xy_nm=max_r,
                median_2d_peak_radius_xy_refined_nm=median_r_ref,
                max_2d_peak_radius_xy_refined_nm=max_r_ref,
                n_2d_peaks_in_annulus_radius_window=n_in_annulus_window,
                n_2d_peaks_in_annulus_radius_window_refined=n_in_annulus_window_ref,
                circle_fit_used=circle_used,
                circle_fit_n_peaks=circle_n,
                circle_fit_center_x_nm=circle_cx,
                circle_fit_center_y_nm=circle_cy,
                circle_fit_radius_nm=circle_r,
                circle_fit_diameter_nm=circle_d,
                circle_fit_center_distance_from_true_center_nm=circle_center_dist,
                circle_fit_radius_in_annulus_window=circle_r_in_window,
                circle_fit_diameter_in_experimental_window=circle_d_in_exp_window,
                circle_fit_rmse_nm=circle_rmse,
                circle_fit_mae_nm=circle_mae,
                circle_fit_normalized_rmse=circle_norm_rmse,
                circle_fit_pseudo_r2=circle_pseudo_r2,
                circle_fit_adjusted_pseudo_r2=circle_adj_pseudo_r2,
                circle_fit_refined_center_x_nm=circle_cx_ref,
                circle_fit_refined_center_y_nm=circle_cy_ref,
                circle_fit_refined_radius_nm=circle_r_ref,
                circle_fit_refined_diameter_nm=circle_d_ref,
                circle_fit_refined_center_distance_from_true_center_nm=circle_center_dist_ref,
                circle_fit_refined_radius_in_annulus_window=circle_r_in_window_ref,
                circle_fit_refined_diameter_in_experimental_window=circle_d_in_exp_window_ref,
                circle_fit_refined_rmse_nm=circle_rmse_ref,
                circle_fit_refined_mae_nm=circle_mae_ref,
                circle_fit_refined_normalized_rmse=circle_norm_rmse_ref,
                circle_fit_refined_pseudo_r2=circle_pseudo_r2_ref,
                circle_fit_refined_adjusted_pseudo_r2=circle_adj_pseudo_r2_ref,
                center_slice_center_density_norm=center_val,
                center_slice_max_density_norm=max_val,
                center_slice_center_to_max_ratio=center_to_max,
                center_slice_annulus_mean_density_norm=ann_mean,
                center_slice_annulus_to_center_ratio=ann_to_center,
                center_slice_center_to_annulus_ratio=center_to_annulus,
                center_slice_annulus_minus_center_density_norm=ann_minus_center,
                annular_enrichment_index=annular_enrichment_idx,
                central_depletion_index=central_depletion_idx,
                fitted_circle_density_z_count=fitted_circle_density["fitted_circle_density_z_count"],
                circle_fit_refined_line_mean_density_norm=fitted_circle_density["circle_fit_refined_line_mean_density_norm"],
                circle_fit_refined_line_max_density_norm=fitted_circle_density["circle_fit_refined_line_max_density_norm"],
                circle_fit_refined_annulus_band_mean_density_norm=fitted_circle_density["circle_fit_refined_annulus_band_mean_density_norm"],
                circle_fit_refined_annulus_band_max_density_norm=fitted_circle_density["circle_fit_refined_annulus_band_max_density_norm"],
                circle_fit_refined_center_mean_density_norm=fitted_circle_density["circle_fit_refined_center_mean_density_norm"],
                circle_fit_refined_center_max_density_norm=fitted_circle_density["circle_fit_refined_center_max_density_norm"],
                circle_fit_refined_line_mean_to_center_mean_ratio=fitted_circle_density["circle_fit_refined_line_mean_to_center_mean_ratio"],
                circle_fit_refined_line_max_to_center_mean_ratio=fitted_circle_density["circle_fit_refined_line_max_to_center_mean_ratio"],
                circle_fit_refined_annulus_band_mean_to_center_mean_ratio=fitted_circle_density["circle_fit_refined_annulus_band_mean_to_center_mean_ratio"],
                circle_fit_refined_annulus_band_max_to_center_mean_ratio=fitted_circle_density["circle_fit_refined_annulus_band_max_to_center_mean_ratio"],
                circle_fit_refined_line_mean_to_center_max_ratio=fitted_circle_density["circle_fit_refined_line_mean_to_center_max_ratio"],
                circle_fit_refined_line_max_to_center_max_ratio=fitted_circle_density["circle_fit_refined_line_max_to_center_max_ratio"],
                circle_fit_refined_annulus_band_mean_to_center_max_ratio=fitted_circle_density["circle_fit_refined_annulus_band_mean_to_center_max_ratio"],
                circle_fit_refined_annulus_band_max_to_center_max_ratio=fitted_circle_density["circle_fit_refined_annulus_band_max_to_center_max_ratio"],
                center_depleted_binary=center_depleted,
                false_annulus_call=false_annulus_call,
                false_annulus_call_strict=false_annulus_call_strict,
            ))

            # Save one representative example per N by default.
            if SAVE_EXAMPLE_PNGS and rep == 1:
                save_example_center_slice(density_norm, vals, peaks_2d_df, n_points, rep, examples_dir)

            if SAVE_ALL_NPZ or (SAVE_EXAMPLE_NPZ and rep == 1):
                np.savez_compressed(
                    os.path.join(npz_dir, f"null_N{n_points}_rep{rep:03d}_density_data.npz"),
                    density_map_physical=density.astype(np.float32),
                    density_map_normalized=density_norm.astype(np.float32),
                    kde_bandwidth=bw,
                    vals=vals.astype(np.float32),
                    pts_locs=pts.astype(np.float32),
                    num_locs=len(pts),
                    sigma_xyz_nm=np.array([SIGMA_X_NM, SIGMA_Y_NM, SIGMA_Z_NM], dtype=np.float32),
                    grid_vox=VOX_NM,
                    null_center_raw=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    processed_center=np.array([CENTER_NM, CENTER_NM, CENTER_NM], dtype=np.float32),
                )

            del pts, density, density_norm, peaks_2d_df
            gc.collect()

    per_rep_df = pd.DataFrame([asdict(r) for r in results])
    per_rep_path = os.path.join(OUTPUT_ROOT, "null_simulation_per_replicate.csv")
    per_rep_df.to_csv(per_rep_path, index=False)
    print(f"\nSaved per-replicate results: {per_rep_path}")

    if SAVE_2D_PEAKS_CSV and all_2d_peak_dfs:
        peaks_all_df = pd.concat(all_2d_peak_dfs, ignore_index=True)
        peaks_path = os.path.join(OUTPUT_ROOT, "null_simulation_2d_peaks_all_replicates.csv")
        peaks_all_df.to_csv(peaks_path, index=False)
        print(f"Saved 2D peak-level results: {peaks_path}")

    summary_df = summarize_results(per_rep_df)
    summary_path = os.path.join(OUTPUT_ROOT, "null_simulation_summary_by_n.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary results: {summary_path}")

    formatted_summary_df = make_formatted_summary_table(summary_df)
    formatted_summary_path = os.path.join(OUTPUT_ROOT, "null_simulation_summary_by_n_formatted_median_95interval.csv")
    formatted_summary_df.to_csv(formatted_summary_path, index=False)
    print(f"Saved formatted summary table: {formatted_summary_path}")

    save_summary_plots(per_rep_df, plots_dir)
    print(f"Saved summary plots in: {plots_dir}")

    print("\n--- Null simulation v5 complete ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_null_simulation()
