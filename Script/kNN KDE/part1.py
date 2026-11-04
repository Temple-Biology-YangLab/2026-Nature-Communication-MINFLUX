# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 1: 3D KDE of a single MINFLUX trace (Excel input), saved as .npz."""

# IMPORTS
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KernelDensity
import gc

# USER PARAMETERS
excel_file = "Nup58C M0101Tid4156.xlsx"  # ← workbook to process
output_data_directory = "kde_data_output" # Directory to save the .npz file
vox = 1  # KDE grid spacing (nm)
k_band = 3 # K-th nearest neighbor for bandwidth calculation

# HELPER FUNCTION
def parse_prefix_from_filename(filename_str):
    """Parses M####Tid#### prefix from the Excel filename."""
    match = re.search(r'\sM(\d{2})(\d{2})Tid(\d+)\.xlsx$', os.path.basename(filename_str))
    if not match:
        raise ValueError(f"Filename '{filename_str}' must match ' … M####Tid####.xlsx' format to extract prefix.")
    meas_idx_str, sub_idx_str, tid_val_str = match.groups()
    return f"M{meas_idx_str}{sub_idx_str}Tid{tid_val_str}"

# MAIN SCRIPT LOGIC
def calculate_and_save_kde():
    """Main function to calculate KDE and save data."""
    try:
        prefix_str = parse_prefix_from_filename(excel_file)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Processing trace: {prefix_str} from file: {excel_file}")

    # Build Grid
    print(f"Building grid with voxel size: {vox} nm...")
    vals_grid = np.arange(0, 121, vox) 
    n_vox_grid = len(vals_grid)
    # Grid for density map (Z,Y,X order)
    Zg_grid, Yg_grid, Xg_grid = np.meshgrid(vals_grid, vals_grid, vals_grid, indexing="ij") 
    # Grid points for KDE input (X,Y,Z order)
    grid_pts_kde_input = np.vstack([Xg_grid.ravel(), Yg_grid.ravel(), Zg_grid.ravel()]).T.astype(np.float32) 
    grid_voxel_volume = vox**3 
    print(f"Grid dimensions: {n_vox_grid}x{n_vox_grid}x{n_vox_grid}")

    # Load Localizations
    print(f"Loading localizations from: {excel_file}...")
    try:
        df_localizations = pd.read_excel(excel_file, sheet_name=0) # Assuming data is in the first sheet
    except FileNotFoundError:
        print(f"Error: Excel file not found at '{excel_file}'"); return
    except Exception as e:
        print(f"Error reading Excel file '{excel_file}': {e}"); return

    if not {"X (nm)", "Y (nm)", "Z (nm)"}.issubset(df_localizations.columns):
        print(f"Error: Excel file '{excel_file}' is missing required columns: X (nm), Y (nm), Z (nm)."); return

    pts_localizations_raw = df_localizations[["X (nm)", "Y (nm)", "Z (nm)"]].to_numpy(np.float32)
    if pts_localizations_raw.shape[0] == 0: 
        print(f"Error: No localization data found in '{excel_file}'."); return

    # Center and filter points
    pts_localizations_processed = pts_localizations_raw - pts_localizations_raw.mean(axis=0) 
    pts_localizations_processed += 60 
    pts_localizations_processed = pts_localizations_processed[np.all((pts_localizations_processed>=0)&(pts_localizations_processed<=120), axis=1)]

    if pts_localizations_processed.size == 0: 
        print(f"Error: No points remain inside the 0–120 nm cube for '{prefix_str}'."); return
    num_locs_processed = len(pts_localizations_processed)
    print(f"Loaded and processed {num_locs_processed} localizations within the 0-120nm cube.")

    # KDE Calculation
    print("Calculating KDE...")
    nn_model_kde = NearestNeighbors(n_neighbors=k_band).fit(pts_localizations_processed)
    distances_nn_kde, _ = nn_model_kde.kneighbors(pts_localizations_processed)
    kde_bandwidth_calculated = np.median(distances_nn_kde[:, k_band-1])

    if kde_bandwidth_calculated == 0 or pd.isna(kde_bandwidth_calculated) or kde_bandwidth_calculated < 1e-6:
        kde_bandwidth_calculated = 0.1 * vox 
        print(f"  Warning: Calculated KDE bandwidth was problematic. Set to default: {kde_bandwidth_calculated:.3f} nm.")
    print(f"  KDE Bandwidth (bw): {kde_bandwidth_calculated:.3f} nm")

    kde_model_instance_fitted = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth_calculated).fit(pts_localizations_processed)
    # Calculate density on the grid
    density_map_physical = np.exp(kde_model_instance_fitted.score_samples(grid_pts_kde_input)).reshape((n_vox_grid, n_vox_grid, n_vox_grid))
    density_map_physical = density_map_physical * num_locs_processed / grid_voxel_volume  # Convert to loc/nm³
    print("KDE calculation complete.")

    # Save Density Data and Metadata
    os.makedirs(output_data_directory, exist_ok=True) # Create output directory if it doesn't exist
    output_npz_filename = f"{prefix_str}_density_data.npz"
    output_npz_path = os.path.join(output_data_directory, output_npz_filename)

    print(f"Saving density data and metadata to: {output_npz_path}...")
    try:
        np.savez_compressed(
            output_npz_path,
            density_map_physical=density_map_physical.astype(np.float32), 
            kde_bandwidth=kde_bandwidth_calculated,
            prefix=prefix_str,
            grid_vox=vox,
            vals=vals_grid, 
            Xg=Xg_grid.astype(np.float16), 
            Yg=Yg_grid.astype(np.float16),
            Zg=Zg_grid.astype(np.float16),
            pts_locs=pts_localizations_processed.astype(np.float32), # Save processed localizations for raw trace plot
            num_locs = num_locs_processed,
            source_excel_file=excel_file 
        )
        print(f"  Successfully saved data to {output_npz_filename}")
    except Exception as e_save:
        print(f"  Error saving .npz file for {prefix_str}: {e_save}")

    # Clean up large arrays
    del density_map_physical, kde_model_instance_fitted, nn_model_kde, pts_localizations_raw, pts_localizations_processed, grid_pts_kde_input
    gc.collect()

if __name__ == "__main__":
    calculate_and_save_kde()
    print("\n--- Script 1: KDE Calculation and Data Saving Complete ---")
