# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings of the SMAP validation. The SMAP analysis settings themselves (grouping,
filters, segmentNPC, NPCsegmentCleanup, LocMoFit) are read from ../01_extract.py."""

SML_FILE = r''          # CamR localization file to validate on, e.g. ...\ROI-R_driftc_sml.mat
SMAP_DIR = r''          # SMAP installation folder (the one holding SMAP.m)
MATLAB_EXE = r'C:\Program Files\MATLAB\R2022b\bin\matlab.exe'

OUTPUT_DIR = 'results'  # relative to this folder, or an absolute path
RUN_NAME = 'run1'       # output goes to OUTPUT_DIR\RUN_NAME

RUN_MATLAB = True       # False = only repeat the comparison with an existing MATLAB result
MATLAB_MODE = 'batch'   # 'batch' (matlab -batch) or 'desktop' if SMAP does not start in batch mode
LOCMOFIT_MAX_SITES = None   # None = every accepted site
