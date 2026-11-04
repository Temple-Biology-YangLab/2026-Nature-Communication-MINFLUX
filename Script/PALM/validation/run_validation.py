# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate the SMAP port (npc_pipeline.smap) against SMAP in MATLAB on one CamR _sml.mat.

Edit settings_validation.py, then run:  python run_validation.py
Report: <OUTPUT_DIR>/<RUN_NAME>/comparison/report.md
"""
import importlib
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import settings_validation as V
import validation_lib as VL


def main():
    importlib.reload(V)
    S = {k: v for k, v in runpy.run_path(os.path.join(ROOT, '01_extract.py')).items() if k.isupper()}
    out = V.OUTPUT_DIR if os.path.isabs(V.OUTPUT_DIR) else os.path.join(HERE, V.OUTPUT_DIR)
    run_dir = os.path.join(out, V.RUN_NAME)
    cfg = VL.make_config(S, os.path.abspath(V.SML_FILE), V.SMAP_DIR, V.LOCMOFIT_MAX_SITES)
    cfg_path = VL.write_config(cfg, run_dir)
    print('input      : %s' % cfg['input_sml'])
    print('run folder : %s' % run_dir)

    if V.RUN_MATLAB:
        for f in (V.MATLAB_EXE, V.SMAP_DIR, V.SML_FILE):
            if not f or not os.path.exists(f):
                raise FileNotFoundError('not found: %r (see settings_validation.py)' % f)
        VL.run_matlab(cfg_path, V.MATLAB_EXE, mode=V.MATLAB_MODE)
    VL.print_matlab_status(run_dir)

    summary = VL.compare_all(run_dir)
    VL.print_summary(summary, run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
