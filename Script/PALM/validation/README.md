# Validation of the SMAP port

Compares `npc_pipeline.smap` (used by `01_extract.py` for CamR pore finding) with SMAP running
in MATLAB, on one CamR localization file (`*_driftc_sml.mat`).

Requirements: MATLAB and a SMAP installation, in addition to the Python packages of
the main project.

## Run

1. Edit `settings_validation.py`: `SML_FILE`, `SMAP_DIR`, `MATLAB_EXE`.
2. `python run_validation.py`

The SMAP settings (grouping, layer filters, cell rendering, segmentNPC,
NPCsegmentCleanup, LocMoFit) are taken from the SETTINGS block of `../01_extract.py`,
so the validation tests the configuration used for the analysis.

MATLAB runs `matlab/smap_reference.m`, which drives SMAP itself and exports each stage.
Python then repeats each stage on the input SMAP used for that stage, so a difference
cannot propagate from one stage to the next.

| stage | compared |
|---|---|
| S2 | grouping and layer-1 filtering |
| S4 | cell images rendered for segmentNPC |
| S5 | segmentNPC sites |
| S6 | NPCsegmentCleanup: localizations per site, ring fit, accept/reject |
| S7 | LocMoFit parameters and log-likelihood per site |
| S8 | end to end: `npc_pipeline.smap_sites` on the same file vs SMAP's accepted sites and LocMoFit results |

Output in `<OUTPUT_DIR>/<RUN_NAME>/`: `config.json`, `matlab_reference/` (SMAP export,
`matlab_log.txt`, `status.json`) and `comparison/` (`report.md`, `summary.json`,
per-site CSV tables, scatter plots).

`RUN_MATLAB = False` repeats only the comparison. If SMAP does not start with
`matlab -batch`, set `MATLAB_MODE = 'desktop'`.

NPCsegmentCleanup and LocMoFit run iterative optimizers (MATLAB `lsqnonlin` /
`fminsearchbnd` vs SciPy / the Python port). Sparse sites can converge to different
local minima; the report separates these from deterministic differences.
