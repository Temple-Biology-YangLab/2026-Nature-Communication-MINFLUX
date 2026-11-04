#!/usr/bin/env python3
# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitted-radius CDFs (OU null vs experiment) from an ou_null_simulation.py run.

The null band is a bootstrap envelope at the experimental sample size:
'simultaneous' (global rank envelope; Myllymaki et al., J R Stat Soc B 79:381,
2017), 'ks' (null CDF +/- KS critical value) or 'pointwise' (2.5-97.5th
percentile). Edit SETTINGS and run.
"""
import os
import glob
import math

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

#  SETTINGS
# Input
RUN_FOLDER   = None            # None = most recent folder in outputs/; or a full path
CONDITION    = 'base'
CONSTRUCTS   = ['GFP-Nup58-HaloTag', 'HaloTag-Nup54-GFP', 'HaloTag-Nup58-GFP', 'HaloTag-Nup62-GFP']
OUTPUT_SUBFOLDER = 'plots'     # figures go to <RUN_FOLDER>/<OUTPUT_SUBFOLDER>/

# Uncertainty band for the OU null
SHOW_NULL_BAND = True
BAND_TYPE      = 'simultaneous'   # 'simultaneous', 'ks', or 'pointwise'
BAND_LEVEL     = 0.95
N_BAND_DRAWS   = 10000
BAND_SEED      = 20261016
BAND_GRID_STEP_NM = 0.05          # radius grid for the band

# Figure layout
GRID_COLUMNS        = 2
FIG_SIZE_GRID_IN    = (9.0, 8.0)  # whole multi-panel figure (width, height), inches
FIG_SIZE_SINGLE_IN  = (4.4, 4.0)
DPI                 = 300
FORMATS             = ('png',)
FONT_FAMILY         = 'Arial'
FONT_SIZE_LABEL     = 10
FONT_SIZE_TICK      = 9
FONT_SIZE_TITLE     = 11
FONT_SIZE_LEGEND    = 8
FONT_SIZE_STATS     = 7.5
PANEL_LETTERS       = False       # 'A', 'B', ... in the multi-panel figure
PANEL_LETTER_SIZE   = 12

# Lines and band
NULL_COLOR      = '#3d4a57'
NULL_LINEWIDTH  = 1.6
NULL_LINESTYLE  = '-'
EXP_COLOR       = '#c0392b'
EXP_LINEWIDTH   = 1.9
EXP_LINESTYLE   = '-'
BAND_COLOR      = '#3d4a57'
BAND_ALPHA      = 0.18
AXIS_LINEWIDTH  = 0.8
TICK_LENGTH     = 3.0

# Axes (defaults for every panel; override per panel below)
X_LIM           = (0, 25)
Y_LIM           = (0, 1.02)
X_TICKS         = [0, 5, 10, 15, 20, 25]
Y_TICKS         = [0, 0.25, 0.5, 0.75, 1.0]
X_LABEL         = 'Fitted circle radius (nm)'
Y_LABEL         = 'Cumulative Fraction (arb. unit)'
SHOW_TITLE      = True
TITLE_LOC       = 'left'          # 'left', 'center' or 'right'
TITLE_ABOVE_LEGEND = True
TITLES          = {'GFP-Nup58-HaloTag': 'Nup58 C terminus', 'HaloTag-Nup54-GFP': 'Nup54 N terminus',
                   'HaloTag-Nup58-GFP': 'Nup58 N terminus', 'HaloTag-Nup62-GFP': 'Nup62 N terminus'}

# Legend
SHOW_LEGEND     = True
LEGEND_LOC      = 'lower right'           # corner of the legend placed at LEGEND_BBOX
LEGEND_BBOX     = (1.0, 1.0)
NULL_LABEL      = 'OU Null'                    # {n} available = number of simulated traces
EXP_LABEL       = 'Experiment'                 # {n} available = number of experimental traces
BAND_LABEL      = 'OU Null, {level:.0%} bootstrap CI'   # {level}, {n_exp} available

# Statistics text
SHOW_STATS      = True
STATS_LOC       = (0.98, 0.04)                # axes fraction of the anchor point
STATS_HA, STATS_VA = 'right', 'bottom'
SHOW_MEDIANS    = True
SHOW_MANNWHITNEY = True
SHOW_KS         = True
SHOW_BAND_P     = True
STATS_COLOR     = '0.25'

PANEL = {
}


def latest_run():
    runs = sorted(glob.glob(os.path.join(HERE, 'outputs', '*')), key=os.path.getmtime)
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        raise FileNotFoundError('No run folders in outputs/')
    return runs[-1]


def load(run, construct):
    def radii(path):
        d = pd.read_csv(path)
        return np.sort(d.loc[d.fit_success.astype(bool), 'radius_nm'].to_numpy(float))
    null = radii(os.path.join(run, f'{CONDITION}_{construct}_null_per_trace.csv'))
    exp = radii(os.path.join(run, f'{CONDITION}_{construct}_experiment_per_trace.csv'))
    return null, exp


def ecdf_on(sorted_vals, grid):
    return np.searchsorted(sorted_vals, grid, side='right') / sorted_vals.size


def envelope(null, n_exp, exp, rng):
    """Sampling envelope of the CDF of n_exp draws from the null, and the envelope p-value."""
    lo_x = min(null.min(), exp.min(), X_LIM[0]); hi_x = max(null.max(), exp.max(), X_LIM[1])
    grid = np.arange(lo_x, hi_x + BAND_GRID_STEP_NM, BAND_GRID_STEP_NM)
    F0 = ecdf_on(null, grid)
    draws = np.sort(rng.choice(null, size=(N_BAND_DRAWS, n_exp), replace=True), axis=1)
    F = np.vstack([np.searchsorted(d, grid, side='right') for d in draws]) / n_exp
    Fe = ecdf_on(exp, grid)
    if BAND_TYPE == 'pointwise':
        a = (1 - BAND_LEVEL) / 2
        lo, hi = np.quantile(F, a, axis=0), np.quantile(F, 1 - a, axis=0)
        stat_draws = np.abs(F - F0).max(axis=1); stat_exp = np.abs(Fe - F0).max()   # KS-type p for reference
    elif BAND_TYPE == 'ks':
        stat_draws = np.abs(F - F0).max(axis=1); stat_exp = np.abs(Fe - F0).max()
        d = np.quantile(stat_draws, BAND_LEVEL)
        lo, hi = F0 - d, F0 + d
    elif BAND_TYPE == 'simultaneous':
        Fs = np.sort(F, axis=0); B = N_BAND_DRAWS
        def depth(G):
            # largest k such that Fs[k] <= G <= Fs[B-1-k] at every radius
            k_lo = np.array([np.searchsorted(Fs[:, j], G[..., j], side='right') for j in range(grid.size)]) - 1
            k_hi = B - 1 - np.array([np.searchsorted(Fs[:, j], G[..., j], side='left') for j in range(grid.size)])
            return np.minimum(k_lo, k_hi).min(axis=0)
        d_draws = depth(F)
        d_exp = int(depth(Fe[None, :])[0])
        k = int(np.sort(d_draws)[int(np.floor((1 - BAND_LEVEL) * B))])
        k = max(k, 0)
        lo, hi = Fs[k], Fs[B - 1 - k]
        stat_draws, stat_exp = -d_draws, -d_exp          # smaller depth = more extreme
    else:
        raise ValueError("BAND_TYPE must be 'simultaneous', 'ks' or 'pointwise'")
    p = (1 + np.sum(stat_draws >= stat_exp)) / (1 + N_BAND_DRAWS)
    outside = float(np.mean((Fe < lo - 1e-12) | (Fe > hi + 1e-12)))
    return grid, np.clip(lo, 0, 1), np.clip(hi, 0, 1), float(p), outside


def fmt_p(p, floor=None):
    if floor is not None and p <= floor * (1 + 1e-9):
        return f'p ≤ {floor:.1e}'                    # resolution limit of the Monte Carlo test
    return 'p < 1e-300' if p == 0 else (f'p = {p:.2g}' if p >= 1e-3 else f'p = {p:.1e}')


def opt(construct, key):
    return PANEL.get(construct, {}).get(key, globals()[key])


def draw(ax, construct, null, exp, band, stats):
    if band is not None:
        grid, lo, hi, _, _ = band
        ax.fill_between(grid, lo, hi, step='post', color=BAND_COLOR, alpha=BAND_ALPHA, lw=0,
                        label=BAND_LABEL.format(level=BAND_LEVEL, n_exp=exp.size))
    ax.step(np.r_[null[0], null], np.r_[0, np.arange(1, null.size + 1) / null.size], where='post',
            c=NULL_COLOR, lw=NULL_LINEWIDTH, ls=NULL_LINESTYLE, label=NULL_LABEL.format(n=null.size))
    ax.step(np.r_[exp[0], exp], np.r_[0, np.arange(1, exp.size + 1) / exp.size], where='post',
            c=EXP_COLOR, lw=EXP_LINEWIDTH, ls=EXP_LINESTYLE, label=EXP_LABEL.format(n=exp.size))
    ax.set_xlim(*opt(construct, 'X_LIM')); ax.set_ylim(*opt(construct, 'Y_LIM'))
    if opt(construct, 'X_TICKS') is not None: ax.set_xticks(opt(construct, 'X_TICKS'))
    if opt(construct, 'Y_TICKS') is not None: ax.set_yticks(opt(construct, 'Y_TICKS'))
    ax.set_xlabel(X_LABEL, fontsize=FONT_SIZE_LABEL); ax.set_ylabel(Y_LABEL, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK, width=AXIS_LINEWIDTH, length=TICK_LENGTH)
    for s in ('left', 'bottom'): ax.spines[s].set_linewidth(AXIS_LINEWIDTH)
    ax.spines[['top', 'right']].set_visible(False)
    pad = 6.0
    if opt(construct, 'SHOW_LEGEND'):
        bbox = opt(construct, 'LEGEND_BBOX')
        leg = ax.legend(frameon=False, fontsize=FONT_SIZE_LEGEND, loc=opt(construct, 'LEGEND_LOC'),
                        bbox_to_anchor=bbox, borderaxespad=0.2 if bbox is not None else 0.5)
        if bbox is not None and bbox[1] >= 1 and TITLE_ABOVE_LEGEND:
            fig = ax.figure
            h = leg.get_window_extent(fig.canvas.get_renderer()).height * 72.0 / fig.dpi
            pad = h + 4.0                       # lift the title above an outside legend
    if SHOW_TITLE:
        ax.set_title(PANEL.get(construct, {}).get('TITLE', TITLES.get(construct, construct)), fontsize=FONT_SIZE_TITLE,
                     loc=opt(construct, 'TITLE_LOC'), pad=pad)
    if opt(construct, 'SHOW_STATS'):
        lines = []
        if SHOW_MEDIANS:
            lines.append(f'Median: null {stats["null_median_nm"]:.2f}, exp {stats["exp_median_nm"]:.2f} nm')
        if SHOW_MANNWHITNEY:
            lines.append(f'Mann–Whitney {fmt_p(stats["mannwhitney_p_two_sided"])}')
        if SHOW_KS:
            lines.append(f'KS D = {stats["ks_D"]:.2f}, {fmt_p(stats["ks_p"])}')
        if SHOW_BAND_P and band is not None:
            lines.append(f'Bootstrap band {fmt_p(stats["band_p"], 1 / (N_BAND_DRAWS + 1))}')
        x, y = opt(construct, 'STATS_LOC')
        ax.text(x, y, '\n'.join(lines), transform=ax.transAxes, ha=opt(construct, 'STATS_HA'),
                va=opt(construct, 'STATS_VA'), fontsize=FONT_SIZE_STATS, color=STATS_COLOR)


def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [FONT_FAMILY, 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    run = RUN_FOLDER or latest_run()
    out = os.path.join(run, OUTPUT_SUBFOLDER); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(BAND_SEED)
    panels, rows = [], []
    for construct in CONSTRUCTS:
        null, exp = load(run, construct)
        band = envelope(null, exp.size, exp, rng) if SHOW_NULL_BAND else None
        st = dict(construct=construct, n_null=null.size, n_exp=exp.size,
                  null_median_nm=float(np.median(null)), exp_median_nm=float(np.median(exp)),
                  mannwhitney_p_two_sided=float(mannwhitneyu(exp, null, alternative='two-sided').pvalue),
                  ks_D=float(ks_2samp(exp, null).statistic), ks_p=float(ks_2samp(exp, null).pvalue))
        if band is not None:
            st.update(band_type=BAND_TYPE, band_level=BAND_LEVEL, n_band_draws=N_BAND_DRAWS,
                      band_p=band[3], fraction_of_grid_outside_band=band[4])
        rows.append(st); panels.append((construct, null, exp, band, st))

    n = len(panels); ncol = min(GRID_COLUMNS, n); nrow = int(math.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=FIG_SIZE_GRID_IN, squeeze=False)
    for i, (ax, p) in enumerate(zip(axs.ravel(), panels)):
        draw(ax, *p)
        if PANEL_LETTERS:
            ax.text(-0.18, 1.05, chr(65 + i), transform=ax.transAxes, fontsize=PANEL_LETTER_SIZE, fontweight='bold')
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout()
    for f in FORMATS:
        fig.savefig(os.path.join(out, f'{CONDITION}_radius_cdf_all.{f}'), dpi=DPI)
    plt.close(fig)
    for p in panels:
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_IN); draw(ax, *p); fig.tight_layout()
        for f in FORMATS:
            fig.savefig(os.path.join(out, f'{CONDITION}_radius_cdf_{p[0]}.{f}'), dpi=DPI)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(out, f'{CONDITION}_band_statistics.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'\nFigures: {out}')


if __name__ == '__main__':
    main()
