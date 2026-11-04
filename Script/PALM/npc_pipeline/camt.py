# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Load CamT localizations (SMAP sml, GDSC single points, GDSC traces) in nm on the SMAP grid.
"""
import os
import re

import numpy as np

from .io_smap import Localizations, read_sml


GDSC_PIXEL_OFFSET = 0.5      # GDSC pixel-corner -> SMAP pixel-centre
DEFAULT_NM_PER_PX = 97.0


def _gdsc_header_line(path):
    """Index of the '#Id...' column header line."""
    with open(path, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i > 400:
                break
            if line.startswith('#Id') or '\tX (px)\t' in line:
                return i
    return None


def _read_gdsc_table(path, sep=None):
    """Read a GDSC results table, ignoring interleaved '#' comment blocks."""
    import pandas as pd
    hdr = _gdsc_header_line(path)
    if sep is None:
        sep = '\t' if path.lower().endswith(('.xls', '.txt')) else ','
    if hdr is None:
        df = pd.read_csv(path, sep=sep, comment='#', engine='python')
    else:
        with open(path, encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i == hdr:
                    cols = [c.lstrip('#').strip() for c in line.rstrip('\n').split(sep)]
                    break
        seen, uniq = {}, []
        for j, c in enumerate(cols):
            if c in ('+/-', ''):
                c = (uniq[-1] + ' err') if uniq else f'col{j} err'
            if c in seen:
                seen[c] += 1
                c = f'{c}.{seen[c]}'
            else:
                seen[c] = 0
            uniq.append(c)
        cols = uniq
        df = pd.read_csv(path, sep=sep, skiprows=hdr + 1, comment='#',
                         names=cols, engine='python')
    df = df.dropna(subset=[c for c in df.columns if c.startswith('X (')][:1])
    return df


def read_gdsc(path, nm_per_px=DEFAULT_NM_PER_PX, pixel_offset=GDSC_PIXEL_OFFSET,
              roi_origin_px=(0.0, 0.0)):
    """Read a GDSC file (single-point or trace) into CamT nm coordinates."""
    df = _read_gdsc_table(path)
    xc = [c for c in df.columns if c.startswith('X (px)')][0]
    yc = [c for c in df.columns if c.startswith('Y (px)')][0]
    fc = 'Frame' if 'Frame' in df.columns else ('T' if 'T' in df.columns else None)
    if fc is None:
        raise ValueError(f'{path}: no Frame/T column')

    x = (df[xc].to_numpy(float) + pixel_offset + roi_origin_px[0]) * nm_per_px
    y = (df[yc].to_numpy(float) + pixel_offset + roi_origin_px[1]) * nm_per_px
    cols = {'xnm': x, 'ynm': y, 'frame': df[fc].to_numpy(float)}

    if 'Precision (nm)' in df.columns:
        cols['locprecnm'] = df['Precision (nm)'].to_numpy(float)
    if 'Id' in df.columns:
        cols['trace_id'] = df['Id'].to_numpy(float)
    for src, dst in (('Intensity (photon)', 'phot'), ('S (px)', 'PSFxnm')):
        if src in df.columns:
            v = df[src].to_numpy(float)
            cols[dst] = v * nm_per_px if src.endswith('(px)') else v
    return Localizations(cols, path)


SOURCES = {
    'smap':   'ROI-T_sml.mat',
    'gdsc':   'Single_Point_CamT.csv',
    'traces': 'Trace_CamT.xls',
}

IGNORED_EXPORTS = ('Single_Point_PALM.csv', 'Single_NPC_CamR.csv',
                   'Nano-2D1_stable.csv', 'Nano-2D2_stable.csv',
                   'Nano-2D1_drifted.csv', 'Nano-2D2_drifted.csv')


def attach_trace_precision(traces, single_point, tol_nm=1.0):
    """Copy per-localization precision from the single-point export onto trace
    localizations, matched by frame and coordinate.
    """
    if 'locprecnm' in traces or 'locprecnm' not in single_point:
        return traces
    from scipy.spatial import cKDTree
    sx = single_point['xnm']; sy = single_point['ynm']
    sf = single_point['frame']; sp = single_point['locprecnm']
    tx = traces['xnm']; ty = traces['ynm']; tf = traces['frame']
    prec = np.full(len(tx), np.nan)
    import collections
    by_frame = collections.defaultdict(list)
    for i, f in enumerate(sf):
        by_frame[f].append(i)
    for j in range(len(tx)):
        idx = by_frame.get(tf[j])
        if not idx:
            continue
        idx = np.asarray(idx)
        d = np.hypot(sx[idx] - tx[j], sy[idx] - ty[j])
        k = int(np.argmin(d))
        if d[k] <= tol_nm:
            prec[j] = sp[idx[k]]
    return traces.copy_with(locprecnm=prec)


def load_camt(cell_dir, which='all', nm_per_px=DEFAULT_NM_PER_PX,
              pixel_offset=GDSC_PIXEL_OFFSET):
    """Load CamT data in native nm (pre-registration). Returns dict of name -> Localizations.
    """
    out = {}
    names = SOURCES if which == 'all' else {which: SOURCES[which]}
    for name, fn in names.items():
        p = os.path.join(cell_dir, fn)
        if not os.path.exists(p):
            continue
        out[name] = (read_sml(p) if name == 'smap'
                     else read_gdsc(p, nm_per_px, pixel_offset))
    if 'traces' in out and 'gdsc' in out:
        out['traces'] = attach_trace_precision(out['traces'], out['gdsc'])
    return out


def to_camr_frame(locs, transform=None, drift_curve=None):
    """Map CamT nm -> CamR nm, then subtract drift. Order matters."""
    x, y = np.asarray(locs['xnm'], float), np.asarray(locs['ynm'], float)
    if transform is not None:
        x, y = transform.apply(x, y)
    if drift_curve is not None:
        if 'frame' not in locs:
            raise ValueError('drift correction needs a frame column')
        x, y = drift_curve.apply(x, y, locs['frame'])
    return locs.copy_with(xnm=x, ynm=y)


def prepare_camt(cell_dir, transform=None, drift_curve=None, which='all', **kw):
    """load_camt + to_camr_frame for every available source."""
    return {k: to_camr_frame(v, transform, drift_curve)
            for k, v in load_camt(cell_dir, which=which, **kw).items()}


def check_pixel_offset(cell_dir, nm_per_px=DEFAULT_NM_PER_PX, n_frames=40,
                       offsets=(0.0, 0.5, -0.5)):
    """Cross-match GDSC against SMAP on the same frames to confirm the offset."""
    from scipy.spatial import cKDTree
    sm = read_sml(os.path.join(cell_dir, SOURCES['smap']),
                  fields=['xnm', 'ynm', 'frame'])
    gp = os.path.join(cell_dir, SOURCES['gdsc'])
    df = _read_gdsc_table(gp)
    xc = [c for c in df.columns if c.startswith('X (px)')][0]
    yc = [c for c in df.columns if c.startswith('Y (px)')][0]
    fc = 'Frame' if 'Frame' in df.columns else 'T'
    gf = df[fc].to_numpy(float)
    sx, sy, sf = sm['xnm'], sm['ynm'], sm['frame']
    res = {}
    for off in offsets:
        gx = (df[xc].to_numpy(float) + off) * nm_per_px
        gy = (df[yc].to_numpy(float) + off) * nm_per_px
        d = []
        for fr in np.unique(gf)[:n_frames]:
            m, s = gf == fr, sf == fr
            if not m.any() or not s.any():
                continue
            t = cKDTree(np.c_[sx[s], sy[s]])
            d += list(t.query(np.c_[gx[m], gy[m]], k=1)[0])
        res[off] = float(np.median(d)) if d else float('nan')
    return res
