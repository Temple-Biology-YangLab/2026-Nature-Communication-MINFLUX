# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read SMAP _sml.mat files: localizations and the stored drift curve."""
import os
import glob
import numpy as np

try:
    import h5py
except ImportError:      # pragma: no cover
    h5py = None

LOC_PATH = ('saveloc', 'loc')
DRIFT_PATH = ('saveloc', 'file', 'driftinfo', 'xy')


class Localizations:
    """Thin column store. Keys are SMAP field names (xnm, ynm, frame, ...)."""

    def __init__(self, cols, source=''):
        self.cols = cols
        self.source = source

    def __contains__(self, k):
        return k in self.cols

    def __getitem__(self, k):
        return self.cols[k]

    def get(self, k, default=None):
        return self.cols.get(k, default)

    def keys(self):
        return self.cols.keys()

    def __len__(self):
        return len(next(iter(self.cols.values()))) if self.cols else 0

    def subset(self, mask):
        return Localizations({k: v[mask] for k, v in self.cols.items()}, self.source)

    def copy_with(self, **kw):
        c = dict(self.cols)
        c.update(kw)
        return Localizations(c, self.source)

    def __repr__(self):
        return f'<Localizations n={len(self)} fields={sorted(self.cols)[:6]}... from {os.path.basename(self.source)}>'


def _grp(h, path):
    g = h
    for p in path:
        if p not in g:
            return None
        g = g[p]
    return g


def read_sml(path, fields=None):
    """Read the localization table of a SMAP _sml.mat (MATLAB v7 or v7.3) as Localizations.

    fields=None keeps every 1-D numeric column."""
    from .smap.io import read_sml as _read
    loc = _read(path)['loc']
    keys = list(loc) if fields is None else [f for f in fields if f in loc]
    cols = {}
    for k in keys:
        a = np.asarray(loc[k]).ravel()
        if a.dtype.kind in 'fiub':
            cols[k] = a.astype(float)
    n = {len(v) for v in cols.values()}
    if len(n) > 1:
        from collections import Counter
        good = Counter(len(v) for v in cols.values()).most_common(1)[0][0]
        cols = {k: v for k, v in cols.items() if len(v) == good}
    return Localizations(cols, path)


def read_driftinfo(path):
    """Return SMAP's stored per-frame drift vector, or None if absent."""
    if h5py is None:
        raise ImportError('h5py is required to read SMAP .mat files')
    with h5py.File(path, 'r') as h:
        xy = _grp(h, DRIFT_PATH)
        if xy is None or 'dxt' not in xy:
            return None
        out = {}
        for k in ('dxt', 'dyt', 'dx', 'dy', 'binframes'):
            if k in xy:
                out[k] = np.array(xy[k]).ravel().astype(float)
    return out


def find_sml(folder, prefer_driftc=True, channel='R'):
    """Locate the sml file for a channel."""
    pats = [f'*{channel}_driftc_sml.mat', f'*-{channel}_driftc_sml.mat',
            f'*{channel}_sml.mat', f'*-{channel}_sml.mat']
    if not prefer_driftc:
        pats = pats[2:] + pats[:2]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(folder, p)))
        if hits:
            return hits[0]
    return None


