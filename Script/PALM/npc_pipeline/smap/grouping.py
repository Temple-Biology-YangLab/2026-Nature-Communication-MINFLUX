# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Grouping of localizations in consecutive frames and render-layer filters (SMAP)."""
import numpy as np
from numba import njit


@njit(cache=True)
def _connect_single(x, y, frames, dX, dT):
    """Port of connectsingle2c.c (cs). Inputs sorted by (frame, x)."""
    lenx = x.shape[0]
    lst = np.zeros(lenx)
    thisentry = 0
    particlenumber = 0
    while thisentry <= lenx - 1:
        while thisentry <= lenx - 1 and lst[thisentry] > 0:
            thisentry += 1
        if thisentry == lenx:
            break
        particlenumber += 1
        xh = x[thisentry]
        yh = y[thisentry]
        frh = frames[thisentry]
        lst[thisentry] = particlenumber
        numdark = 0
        testentry = thisentry
        while testentry <= lenx - 1 and numdark <= dT and frames[testentry] >= frh:
            particlefound = 0
            while testentry <= lenx - 1 and frames[testentry] == frh:
                testentry += 1
            frtest = frh + 1
            while testentry <= lenx - 1 and x[testentry] < xh - dX and frames[testentry] == frtest:
                testentry += 1
            while testentry <= lenx - 1 and x[testentry] < xh + dX and frames[testentry] == frtest:
                if y[testentry] > yh - dX and y[testentry] < yh + dX and lst[testentry] == 0:
                    particlefound = 1
                    lst[testentry] = particlenumber
                    xh = (x[testentry] + xh) / 2
                    yh = (y[testentry] + yh) / 2
                    frh = frames[testentry]
                    numdark = 0
                    break
                else:
                    testentry += 1
            if particlefound == 0:
                frh = frtest
                numdark += 1
    return lst


COMBINE_MODES = {
    'channel': 'first', 'color': 'mean', 'LLsecond': 'min',
    'xnm': 'mean', 'ynm': 'mean', 'znm': 'mean', 'xpix': 'mean', 'ypix': 'mean', 'zpix': 'mean',
    'bg': 'sum', 'bg1': 'sum', 'bg2': 'sum', 'phot': 'sum', 'phot1': 'sum', 'phot2': 'sum',
    'PSFxnm': 'mean', 'PSFynm': 'mean', 'PSFxpix': 'mean', 'PSFypix': 'mean',
    'locprecnm': 'locp', 'locprecxnm': 'locp', 'locprecynm': 'locp', 'locprecznm': 'locp',
    'CRLBphot': 'min', 'frame': 'min', 'logLikelihood': 'max', 'loglikelihood': 'max',
    'LLrel': 'max', 'numberInGroup': 'first', 'groupindex': 'first', 'filenumber': 'first',
    'clusterdensity': 'mean', 'cellnumbers': 'max', 'sitenumbers': 'max',
}


def group_localizations(loc, dx=35.0, dt=1):
    """SMAP LocalizationData.regroup with group_mode 'fix' (defaults dX=35 nm, dT=1)."""
    n = len(loc['xnm'])
    if n == 0:
        return {k: v[:0] for k, v in loc.items()}, np.zeros(0), np.zeros(0)
    frame = np.asarray(loc['frame'], dtype=np.float64)
    x = np.asarray(loc['xnm'], dtype=np.float64)
    y = np.asarray(loc['ynm'], dtype=np.float64)
    fn = np.asarray(loc.get('filenumber', np.ones(n)), dtype=np.float64)
    ch = np.asarray(loc.get('channel', np.zeros(n)), dtype=np.float64)
    order = np.lexsort((x, frame, ch, fn))  # sortrows by filenumber, channel, frame, x
    fs = frame[order].copy()
    newgroup = (np.diff(fn[order]) != 0) | (np.diff(ch[order]) != 0)
    fng = np.nonzero(newgroup)[0]
    if fng.size:
        if fng[0] == 0:
            fng = fng[1:]
        newgroup[fng - 1] = False
        fs[:-1][newgroup] = 0
    lst = _connect_single(x[order], y[order], fs, float(dx), int(dt))
    if lst[-1] == 0:
        lst[-1] = lst.max() + 1
    if lst[0] == 0:
        lst[0] = 1
    groupindex = np.empty(n)
    groupindex[order] = lst
    uniq, inv, counts = np.unique(groupindex, return_inverse=True, return_counts=True)
    number_in_group = counts[inv].astype(np.float32)
    grouploc = combine_groups(loc, groupindex, inv, len(uniq))
    grouploc['numberInGroup'] = counts.astype(np.float32)
    grouploc['groupindex'] = uniq
    return grouploc, groupindex, number_in_group


def combine_groups(loc, groupindex, inv, ngroups):
    """Grouper.combine for all fields."""
    n = len(groupindex)
    if 'locprecnm' in loc:
        with np.errstate(divide='ignore'):
            w = 1.0 / np.asarray(loc['locprecnm'], dtype=np.float64) ** 2
    elif 'xnmerr' in loc:
        w = 1.0 / np.asarray(loc['xnmerr'], dtype=np.float64) ** 2
    else:
        w = np.asarray(loc['phot'], dtype=np.float64)
    w[np.isinf(w)] = 1
    gw = np.bincount(inv, weights=w, minlength=ngroups)
    last_idx = np.zeros(ngroups, dtype=np.int64)
    np.maximum.at(last_idx, inv, np.arange(n))
    out = {}
    for k, v in loc.items():
        v = np.asarray(v)
        if v.ndim != 1 or v.shape[0] != n:
            continue
        mode = COMBINE_MODES.get(k, 'locp' if k.endswith('err') else 'mean')
        vd = v.astype(np.float64)
        if mode == 'sum':
            r = np.bincount(inv, weights=vd, minlength=ngroups)
        elif mode == 'mean':
            r = np.bincount(inv, weights=vd * w, minlength=ngroups) / gw
        elif mode == 'locp':
            with np.errstate(divide='ignore'):
                r = 1.0 / np.sqrt(np.bincount(inv, weights=1.0 / vd ** 2, minlength=ngroups))
        elif mode == 'min':
            r = np.full(ngroups, np.inf)
            np.minimum.at(r, inv, vd)
        elif mode == 'max':
            r = np.full(ngroups, -np.inf)
            np.maximum.at(r, inv, vd)
        else:  # first
            r = vd[last_idx]
        out[k] = r.astype(v.dtype) if v.dtype.kind in 'fiu' else r
    return out


class Layer:
    """A SMAP render layer: field filters + grouped/ungrouped switch."""

    def __init__(self, filters=None, grouped=True):
        self.filters = dict(filters or {})
        self.grouped = grouped

    def mask(self, loc):
        n = len(next(iter(loc.values())))
        m = np.ones(n, dtype=bool)
        for k, (lo, hi) in self.filters.items():
            if k not in loc:
                continue
            v = np.asarray(loc[k])
            m &= (v >= lo) & (v <= hi)
        return m


DEFAULT_LAYER_FILTERS = {'locprecnm': (0, 30), 'PSFxnm': (0, 175), 'frame': (0, np.inf)}


class LocData:
    """Container for ungrouped (loc) and grouped (grouploc) localizations."""

    def __init__(self, loc, group_dx=35.0, group_dt=1, info=None, grouploc=None):
        self.loc = {k: np.asarray(v) for k, v in loc.items()}
        self.info = info or {}
        self.group_dx = group_dx
        self.group_dt = group_dt
        if grouploc is None:
            self.regroup()
        else:
            self.grouploc = {k: np.asarray(v) for k, v in grouploc.items()}
            self._cache = {}

    def regroup(self):
        g, gi, nig = group_localizations(self.loc, self.group_dx, self.group_dt)
        self.loc['groupindex'] = gi
        self.loc['numberInGroup'] = nig
        self.grouploc = g
        self._cache = {}

    def get(self, layer: Layer = None, grouped=None, fields=None):
        """Localizations passing a layer's filters (grouped or not)."""
        if grouped is None:
            grouped = layer.grouped if layer is not None else False
        key = (grouped, None if layer is None else tuple(sorted(layer.filters.items())))
        cache = getattr(self, '_cache', None)
        if cache is None:
            cache = self._cache = {}
        if key not in cache:
            src = self.grouploc if grouped else self.loc
            m = layer.mask(src) if layer is not None else np.ones(len(src['xnm']), bool)
            cache[key] = {k: v[m] for k, v in src.items()}
        sel = cache[key]
        keys = fields if fields is not None else sel.keys()
        return {k: sel[k] for k in keys if k in sel}

    def invalidate(self):
        """Call after modifying loc/grouploc in place."""
        self._cache = {}
