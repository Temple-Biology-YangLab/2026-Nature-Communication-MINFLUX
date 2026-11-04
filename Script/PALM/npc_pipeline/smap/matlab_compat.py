# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""MATLAB-compatible image filters and helpers."""
import numpy as np
from scipy import ndimage, signal


def fspecial_gaussian(size, sigma):
    """fspecial('gaussian', size, sigma)."""
    if np.isscalar(size):
        size = (int(size), int(size))
    m, n = [(s - 1.0) / 2.0 for s in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    s = h.sum()
    if s != 0:
        h /= s
    return h


def fspecial_disk(radius):
    """fspecial('disk', radius): averaging filter, exact pixel/circle overlap."""
    rad = float(radius)
    crad = int(np.ceil(rad - 0.5))
    x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.maximum(np.abs(x), np.abs(y))
    minxy = np.minimum(np.abs(x), np.abs(y))
    with np.errstate(invalid='ignore'):
        m1 = (rad ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + \
             (rad ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * \
            np.sqrt(np.maximum(rad ** 2 - (maxxy + 0.5) ** 2, 0) + 0j).real
        m2 = (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + \
             (rad ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * \
            np.sqrt(np.maximum(rad ** 2 - (maxxy - 0.5) ** 2, 0) + 0j).real
        sgrid = (rad ** 2 * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) +
                             0.25 * (np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad))))
                 - (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * \
            ((((rad ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2) &
               (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)) |
              ((minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad))))
    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < rad ** 2)
    sgrid[crad, crad] = min(np.pi * rad ** 2, np.pi / 2)
    if crad > 0 and rad > crad - 0.5 and rad ** 2 < (crad - 0.5) ** 2 + 0.25:
        m1 = np.sqrt(rad ** 2 - (crad - 0.5) ** 2)
        m1n = m1 / rad
        sg0 = 2 * (rad ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
        sgrid[2 * crad, crad] = sg0
        sgrid[crad, 2 * crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2 * crad - 1, crad] = sgrid[2 * crad - 1, crad] - sg0
        sgrid[crad, 2 * crad - 1] = sgrid[crad, 2 * crad - 1] - sg0
        sgrid[crad, 1] = sgrid[crad, 1] - sg0
        sgrid[1, crad] = sgrid[1, crad] - sg0
    sgrid[crad, crad] = min(sgrid[crad, crad], 1)
    return sgrid / sgrid.sum()


def filter2_same(h, X):
    """MATLAB filter2(h, X, 'same') == imfilter(X, h) (correlation, zero padding)."""
    h = np.asarray(h, dtype=np.float64)
    kr, kc = h.shape
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        return np.stack([filter2_same(h, x) for x in X])
    if kr * kc > 150:
        full = signal.fftconvolve(X, h[::-1, ::-1], mode='full')
        return full[kr // 2:kr // 2 + X.shape[0], kc // 2:kc // 2 + X.shape[1]]
    origin_r = (kr - 1) // 2 - kr // 2
    origin_c = (kc - 1) // 2 - kc // 2
    return ndimage.correlate(X, h, mode='constant', cval=0.0, origin=(origin_r, origin_c))


def nth_quantile(values, q):
    """myquantilefast: value at sorted index ceil(n*q) (1-based)."""
    v = np.asarray(values).ravel()
    n = v.size
    if n == 0:
        return np.full(np.shape(q), np.nan)
    q = np.atleast_1d(q)
    out = np.empty(q.shape)
    for k, qq in enumerate(q):
        p = int(np.ceil(n * qq))
        p = min(max(p, 1), n)
        out[k] = np.partition(v, p - 1)[p - 1]
    return out


def matlab_round(x):
    """MATLAB round: half away from zero."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def matlab_quantile(x, p):
    """MATLAB quantile (Statistics Toolbox, 'method 5' piecewise linear)."""
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    x = x[~np.isnan(x)]
    n = x.size
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if n == 0:
        return np.full(p.shape, np.nan)
    pos = n * p + 0.5  # 1-based
    pos = np.clip(pos, 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n)
    frac = pos - lo
    return x[lo - 1] * (1 - frac) + x[hi - 1] * frac


def histcounts(x, edges):
    """MATLAB histcounts with explicit edges (last bin closed)."""
    h, _ = np.histogram(x, bins=edges)
    return h
