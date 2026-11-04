# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian rendering of localizations as in SMAP."""
import numpy as np
from numba import njit

from .matlab_compat import nth_quantile

ROIKS = 2.7
_G_SIZE = 600
_G_SIGMA = _G_SIZE / (2 * ROIKS) / 2


def _gauss_template():
    xg = np.arange(-_G_SIZE, _G_SIZE + 1)
    Xg, Yg = np.meshgrid(xg, xg)
    return np.exp(-(Xg ** 2 + Yg ** 2) / 2 / _G_SIGMA ** 2).astype(np.float32)


_TEMPLATE = None


def _erf(x):
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    s = np.sign(x)
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    return s * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x))


@njit(cache=True)
def _erf_nb(x):
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    sign = 1.0
    if x < 0:
        sign = -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    return sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x))


@njit(cache=True)
def _gaussrender_c(xpix, ypix, sx0, sx1, sigma, G, Gsigma, roiks, N, uselut, c, lut, c0, c1, ncol):
    """gaussrenderc.c, with its single-precision (float) arithmetic reproduced."""
    f1 = np.float32(1.0)
    Gx = G.shape[0]
    Gs = np.float32((Gx - 1) // 2)
    sl = lut.shape[0]
    slm1 = np.float32(sl - 1)
    img = np.zeros((ncol, sx0, sx1), dtype=np.float32)
    nlocs = 0
    for k in range(xpix.shape[0]):
        sk = sigma[k]
        dn = int(roiks * sk + f1)            # float -> long: truncation
        xr = int(np.float64(xpix[k]) + 0.5)  # float + double -> long
        yr = int(np.float64(ypix[k]) + 0.5)
        if xr >= -dn and xr < sx0 + dn and yr >= -dn and yr < sx1 + dn:
            if xr >= 0 and xr < sx0 and yr >= 0 and yr < sx1:
                nlocs += 1
            dx = xpix[k] - np.float32(xr)
            dy = ypix[k] - np.float32(yr)
            ic = _erf_nb((dn + 0.5) / np.float64(sk) / 1.4142135624)
            gnorm = np.float32(np.float64(N[k]) / (2 * 3.1415926536 * np.float64(sk) * np.float64(sk) * ic * ic))
            indc = 0
            if uselut:
                indc = int((c[k] - c0) / (c1 - c0) * slm1)
            for xax in range(-dn, dn + 1):
                xtf = np.float64((np.float32(xax) - dx) * Gsigma / sk + Gs) + 0.5
                xt = int(xtf)                  # truncation toward zero; (-1, 0) -> 0 as in C
                if xt < 0:
                    continue
                for yax in range(-dn, dn + 1):
                    ytf = np.float64((np.float32(yax) - dy) * Gsigma / sk + Gs) + 0.5
                    yt = int(ytf)
                    if yt < 0:
                        continue
                    xp = xr + xax
                    yp = yr + yax
                    if xp >= 0 and xp < sx0 and yp >= 0 and yp < sx1 and xt < Gx and yt < Gx:
                        v = G[yt, xt] * gnorm  # template[xt + yt*Gx] (MATLAB col-major), float
                        if uselut:
                            if indc >= 0 and indc < sl:
                                for col in range(3):
                                    img[col, xp, yp] += v * lut[indc, col]
                        else:
                            img[0, xp, yp] += v
    return img, nlocs


def _quantile_sub(v, q, maxnum):
    """myquantilefast(in, q, maxnum): every dn-th element if maxnum < 0.75 n."""
    v = np.asarray(v).ravel()
    if maxnum < v.size * 0.75:
        v = v[::int(np.ceil(v.size / maxnum))]
    return nth_quantile(v, q)


def jet_lut(n=256):
    """MATLAB jet(n)."""
    m = n
    nn = int(np.ceil(m / 4))
    u = np.concatenate([np.arange(1, nn + 1) / nn, np.ones(nn - 1), np.arange(nn, 0, -1) / nn])
    g = int(np.ceil(nn / 2)) - (1 if m % 4 == 1 else 0) + np.arange(1, len(u) + 1)
    r = g + nn
    b = g - nn
    g = g[g <= m]
    r = r[r <= m]
    b = b[b >= 1]
    J = np.zeros((m, 3))
    J[r - 1, 0] = u[:len(r)]
    J[g - 1, 1] = u[:len(g)]
    J[b - 1, 2] = u[-len(b):]
    return J


def hot_lut(m=256):
    """MATLAB hot(m) (SMAP LUT 'red hot')."""
    n = int(np.fix(3 / 8 * m))
    r = np.r_[np.arange(1, n + 1) / n, np.ones(m - n)]
    g = np.r_[np.zeros(n), np.arange(1, n + 1) / n, np.ones(m - 2 * n)]
    b = np.r_[np.zeros(2 * n), np.arange(1, m - 2 * n + 1) / (m - 2 * n)]
    return np.c_[r, g, b]


LUTS = {'jet': jet_lut, 'red hot': hot_lut}


def make_lut(name):
    """SMAP mymakelut for the ported LUTs ('jet', 'red hot')."""
    if name not in LUTS:
        raise ValueError('LUT %r not ported (available: %s)' % (name, ', '.join(LUTS)))
    return LUTS[name](256)


def colorfield_range(v):
    """SMAP 'auto' colour range of a render field (GuiChannel.renderfield_callback)."""
    v = np.asarray(v, float)
    q0 = nth_quantile(v, [0.001, 0.999])
    vv = v[(v > q0[0]) & (v < q0[1])]
    q = nth_quantile(vv, [0.01, 0.99]) if vv.size else np.array([np.nan, np.nan])
    if vv.size == 0 or q[1] - q[0] == 0:
        q = np.array([v.min(), v.max()])
    dx = 10 ** np.floor(np.log10(abs(q[1]) / 100))
    return np.round(q[0] / dx) * dx, np.round(q[1] / dx) * dx


def render(locs, rangex, rangey, pixrec, mode='gauss', gaussfac=0.4, mingausspix=0.7,
           mingaussnm=3.0, colorfield=None, crange=None, lut=None):
    """renderSMAP for a set of localizations."""
    global _TEMPLATE
    pr = float(pixrec)
    rx = np.array([rangex[0] - pr / 2, rangex[1] - pr / 2])
    ry = np.array([rangey[0] - pr / 2, rangey[1] - pr / 2])
    f32 = np.float32
    x = np.asarray(locs['xnm'], dtype=np.float32)
    y = np.asarray(locs['ynm'], dtype=np.float32)
    sx0 = int(np.floor((rx[1] - rx[0]) / pr + 0.5))
    sx1 = int(np.floor((ry[1] - ry[0]) / pr + 0.5))
    xpix = ((x - f32(rx[0])) / f32(pr) - f32(1)).astype(np.float32)
    ypix = ((y - f32(ry[0])) / f32(pr) - f32(1)).astype(np.float32)
    N = np.ones(len(x), dtype=np.float32)
    uselut = colorfield is not None
    if uselut:
        lut = jet_lut(256) if lut is None else (make_lut(lut) if isinstance(lut, str) else lut)
        c = np.asarray(locs[colorfield], dtype=np.float32)
        if crange is None:
            crange = colorfield_range(c)
        c = np.minimum(np.maximum(c, f32(crange[0])), f32(crange[1]))  # 'restrict' unchecked: clamp
        ncol = 3
    else:
        lut = np.zeros((1, 3))
        c = np.zeros(len(x), dtype=np.float32)
        crange = (0.0, 0.0)
        ncol = 1
    if mode == 'hist':
        img = np.zeros((ncol, sx0, sx1), np.float32)
        xr = np.trunc(xpix + 0.5).astype(int)
        yr = np.trunc(ypix + 0.5).astype(int)
        ok = (xr >= 0) & (xr < sx0) & (yr >= 0) & (yr < sx1)
        if uselut:
            indc = ((c - crange[0]) / (crange[1] - crange[0]) * (len(lut) - 1)).astype(int)
            for col in range(3):
                np.add.at(img[col], (xr[ok], yr[ok]), lut[indc[ok], col])
        else:
            np.add.at(img[0], (xr[ok], yr[ok]), 1)
        nlocs = int(ok.sum())
    else:
        if _TEMPLATE is None:
            _TEMPLATE = _gauss_template()
        smin = f32(max(mingaussnm, mingausspix * pr))
        if 'locprecnm' in locs and len(locs['locprecnm']):
            sd = np.asarray(locs['locprecnm'], dtype=np.float32)
            s = np.minimum(np.maximum(sd * f32(gaussfac), smin), f32(400))
        else:
            s = np.full(len(x), smin, dtype=np.float32)
        spix = (s / f32(pr)).astype(np.float32)
        if len(spix):
            spixmed = min(f32(_quantile_sub(spix[~np.isnan(spix)], 0.5, 1000)[0]) * f32(10), f32(200))
            spix[spix > spixmed] = spixmed
        img, nlocs = _gaussrender_c(xpix, ypix, sx0, sx1, spix, _TEMPLATE, np.float32(_G_SIGMA),
                                    np.float32(ROIKS), N, uselut, c, np.asarray(lut, np.float32),
                                    np.float32(crange[0]), np.float32(crange[1]), ncol)
    image = np.transpose(img, (2, 1, 0))  # -> [y, x, col]
    if not uselut:
        image = image[:, :, 0]
    return {'image': image, 'rangex': rx, 'rangey': ry, 'numberOfLocs': nlocs}
