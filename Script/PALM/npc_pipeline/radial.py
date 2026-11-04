# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Radial density profiles about pore centres."""
import numpy as np
from scipy.optimize import curve_fit


def radial_distances(x, y, centres, r_keep):
    """Distances of points to their nearest assigned centre, within r_keep."""
    out = []
    for c in centres:
        cx, cy = (c['x0'], c['y0']) if isinstance(c, dict) else (c[0], c[1])
        m = (np.abs(x - cx) < r_keep) & (np.abs(y - cy) < r_keep)
        if not m.any():
            continue
        d = np.hypot(x[m] - cx, y[m] - cy)
        out.append(d[d <= r_keep])
    return np.concatenate(out) if out else np.array([])


def radial_density(r, cfg):
    """Area-corrected radial density. Returns (centres, density, counts)."""
    h, e = np.histogram(r, bins=cfg.n_bins, range=(0, cfg.r_keep_nm))
    area = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    return 0.5 * (e[1:] + e[:-1]), h / area, h


def fit_peak(r, cfg):
    """Gaussian fit to the radial profile."""
    ctr, dens, cnt = radial_density(r, cfg)
    y = dens if cfg.use_density else cnt.astype(float)
    if y.max() <= 0:
        return None
    def g(xx, a, mu, sg):
        return a * np.exp(-(xx - mu) ** 2 / (2 * sg ** 2))
    p0 = [y.max(), ctr[np.argmax(y)], 15.0]
    lo = [0.0, cfg.fit_center_bounds_nm[0], cfg.fit_sigma_bounds_nm[0]]
    hi = [y.max() * 5 + 1e-9, cfg.fit_center_bounds_nm[1], cfg.fit_sigma_bounds_nm[1]]
    try:
        popt, pcov = curve_fit(g, ctr, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    except Exception:
        return None
    resid = y - g(ctr, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    err = np.sqrt(np.diag(pcov))
    return dict(peak_nm=float(popt[1]), peak_err_nm=float(err[1]),
                sigma_nm=float(popt[2]), amp=float(popt[0]),
                R2=float(1 - ss_res / ss_tot) if ss_tot else float('nan'),
                n=int(len(r)),
                centre_depletion=float(y[ctr < 0.3 * popt[1]].mean() / y.max())
                if (ctr < 0.3 * popt[1]).any() else float('nan'))
