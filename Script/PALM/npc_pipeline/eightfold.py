# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-pore 8-fold rotational phase estimation."""
import numpy as np


def angular_order(x, y, cx=0.0, cy=0.0, fold=8, r_min=None, r_max=None):
    """Complex order parameter for `fold`-fold symmetry."""
    x = np.asarray(x, float) - cx
    y = np.asarray(y, float) - cy
    r = np.hypot(x, y)
    m = np.ones(len(x), bool)
    if r_min is not None:
        m &= r >= r_min
    if r_max is not None:
        m &= r <= r_max
    m &= r > 1e-9
    if m.sum() < 3:
        return 0.0, 0.0, int(m.sum())
    th = np.arctan2(y[m], x[m])
    z = np.exp(1j * fold * th).mean()
    return float(np.abs(z)), float(np.angle(z) / fold), int(m.sum())


def expected_null_order(n, fold=8):
    """Expected order parameter for n uniformly distributed angles."""
    n = np.asarray(n, float)
    return np.where(n > 0, np.sqrt(np.pi) / (2 * np.sqrt(np.maximum(n, 1))), np.nan)


def estimate_phase(x, y, cx, cy, fold=8, r_min=30.0, r_max=70.0):
    """Per-pore 8-fold phase from the reference channel."""
    mag, ph, n = angular_order(x, y, cx, cy, fold, r_min, r_max)
    return dict(order=mag, phase=ph, n=n,
                null_order=float(expected_null_order(n, fold)),
                order_ratio=mag / expected_null_order(n, fold) if n > 0 else np.nan)


def estimate_phase_lsq(x, y, cx=0.0, cy=0.0, fold=8, r_min=30.0, r_max=70.0):
    """8-fold rotation by least-squares phase estimation."""
    x = np.asarray(x, float) - cx
    y = np.asarray(y, float) - cy
    r = np.hypot(x, y)
    m = (r >= r_min) & (r <= r_max)
    if m.sum() < 3:
        return dict(phase=0.0, n=int(m.sum()), rmsd_rad=np.nan)
    phi = np.arctan2(y[m], x[m])
    period = 2 * np.pi / fold
    half = period / 2.0

    def cost(prot):
        d = ((phi - prot + half) % period) - half
        return np.sum(d ** 2)

    grid = np.linspace(0, period, 361)
    c = np.array([cost(g) for g in grid])
    p0 = grid[np.argmin(c)]
    from scipy.optimize import minimize_scalar
    r_ = minimize_scalar(cost, bounds=(p0 - period / 20, p0 + period / 20),
                         method='bounded')
    prot = r_.x % period
    d = ((phi - prot + half) % period) - half
    return dict(phase=float(prot), n=int(m.sum()),
                rmsd_rad=float(np.sqrt(np.mean(d ** 2))))


def rotate(x, y, cx, cy, angle_rad):
    """Rotate points about (cx, cy) by -angle so the phase lands at 0."""
    xr = np.asarray(x, float) - cx
    yr = np.asarray(y, float) - cy
    c, s = np.cos(-angle_rad), np.sin(-angle_rad)
    return c * xr - s * yr, s * xr + c * yr


def rotate_and_merge(pores, ref_xy, target_xy, fold=8, r_min=25.0, r_max=90.0,
                     angle_from='reference'):
    """Centre and rotationally align every pore, then pool."""
    dxs, dys, phases, orders, nulls, ns, pore_id = [], [], [], [], [], [], []
    kept = 0          # index into the OUTPUT arrays, not into `pores`
    for i, p in enumerate(pores):
        cx, cy = p['x0'], p['y0']
        rx, ry = ref_xy(p)
        if rx is None or len(rx) < 3:
            continue
        src_x, src_y = (rx, ry) if angle_from == 'self' else (rx, ry)
        est = estimate_phase(src_x, src_y, cx, cy, fold, r_min, r_max)
        tx, ty = target_xy(p)
        if tx is None or len(tx) == 0:
            continue
        ax, ay = rotate(tx, ty, cx, cy, est['phase'])
        dxs.append(ax); dys.append(ay)
        phases.append(est['phase']); orders.append(est['order'])
        nulls.append(est['null_order']); ns.append(est['n'])
        pore_id.append(np.full(len(ax), kept, float))
        kept += 1
    if not dxs:
        return dict(dx=np.array([]), dy=np.array([]), n_pores=0)
    return dict(dx=np.concatenate(dxs), dy=np.concatenate(dys),
                pore_id=np.concatenate(pore_id),
                phase=np.array(phases), order=np.array(orders),
                null_order=np.array(nulls), n_ref=np.array(ns),
                n_pores=len(phases), fold=fold,
                circular=(angle_from == 'self'),
                angle_from=angle_from)


def merged_order(dx, dy, fold=8, r_min=25.0, r_max=90.0):
    """Order parameter of an already-merged, already-rotated cloud."""
    mag, ph, n = angular_order(dx, dy, 0.0, 0.0, fold, r_min, r_max)
    return dict(order=mag, phase=ph, n=n,
                null_order=float(expected_null_order(n, fold)),
                order_ratio=mag / expected_null_order(n, fold) if n else np.nan)


def null_test(pores, ref_xy, target_xy, n_perm=200, fold=8, r_min=25.0,
              r_max=90.0, seed=0):
    """Permutation null: shuffle which pore's angle is applied to which pore."""
    rng = np.random.default_rng(seed)
    obs = rotate_and_merge(pores, ref_xy, target_xy, fold, r_min, r_max)
    if obs['n_pores'] == 0:
        return dict(n_pores=0)
    o = merged_order(obs['dx'], obs['dy'], fold, r_min, r_max)['order']

    cache = []
    for p in pores:
        rx, ry = ref_xy(p)
        tx, ty = target_xy(p)
        if rx is None or len(rx) < 3 or tx is None or len(tx) == 0:
            continue
        est = estimate_phase(rx, ry, p['x0'], p['y0'], fold, r_min, r_max)
        cache.append((p['x0'], p['y0'], np.asarray(tx), np.asarray(ty), est['phase']))
    if len(cache) < 3:
        return dict(n_pores=len(cache))
    phases = np.array([c[4] for c in cache])

    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(cache))
        X, Y = [], []
        for k, (cx, cy, tx, ty, _) in enumerate(cache):
            ax, ay = rotate(tx, ty, cx, cy, phases[perm[k]])
            X.append(ax); Y.append(ay)
        null.append(merged_order(np.concatenate(X), np.concatenate(Y),
                                 fold, r_min, r_max)['order'])
    null = np.array(null)
    sd = null.std()
    return dict(observed=float(o), null_mean=float(null.mean()),
                null_sd=float(sd),
                z=float((o - null.mean()) / sd) if sd > 0 else np.nan,
                p_value=float((np.sum(null >= o) + 1) / (n_perm + 1)),
                n_pores=len(cache), n_perm=n_perm, fold=fold)
