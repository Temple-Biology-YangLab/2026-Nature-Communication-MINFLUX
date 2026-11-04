# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nelder-Mead simplex (fminsearch) and bounded variant (fminsearchbnd)."""
import numpy as np


def fminsearch(fun, x0, tolx=1e-4, tolfun=1e-4, maxfunevals=None, maxiter=None):
    """Exact port of MATLAB fminsearch (default options)."""
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    if maxfunevals is None:
        maxfunevals = 200 * n
    if maxiter is None:
        maxiter = 200 * n
    if n == 0:
        return x0, fun(x0), 0
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    v = np.zeros((n, n + 1))
    fv = np.zeros(n + 1)
    v[:, 0] = x0
    fv[0] = fun(x0)
    for j in range(n):
        y = x0.copy()
        if y[j] != 0:
            y[j] = (1 + 0.05) * y[j]
        else:
            y[j] = 0.00025
        v[:, j + 1] = y
        fv[j + 1] = fun(y)
    idx = np.argsort(fv, kind='stable')
    fv = fv[idx]
    v = v[:, idx]
    itercount = 1
    func_evals = n + 1
    while func_evals < maxfunevals and itercount < maxiter:
        if (np.max(np.abs(fv[0] - fv[1:])) <= max(tolfun, 10 * np.spacing(abs(fv[0]))) and
                np.max(np.abs(v[:, 1:] - v[:, [0]])) <= max(tolx, 10 * np.spacing(abs(np.max(v[:, 0]))))):
            break
        xbar = v[:, :n].sum(axis=1) / n
        xr = (1 + rho) * xbar - rho * v[:, -1]
        fxr = fun(xr)
        func_evals += 1
        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[:, -1]
            fxe = fun(xe)
            func_evals += 1
            if fxe < fxr:
                v[:, -1] = xe
                fv[-1] = fxe
            else:
                v[:, -1] = xr
                fv[-1] = fxr
        else:
            if fxr < fv[n - 1]:
                v[:, -1] = xr
                fv[-1] = fxr
            else:
                shrink = False
                if fxr < fv[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * v[:, -1]
                    fxc = fun(xc)
                    func_evals += 1
                    if fxc <= fxr:
                        v[:, -1] = xc
                        fv[-1] = fxc
                    else:
                        shrink = True
                else:
                    xcc = (1 - psi) * xbar + psi * v[:, -1]
                    fxcc = fun(xcc)
                    func_evals += 1
                    if fxcc < fv[-1]:
                        v[:, -1] = xcc
                        fv[-1] = fxcc
                    else:
                        shrink = True
                if shrink:
                    for j in range(1, n + 1):
                        v[:, j] = v[:, 0] + sigma * (v[:, j] - v[:, 0])
                        fv[j] = fun(v[:, j])
                    func_evals += n
        idx = np.argsort(fv, kind='stable')
        fv = fv[idx]
        v = v[:, idx]
        itercount += 1
    return v[:, 0], fv[0], func_evals


def _bound_class(lb, ub):
    cls = np.zeros(len(lb), dtype=int)
    for i in range(len(lb)):
        k = int(np.isfinite(lb[i])) + 2 * int(np.isfinite(ub[i]))
        if k == 3 and lb[i] == ub[i]:
            k = 4
        cls[i] = k
    return cls


def _xtransform(xu, cls, lb, ub):
    x = np.zeros(len(cls))
    k = 0
    for i, c in enumerate(cls):
        if c == 1:
            x[i] = lb[i] + xu[k] ** 2
            k += 1
        elif c == 2:
            x[i] = ub[i] - xu[k] ** 2
            k += 1
        elif c == 3:
            t = (np.sin(xu[k]) + 1) / 2
            t = t * (ub[i] - lb[i]) + lb[i]
            x[i] = max(lb[i], min(ub[i], t))
            k += 1
        elif c == 4:
            x[i] = lb[i]
        else:
            x[i] = xu[k]
            k += 1
    return x


def fminsearchbnd(fun, x0, lb, ub, **opts):
    """fminsearchbnd: bounds via sin / quadratic transforms."""
    x0 = np.asarray(x0, float).ravel()
    n = x0.size
    lb = np.full(n, -np.inf) if lb is None else np.asarray(lb, float).ravel()
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, float).ravel()
    cls = _bound_class(lb, ub)
    x0u = []
    for i, c in enumerate(cls):
        if c == 1:
            x0u.append(0.0 if x0[i] <= lb[i] else np.sqrt(x0[i] - lb[i]))
        elif c == 2:
            x0u.append(0.0 if x0[i] >= ub[i] else np.sqrt(ub[i] - x0[i]))
        elif c == 3:
            if x0[i] <= lb[i]:
                x0u.append(-np.pi / 2)
            elif x0[i] >= ub[i]:
                x0u.append(np.pi / 2)
            else:
                t = 2 * (x0[i] - lb[i]) / (ub[i] - lb[i]) - 1
                x0u.append(2 * np.pi + np.arcsin(max(-1, min(1, t))))
        elif c == 0:
            x0u.append(x0[i])
    x0u = np.array(x0u)
    if x0u.size == 0:
        x = _xtransform(x0u, cls, lb, ub)
        return x, fun(x), 0

    def intrafun(xu):
        return fun(_xtransform(xu, cls, lb, ub))

    xu, fval, nfev = fminsearch(intrafun, x0u, **opts)
    return _xtransform(xu, cls, lb, ub), fval, nfev
