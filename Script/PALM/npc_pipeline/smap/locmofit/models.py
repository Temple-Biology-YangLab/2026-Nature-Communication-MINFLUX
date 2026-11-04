# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""LocMoFit geometric models."""
import numpy as np


def matlab_linspace(a, b, n):
    """MATLAB linspace: n is floored."""
    n = int(np.floor(n))
    if n < 1:
        return np.array([])
    if n == 1:
        return np.array([b], dtype=float)
    return np.linspace(a, b, n)


class GeometricModel:
    name = []
    fix = []
    value = []
    lb = []
    ub = []
    min = []
    max = []
    model_type = 'discretized'
    dimension = 2
    max_locs_prec_factor = None

    def par_table(self):
        return [dict(name=n, fix=bool(f), value=float(v), lb=float(l), ub=float(u), min=float(mi),
                     max=float(ma))
                for n, f, v, l, u, mi, ma in zip(self.name, self.fix, self.value, self.lb, self.ub,
                                                 self.min, self.max)]

    def reference(self, par, dx, locs_prec_factor):
        raise NotImplementedError


class Ring2D(GeometricModel):
    """LocMoFit/models/ring2D.m: points evenly spaced on a circle."""
    name = ['radius']
    fix = [True]
    value = [53.7]
    lb = [0.0]
    ub = [0.0]
    min = [0.0]
    max = [100.0]

    def reference(self, par, dx, locs_prec_factor):
        minD = locs_prec_factor * dx
        circumference = 2 * np.pi * par['radius']
        corner_num = circumference / minD
        pos = matlab_linspace(0, 2 * np.pi, corner_num + 1)[:-1]
        x = par['radius'] * np.cos(pos)
        y = par['radius'] * np.sin(pos)
        return {'x': x, 'y': y, 'n': np.ones(len(x))}


class Arc2D(GeometricModel):
    """LocMoFit/models/arc2D.m: arc of a circle, parameters radius and
    opening angle theta (deg); centred on its centroid.
    """
    name = ['radius', 'theta']
    fix = [False, False]
    value = [10.0, 60.0]
    lb = [-np.inf, -np.inf]
    ub = [np.inf, np.inf]
    min = [5.0, 5.0]
    max = [30.0, 360.0]
    max_locs_prec_factor = 5

    def reference(self, par, dx, locs_prec_factor):
        r = par['radius']
        theta = par['theta']
        lpf = min(locs_prec_factor, 5)
        arclen = 2 * np.pi * r * theta / 360
        nsample = max(int(np.floor(arclen / (dx * lpf) + 0.5)), 1)
        half = np.deg2rad(theta) / 2
        centroidx = r * np.sin(half) / half
        ang = matlab_linspace(-half, half, nsample)
        x = r * np.cos(ang) - centroidx
        y = r * np.sin(ang)
        return {'x': x, 'y': y, 'n': np.ones(len(x))}


class Arc2DArcLen(GeometricModel):
    """LocMoFit/models/arc2D_arcLen.m: arc parametrised by arc length and theta."""
    name = ['arcLength', 'theta']
    fix = [False, False]
    value = [10.0, 30.0]
    lb = [-np.inf, -np.inf]
    ub = [np.inf, np.inf]
    min = [5.0, 5.0]
    max = [30.0, 180.0]
    max_locs_prec_factor = 5

    def reference(self, par, dx, locs_prec_factor):
        arclen = par['arcLength']
        twice_theta = np.deg2rad(2 * par['theta'])
        r = arclen / twice_theta
        lpf = min(locs_prec_factor, 5)
        nsample = max(int(np.floor(arclen / (dx * lpf) + 0.5)), 1)
        half = twice_theta / 2
        centroidx = r * np.sin(half) / half
        ang = matlab_linspace(-half, half, nsample)
        x = r * np.cos(ang) - centroidx
        y = r * np.sin(ang)
        a = np.pi / 2
        xo = np.cos(a) * x + np.sin(a) * y
        yo = np.cos(a) * y - np.sin(a) * x
        return {'x': xo, 'y': yo, 'n': np.ones(len(x))}


class Ellipse2D(GeometricModel):
    """LocMoFit/models/ellipse2D.m: discretized ellipse with semi-axes a, b."""
    name = ['a', 'b']
    fix = [True, True]
    value = [40.0, 50.0]
    lb = [-np.inf, -np.inf]
    ub = [np.inf, np.inf]
    min = [0.0, 0.0]
    max = [100.0, 100.0]

    def reference(self, par, dx, locs_prec_factor):
        a, b = par['a'], par['b']
        minD = locs_prec_factor * dx
        h = ((a - b) / (a + b)) ** 2
        circ = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        t = matlab_linspace(0, 2 * np.pi, circ / minD + 1)[:-1]
        x = a * np.cos(t)
        y = b * np.sin(t)
        return {'x': x, 'y': y, 'n': np.ones(len(x))}


MODELS = {'ring2D': Ring2D, 'arc2D': Arc2D, 'arc2D_arcLen': Arc2DArcLen, 'ellipse2D': Ellipse2D}
