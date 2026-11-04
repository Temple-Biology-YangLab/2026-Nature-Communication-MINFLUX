# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""LocMoFit: maximum-likelihood model fitting of localization data (2D, discretized models).
"""
import copy
import numpy as np

from .optim import fminsearchbnd
from .models import GeometricModel


def _default_lpars(roi_size):
    r = roi_size / 2
    return [
        dict(name='x', fix=False, lb=-r, ub=r, min=-r, max=r, value=0.0),
        dict(name='y', fix=False, lb=-r, ub=r, min=-r, max=r, value=0.0),
        dict(name='zrot', fix=False, lb=-np.inf, ub=np.inf, min=-np.inf, max=np.inf, value=0.0),
        dict(name='variation', fix=True, lb=0.0, ub=10.0, min=0.0, max=20.0, value=0.0),
        dict(name='xscale', fix=True, lb=1.0, ub=1.0, min=1.0, max=1.0, value=1.0),
        dict(name='yscale', fix=True, lb=1.0, ub=1.0, min=1.0, max=1.0, value=1.0),
        dict(name='weight', fix=False, lb=1e-5, ub=1.0, min=1e-5, max=1.0, value=1e-5),
    ]


def _default_offset():
    return dict(name='weight', fix=False, lb=0.001, ub=0.999, min=0.0, max=1.0, value=0.0)


class ModelEntry:
    def __init__(self, geo: GeometricModel, layer=1, sigma_factor=(1.0, 0.0), fix_sigma=False,
                 sigma=15.0, weight=1.0):
        self.geo = geo
        self.layer = layer
        sf = np.atleast_1d(np.asarray(sigma_factor, float))
        self.sigma_factor = np.array([sf[0], sf[1] if sf.size > 1 else 0.0])
        self.fix_sigma = fix_sigma
        self.sigma = sigma
        self.weight = weight  # exponent 'model.weight' (SMLMModel.weight)


class LocMoFit:
    def __init__(self, roi_size=220.0, refpoint_spacing=0.75, min_sigma='median',
                 layer_norm=True, solver_options=None):
        self.roi_size = float(roi_size)
        self.refpoint_spacing = refpoint_spacing
        self.min_sigma = min_sigma
        self.layer_norm = layer_norm
        self.solver_options = dict(solver_options or {})
        self.models = []
        self.pars = []
        self.fit_info = None

    def add_model(self, geo, layer=1, **kw):
        """Add a geometric model (e.g. Ring2D()) fitted to `layer`."""
        mid = len(self.models) + 1
        self.models.append(ModelEntry(geo, layer, **kw))
        rows = [dict(model=mid, type='lPar', **p) for p in _default_lpars(self.roi_size)]
        rows += [dict(model=mid, type='mPar', **p) for p in geo.par_table()]
        offsets = [r for r in self.pars if r['type'] == 'offset']
        others = [r for r in self.pars if r['type'] != 'offset']
        if not any(r['model'] == 90 + layer for r in offsets):
            offsets.append(dict(model=90 + layer, type='offset', **_default_offset()))
        self.pars = others + rows + offsets
        return mid

    @property
    def model_layers(self):
        return sorted(set(m.layer for m in self.models))

    def _find(self, pid):
        parts = pid.split('.')
        mid = int(parts[0].lstrip('m'))
        name = parts[-1]
        typ = parts[1] if len(parts) == 3 else None
        for r in self.pars:
            if r['model'] == mid and r['name'] == name and (typ is None or r['type'] == typ):
                return r
        raise KeyError(pid)

    def set_par(self, pid, **kw):
        """Set fields (value, fix, lb, ub, min, max) of parameter 'm1.mPar.radius' etc.
        """
        r = self._find(pid)
        for k, v in kw.items():
            if k not in ('value', 'fix', 'lb', 'ub', 'min', 'max'):
                raise KeyError(k)
            r[k] = bool(v) if k == 'fix' else float(v)

    def get_par(self, pid):
        return self._find(pid)['value']

    def par_table(self):
        """Parameter table as a list of dicts (like the LocMoFit GUI table)."""
        return copy.deepcopy(self.pars)

    def _structs(self, fitpars):
        """inputPar2struct + exportOffset: fit vector -> per-model dicts."""
        vals = np.array([r['value'] for r in self.pars], float)
        free = np.array([not r['fix'] for r in self.pars])
        vals[free] = fitpars
        lpars, mpars, offs = {}, {}, {}
        for r, v in zip(self.pars, vals):
            d = {'lPar': lpars, 'mPar': mpars, 'offset': offs}[r['type']]
            d.setdefault(r['model'], {})[r['name']] = v
        return lpars, mpars, offs

    def _sigma(self, m: ModelEntry, locprec):
        if m.fix_sigma:
            return np.array([1.0, 0.0]), np.full_like(locprec, m.sigma)
        s = locprec.copy()
        if m.geo.model_type != 'discrete' and self.min_sigma == 'median':
            med = np.median(locprec)
            s[s < med] = med
        return m.sigma_factor.copy(), s

    def model_points(self, k, mpars, lpars, locs_prec_factor):
        """Reference points of model k (1-based) after its lPar transforms."""
        m = self.models[k - 1]
        ref = m.geo.reference(mpars[k], self.refpoint_spacing, locs_prec_factor)
        x = np.asarray(ref['x'], float)
        y = np.asarray(ref['y'], float)
        n = np.asarray(ref['n'], float)
        keep = n != 0
        x, y, n = x[keep], y[keep], n[keep]
        for s in sorted(set([1, k])):
            zrot = np.deg2rad(lpars[s]['zrot'])
            xo = np.cos(zrot) * x + np.sin(zrot) * y
            yo = np.cos(zrot) * y - np.sin(zrot) * x
            x = xo + lpars[s]['x']
            y = yo + lpars[s]['y']
        return x, y, n

    def intensity(self, fitpars, locs):
        """intensityCal: likelihood of every localization (2D)."""
        lpars, mpars, offs = self._structs(fitpars)
        in_layers = np.isin(locs['layer'], self.model_layers)
        lx = np.asarray(locs['xnm'], float)[in_layers]
        ly = np.asarray(locs['ynm'], float)[in_layers]
        llayer = np.asarray(locs['layer'])[in_layers]
        lprec = np.asarray(locs['locprecnm'], float)[in_layers]
        total = np.zeros(len(lx))
        one = []
        for k, m in enumerate(self.models, start=1):
            lprec = np.sqrt(lprec ** 2 + lpars[k]['variation'] ** 2)
            sf, sset = self._sigma(m, lprec)
            lpf = float(np.min(sset)) if sset.size else m.sigma
            x, y, n = self.model_points(k, mpars, lpars, lpf)
            sig = (sset + sf[1]) * sf[0]
            if x.size == 0 or sig.size == 0:
                one.append(np.zeros(len(lx)))
                continue
            d2 = (lx[:, None] - x[None, :]) ** 2 + (ly[:, None] - y[None, :]) ** 2
            svol = n[None, :] * (2 * np.pi ** -1 / (sig[:, None] * sig[:, None]))
            val = np.sum(svol * np.exp(-d2 / (2 * sig[:, None] ** 2)), axis=1) / n.sum()
            one.append(val)
        for layer in self.model_layers:
            ks = [k for k, m in enumerate(self.models, start=1) if m.layer == layer]
            allw = sum(lpars[k]['weight'] for k in ks)
            for k in ks:
                wk = lpars[k]['weight'] / allw
                sel = llayer == layer
                v = one[k - 1].copy()
                v[sel] = (v[sel] * wk) ** self.models[k - 1].weight
                one[k - 1] = v
        for v in one:
            total = total + v
        offset_unit = 1 / (np.pi * (self.roi_size / 2) ** 2)
        nl = len(self.model_layers)
        for ch_i, layer in enumerate(self.model_layers, start=1):
            w = offs[90 + layer]['weight']
            sel = llayer == layer
            total[sel] = ((1 - w) * total[sel] + w * offset_unit) * (1.0 / nl)
        return total, llayer

    def loglikelihood(self, fitpars, locs, comp=None):
        total, llayer = self.intensity(fitpars, locs)
        if comp is None:
            comp = np.ones(len(total))
        used = total != 0  # intensityCal drops localizations with zero intensity
        return float(np.sum(comp[used] * np.log(total[used])))

    def fit(self, locs):
        """Fit the models to one site."""
        locs = dict(locs)
        if 'layer' not in locs:
            locs['layer'] = np.ones(len(locs['xnm']), int)
        layers = np.asarray(locs['layer'])
        maxl = int(layers.max()) if layers.size else 1
        nper = np.array([np.sum(layers == l) for l in range(1, maxl + 1)], float)
        used = [l for l in self.model_layers if l <= maxl]
        comp = np.ones(len(layers))
        if self.layer_norm and used:
            nmax = max(nper[l - 1] for l in used)
            for l in used:
                comp[layers == l] = nmax / nper[l - 1]
        comp = comp[np.isin(layers, self.model_layers)]
        free = np.array([not r['fix'] for r in self.pars])
        init = np.array([r['value'] for r in self.pars])[free]
        lb = np.array([r['lb'] for r in self.pars])[free] + init
        ub = np.array([r['ub'] for r in self.pars])[free] + init
        mn = np.array([r['min'] for r in self.pars])[free]
        mx = np.array([r['max'] for r in self.pars])[free]
        lb = np.where(lb < mn, mn, lb)
        ub = np.where(ub > mx, mx, ub)

        def objfun(p):
            return -self.loglikelihood(p, locs, comp)

        opts = {}
        so = self.solver_options
        if 'TolX' in so:
            opts['tolx'] = so['TolX']
        if 'TolFun' in so:
            opts['tolfun'] = so['TolFun']
        if 'MaxFunEvals' in so:
            opts['maxfunevals'] = so['MaxFunEvals']
        if 'MaxIter' in so:
            opts['maxiter'] = so['MaxIter']
        best, mll, nfev = fminsearchbnd(objfun, init, lb, ub, **opts)
        for r, v in zip([r for r in self.pars if not r['fix']], best):
            r['value'] = float(v)
        nfit = int(np.sum(np.isin(layers, self.model_layers)))
        k = int(free.sum())
        info = {'LLfit': -mll / nfit if nfit else np.nan, 'numOfLocsPerLayer': nper,
                'AIC': 2 * k + 2 * mll, 'nfev': nfev}
        info['AICc'] = info['AIC'] + 2 * (k ** 2 + k) / (nfit - k - 1) if nfit - k - 1 != 0 else np.nan
        info['normAICc'] = info['AICc'] / nfit if nfit else np.nan
        bgd = []
        for l in self.model_layers:
            w = self._find('m%d.offset.weight' % (90 + l))['value']
            nl = nper[l - 1] if l - 1 < len(nper) else 0
            bgd.append(w * nl / (np.pi * (0.5 * self.roi_size / 1000) ** 2))
        info['BGDensity'] = np.array(bgd)
        self.fit_info = info
        out = {'m%d.%s.%s' % (r['model'], r['type'], r['name']): r['value'] for r in self.pars}
        out.update(info)
        return out

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, d, roi_size):
        """Build a fitter from a dict {'models': [{'class': 'ring2D', 'layer': 1}, ...],
        'pars': [{'id': 'm1.mPar.radius', 'fix': False, 'value': 53.7, 'lb': -20, ...}, ...]}
        (the format of LOCMOFIT_MODEL in 01_extract.py). Values beyond +-1e299 mean +-inf.
        """
        from .models import MODELS
        f = cls(roi_size=roi_size)
        for m in d['models']:
            f.add_model(MODELS[m['class']](), layer=int(m.get('layer', 1)))
        for p in d.get('pars', []):
            kw = {}
            for k in ('value', 'fix', 'lb', 'ub', 'min', 'max'):
                if k in p and p[k] is not None:
                    v = p[k]
                    if k == 'fix':
                        v = bool(v)
                    else:
                        v = float(v)
                        v = np.inf if v >= 1e299 else (-np.inf if v <= -1e299 else v)
                    kw[k] = v
            f.set_par(p['id'], **kw)
        return f

