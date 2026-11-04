# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""SMAP ROI manager: makeCellGrid, cell rendering, segmentNPC, NPCsegmentCleanup."""
from dataclasses import dataclass, field
import numpy as np
from numba import njit
from scipy.optimize import least_squares

from .matlab_compat import fspecial_gaussian, fspecial_disk, filter2_same
from .render import render, colorfield_range


@dataclass
class SESettings:
    site_pixelsize: float = 3.0
    site_fov: float = 400.0
    site_roi: float = 220.0
    cell_pixelsize: float = 15.0
    cell_fov: float = 5000.0


@dataclass
class Cell:
    ID: int
    pos: np.ndarray
    filenumber: int = 1
    image: dict = None


@dataclass
class Site:
    ID: int
    pos: np.ndarray
    cell: int = 0
    filenumber: int = 1
    annotation: dict = field(default_factory=lambda: {'use': True})
    evaluation: dict = field(default_factory=dict)


def make_cell_grid(loc, se: SESettings, filenumbers=(1,)):
    """makeCellGrid: tile the field of view with cells of size cell_fov."""
    xm, xx = np.min(loc['xnm']), np.max(loc['xnm'])
    ym, yx = np.min(loc['ynm']), np.max(loc['ynm'])
    posx = np.arange(xm - se.site_fov, xx + se.site_fov + 1e-9, se.cell_fov)
    posy = np.arange(ym - se.site_fov, yx + se.site_fov + 1e-9, se.cell_fov)
    cells = []
    ID = 1
    for fnum in filenumbers:
        for x in posx:
            for y in posy:
                cells.append(Cell(ID, np.array([x, y]) + se.cell_fov / 2, fnum))
                ID += 1
    return cells


def render_cells(cells, locdata, layers, se: SESettings, render_kw=None):
    """Render the 'raw' image of every cell for every layer (SiteExplorer.plotobject).
    """
    render_kw = dict(render_kw or {})
    half = se.cell_fov / 2
    kws = []
    for layer in layers:
        kw = dict(render_kw)
        if kw.get('colorfield') and kw.get('crange') is None:
            kw['crange'] = colorfield_range(locdata.get(layer)[kw['colorfield']])
        kws.append(kw)
    for cell in cells:
        imgs = []
        for layer, kw in zip(layers, kws):
            locs = locdata.get(layer)
            if 'filenumber' in locs:
                fsel = locs['filenumber'] == cell.filenumber
                locs = {k: v[fsel] for k, v in locs.items()}
            sel = (np.abs(locs['xnm'] - cell.pos[0]) < half) & (np.abs(locs['ynm'] - cell.pos[1]) < half)
            locs = {k: v[sel] for k, v in locs.items()}
            im = render(locs, [cell.pos[0] - half, cell.pos[0] + half],
                        [cell.pos[1] - half, cell.pos[1] + half], se.cell_pixelsize, **kw)
            imgs.append(im)
        cell.image = {'layers': imgs, 'rangex': imgs[0]['rangex'], 'rangey': imgs[0]['rangey'],
                      'pixrec': se.cell_pixelsize}
    return cells


@njit(cache=True)
def nms2d_block(img, n, maxmax):
    """NMS2DBlockC2.c (block non-maximum suppression). img[row, col]."""
    sx = img.shape[0]  # rows (fast index in MATLAB)
    sy = img.shape[1]  # cols
    out = np.zeros((maxmax, 3))
    maxind = 0
    i1 = n
    while i1 < sy - n - n:
        j1 = n
        while j1 < sx - n - n:
            mi = i1
            mj = j1
            for i2 in range(i1, i1 + n + 1):
                for j2 in range(j1, j1 + n + 1):
                    if img[j2, i2] > img[mj, mi]:
                        mi = i2
                        mj = j2
            stop = False
            for i2 in range(mi - n, mi + n + 1):
                for j2 in range(mj - n, mj + n + 1):
                    if (i2 < i1 or i2 > i1 + n) or (j2 < j1 or j2 > j1 + n):
                        if img[j2, i2] > img[mj, mi]:
                            stop = True
                            break
                if stop:
                    break
            if (not stop) and maxind < maxmax:
                out[maxind, 0] = mj + 1
                out[maxind, 1] = mi + 1
                out[maxind, 2] = img[mj, mi]
                maxind += 1
            j1 += n + 1
        i1 += n + 1
    return out[:maxind]


def segment_npc(cells, cutoff=0.06, diameterNPC=100.0, rim=20.0, sigmaNMS=3, return_images=False):
    """segmentNPC: find ring-shaped structures of a given diameter in the cell images.
    """
    sigmaNMS = int(np.floor((sigmaNMS - 1) / 2 + 0.5)) * 2 + 1
    pixrec = cells[0].image['pixrec']
    rRingO = (diameterNPC / 2 + rim) / pixrec
    n = np.arange(-diameterNPC, diameterNPC + 1e-9, pixrec)
    X, Y = np.meshgrid(n, n)
    h = np.exp(-np.abs(X ** 2 + Y ** 2 - (diameterNPC / 2) ** 2) / 4 / rim ** 2)
    h = h / h.sum()
    g1 = fspecial_gaussian(21, max(1, 0.7 * rRingO))
    g2 = fspecial_gaussian(51, max(2, 2 * rRingO))
    sites = []
    ID = 1
    images = []
    for cell in cells:
        srim = 0
        for im in cell.image['layers']:
            srim = srim + im['image']
        srim = np.asarray(srim, dtype=np.float64)
        srimbw = srim.sum(axis=2) if srim.ndim == 3 else srim
        imfD = filter2_same(h, np.sqrt(srimbw))
        srimf = filter2_same(g1, imfD) - filter2_same(g2, imfD)
        s = srimf.shape
        maxmax = int(np.floor(s[0] * s[1] / (2 * sigmaNMS) ** 2) * 4)
        mo = nms2d_block(srimf.astype(np.float32).astype(np.float64), sigmaNMS, maxmax)
        pc = cell.image['pixrec']
        ysite = mo[:, 0] * pc + cell.image['rangey'][0]
        xsite = mo[:, 1] * pc + cell.image['rangex'][0]
        good = mo[:, 2] >= cutoff
        for x, y in zip(xsite[good], ysite[good]):
            sites.append(Site(ID, np.array([x, y, 0.0]), cell.ID, cell.filenumber))
            ID += 1
        if return_images:
            images.append(dict(cell=cell.ID, filtered=srimf, maxima=mo, good=good))
    if return_images:
        return sites, images
    return sites


_TREES = {}


def get_site_locs(locdata, site, layer, radius, fields=None, pos=None):
    """SEEvaluationProcessor.getLocs with 'size' = radius (circular ROI)."""
    from scipy.spatial import cKDTree
    locs = locdata.get(layer)
    pos = site.pos if pos is None else pos
    key = (id(locdata), id(locs['xnm']))
    tree = _TREES.get(key)
    if tree is None or tree[1] is not locs['xnm']:
        _TREES.clear()
        tree = (cKDTree(np.c_[locs['xnm'], locs['ynm']].astype(np.float64)), locs['xnm'])
        _TREES[key] = tree
    cand = np.array(sorted(tree[0].query_ball_point([pos[0], pos[1]], radius * (1 + 1e-9) + 1e-6)), dtype=int)
    if cand.size:
        xr = locs['xnm'][cand] - pos[0]
        yr = locs['ynm'][cand] - pos[1]
        ok = xr ** 2 + yr ** 2 <= radius ** 2
        if 'filenumber' in locs:
            ok &= locs['filenumber'][cand] == site.filenumber
        cand = cand[ok]
        xr, yr = xr[ok], yr[ok]
    else:
        xr = yr = np.zeros(0)
    keys = fields if fields is not None else list(locs.keys())
    out = {k: locs[k][cand] for k in keys if k in locs}
    out['xnm'] = locs['xnm'][cand]
    out['ynm'] = locs['ynm'][cand]
    out['xnmrot'] = xr
    out['ynmrot'] = yr
    return out


def fitposring(x, y, R=None):
    """fitposring.m via implicitfit.m: ring fit minimising sum(|r - R|)
    (lsqnonlin on err/sqrt(|err|)). Returns x0, y0, R0, resnorm.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def callfit(par):
        if R is None:
            err = np.sqrt((x - par[0]) ** 2 + (y - par[1]) ** 2) - par[2]
        else:
            err = np.sqrt((x - par[0]) ** 2 + (y - par[1]) ** 2) - R
        with np.errstate(invalid='ignore', divide='ignore'):
            out = err / np.sqrt(np.abs(err))
        out[~np.isfinite(out)] = 0
        return out

    if R is None:
        start = [x.mean(), y.mean(), np.sqrt(np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2)]
    else:
        start = [x.mean(), y.mean()]
    r = least_squares(callfit, start, method='trf', xtol=1e-6, ftol=1e-6, max_nfev=100 * len(start))
    resnorm = np.sqrt(np.sum(r.fun ** 2))
    if R is None:
        return r.x[0], r.x[1], r.x[2], resnorm
    return r.x[0], r.x[1], R, resnorm


@dataclass
class CleanupParameters:
    """NPCsegmentCleanup GUI parameters (defaults)."""
    R: float = 55.0
    dR: float = 15.0
    center: bool = True
    inside: float = 0.3
    outside: float = 0.75
    radiusrange: tuple = (45.0, 65.0)
    minsize: float = 25.0
    minlocs: int = 10
    maxPSF: float = 150.0


def npc_segment_cleanup(locdata, sites, layer, se: SESettings, p: CleanupParameters = None):
    """NPCsegmentCleanup for every site (in place). Sets site.annotation['use'],
    stores the evaluation in site.evaluation['NPCsegmentCleanup'] and, with
    p.center, moves site.pos to the fitted ring centre.
    """
    p = p or CleanupParameters()
    R, dR = p.R, p.dR
    rad = se.site_roi / 2
    for site in sites:
        locs = get_site_locs(locdata, site, layer, rad, ['xnm', 'ynm', 'PSFxnm'])
        out = {}
        if len(locs['xnm']) == 0:
            site.annotation['use'] = False
            site.evaluation['NPCsegmentCleanup'] = dict(R0=0, inside=1, outside=1, inring=0, sizeav=0, locs=0)
            continue
        if p.center:
            locsc = get_site_locs(locdata, site, layer, rad * 1.2, ['xnm', 'ynm'])
            x0c, y0c, _, _ = fitposring(locsc['xnm'], locsc['ynm'], R)
            site.pos = np.array([x0c, y0c, site.pos[2] if len(site.pos) > 2 else 0.0])
        x0, y0, R0, _ = fitposring(locs['xnm'], locs['ynm'], R)
        xm = locs['xnm'] - x0
        ym = locs['ynm'] - y0
        _, _, R0r, _ = fitposring(xm, ym)
        r2 = xm ** 2 + ym ** 2
        goodradius = p.radiusrange[0] < R0r < p.radiusrange[1]
        inside = int(np.sum(r2 < (R - dR) ** 2))
        inring = int(np.sum((r2 > (R - dR) ** 2) & (r2 < (R + dR) ** 2)))
        outside = int(np.sum(r2 > (R + dR) ** 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            goodin = (inside / inring) < p.inside if inring else (inside == 0 and False)
            goodout = (outside / inring) < p.outside if inring else False
        if len(xm) > 1:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([xm, ym])))
            sm = np.sqrt(np.maximum(ev, 0).astype(complex)).real
            smav = float(np.sqrt(np.prod(sm)))
        else:
            smav = 0.0
        goodsize = smav > p.minsize
        goodlocs = len(xm) > p.minlocs
        if 'PSFxnm' in locs and len(locs['PSFxnm']):
            mpsf = float(np.mean(locs['PSFxnm']))
            goodpsf = mpsf <= p.maxPSF
        else:
            mpsf, goodpsf = 0.0, True
        use = bool(goodradius and goodout and goodin and goodsize and goodlocs and goodpsf)
        site.annotation['list4'] = int(use) + 1
        site.annotation['use'] = use
        site.evaluation['NPCsegmentCleanup'] = dict(
            R0=R0r, inside=inside, outside=outside, inring=inring, sizeav=smav, locs=len(xm),
            meanPSF=mpsf, goodradius=goodradius, goodin=goodin, goodout=goodout,
            goodsize=goodsize, goodlocs=goodlocs, goodpsf=goodpsf)
    return sites
