# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of npc_pipeline.smap against SMAP (MATLAB): stage-isolated comparison of grouping,
cell rendering, segmentNPC, NPCsegmentCleanup and LocMoFit, plus an end-to-end check of
npc_pipeline.smap_sites on the same file.
"""
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from npc_pipeline.smap.grouping import LocData, Layer, group_localizations
from npc_pipeline.smap.roimanager import (Cell, Site, SESettings, CleanupParameters, segment_npc,
                                npc_segment_cleanup, get_site_locs)
from npc_pipeline.smap.locmofit import LocMoFit

BIG = 1e300


def write_config(cfg, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    cfg = dict(cfg)
    cfg['output_dir'] = run_dir
    path = os.path.join(run_dir, 'config.json')
    with open(path, 'w') as fh:
        json.dump(cfg, fh, indent=1)
    return path


def py_layer(cfg):
    f = {k: (v[0] if v[0] > -BIG / 10 else -np.inf, v[1] if v[1] < BIG / 10 else np.inf)
         for k, v in cfg['layer1']['filters'].items()}
    return Layer(f, grouped=bool(cfg['layer1'].get('grouped', True)))


def py_se(cfg):
    s = cfg['se']
    return SESettings(site_pixelsize=s['site_pixelsize'], site_fov=s['site_fov'],
                      site_roi=s['site_roi'], cell_pixelsize=s['cell_pixelsize'],
                      cell_fov=s['cell_fov'])


def py_cleanup(cfg):
    c = cfg['cleanup']
    return CleanupParameters(R=c['R'], dR=c['dR'], center=bool(c['center']), inside=c['inside'],
                             outside=c['outside'], radiusrange=tuple(c['radiusrange']),
                             minsize=c['minsize'], minlocs=c['minlocs'], maxPSF=c['maxPSF'])


def py_render_kw(cfg):
    r = cfg.get('render', {'colormode': 'normal'})
    if r['colormode'] == 'normal':
        return {}
    from npc_pipeline.smap.render import LUTS
    if r['colormode'] != 'field' or r.get('lut', 'jet') not in LUTS:
        raise ValueError('render: only colormode normal, or field with LUT %s, are ported' % ' / '.join(LUTS))
    return {'colorfield': r['renderfield'], 'lut': r.get('lut', 'jet')}


def py_locmofit(lc, roi):
    return LocMoFit.from_dict(lc, roi)


def matlab_status(run_dir):
    p = os.path.join(run_dir, 'matlab_reference', 'status.json')
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


def lm(path):
    return loadmat(path, squeeze_me=True, simplify_cells=True)


def as_list(x):
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    return list(x)


def locdict(s):
    """MATLAB loc struct -> dict of 1D float arrays."""
    return {k: np.atleast_1d(np.asarray(v)).ravel() for k, v in s.items()
            if np.asarray(v).dtype.kind in 'fiub'}


def has_error(x):
    e = x.get('error', '')
    return isinstance(e, str) and len(e) > 0


def ref_path(run_dir, name):
    return os.path.join(run_dir, 'matlab_reference', name)


def absdiff_stats(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0)
    return dict(n=int(d.size), median=float(np.median(d)), p99=float(np.percentile(d, 99)),
                p999=float(np.percentile(d, 99.9)), max=float(d.max()))


def _close(a, b, rtol=1e-4, atol=1e-2):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (np.abs(a - b) <= atol + rtol * np.abs(b))))


def stage_S2(cfg, run_dir, out):
    r = lm(ref_path(run_dir, 'S2_loaded.mat'))
    loc = locdict(r['loc'])
    gref = locdict(r['grouploc'])
    sel = locdict(r['sel'])
    inp = {k: v for k, v in loc.items() if k not in ('groupindex', 'numberInGroup')}
    g, gi, nig = group_localizations(inp, cfg['grouping']['dx'], cfg['grouping']['dt'])
    m = dict(n_locs=len(loc['xnm']), n_groups_smap=len(gref['xnm']), n_groups_python=len(g['xnm']))
    fields, ok = {}, True
    if 'groupindex' in loc:
        sg = np.asarray(loc['groupindex'], float)
        pairs = np.unique(np.c_[gi, sg], axis=0)
        same_partition = len(pairs) == len(np.unique(gi)) == len(np.unique(sg))
        m['same_partition'] = bool(same_partition)
        ok &= same_partition
        if same_partition:
            py2sm = dict(zip(pairs[:, 0], pairs[:, 1]))
            srow = {v: i for i, v in enumerate(np.asarray(gref['groupindex'], float))}
            rows = np.array([srow.get(py2sm[v], -1) for v in g['groupindex']])
            if np.all(rows >= 0):
                for k in ['xnm', 'ynm', 'frame', 'phot', 'bg', 'locprecnm', 'PSFxnm', 'LLrel', 'numberInGroup']:
                    if k in g and k in gref:
                        fields[k] = absdiff_stats(g[k], gref[k][rows])
                        fields[k]['pass'] = _close(g[k], gref[k][rows])
                        ok &= fields[k]['pass']
            else:
                ok = False
                m['error'] = 'group indices of SMAP grouploc do not match loc.groupindex'
    else:
        m['note'] = 'SMAP loc has no groupindex; partition not compared'
    m['fields'] = fields
    lay = py_layer(cfg)
    msk = lay.mask(gref)
    pysel = {k: v[msk] for k, v in gref.items()}
    m['layer1_n_smap'] = len(sel['xnm'])
    m['layer1_n_python'] = int(msk.sum())
    a = set(zip(np.round(sel['frame']).astype(int), np.round(sel['xnm'], 2), np.round(sel['ynm'], 2)))
    b = set(zip(np.round(pysel['frame']).astype(int), np.round(pysel['xnm'], 2), np.round(pysel['ynm'], 2)))
    m['layer1_identical'] = bool(a == b)
    m['layer1_only_smap'] = len(a - b)
    m['layer1_only_python'] = len(b - a)
    ft = r.get('filtertable', {})
    if isinstance(ft, dict):
        m['smap_active_filters_layer1'] = {k: np.atleast_1d(v).tolist() for k, v in ft.items()}
    lo = np.atleast_1d(matlab_status(run_dir).get('layers_on', [1])).tolist()
    m['smap_layers_on'] = lo
    if lo != [1]:
        m['warning'] = 'SMAP had render layers %s on; the port (and this comparison) uses layer 1 only' % lo
        ok = False
    m['pass'] = bool(ok and m['n_groups_smap'] == m['n_groups_python'] and m['layer1_identical'])
    return m


def stage_S4(cfg, run_dir, out):
    """Python renderer (render_cells, production code) on SMAP's loaded
    localizations, layer 1, compared with SMAP's cell images.
    """
    from npc_pipeline.smap.roimanager import render_cells
    ld = _locdata_ref(cfg, run_dir)
    cells_ref = as_list(lm(ref_path(run_dir, 'S4_cells.mat'))['cells'])
    se = py_se(cfg)
    cells = [Cell(int(c['ID']), np.atleast_1d(c['pos']).astype(float)[:2], int(c['filenumber']))
             for c in cells_ref]
    kw = py_render_kw(cfg)
    m = {}
    rp = ref_path(run_dir, 'S4_render_parameters.json')
    if os.path.exists(rp):
        with open(rp) as fh:
            m['smap_render_parameters'] = json.load(fh)
        srp = m['smap_render_parameters']
        used = {'colormode': srp.get('render_colormode'), 'field': srp.get('renderfield'),
                'lut': srp.get('lut'), 'renderer': srp.get('rendermode', 'Gauss'),
                'intensity': srp.get('intensitycoding', 'normal')}
        want = {'colormode': cfg['render']['colormode'], 'field': cfg['render'].get('renderfield'),
                'lut': cfg['render'].get('lut', 'jet'), 'renderer': 'Gauss', 'intensity': 'normal'}
        if want['colormode'] == 'normal':
            used.pop('field'); used.pop('lut'); want.pop('field'); want.pop('lut')
        m['smap_render_settings_used'] = used
        m['smap_render_settings_as_configured'] = used == want
        if used['colormode'] == 'field':
            kw = {'colorfield': used['field'], 'lut': used['lut']}
        else:
            kw = {}
    if kw:
        from npc_pipeline.smap.render import colorfield_range
        py_range = colorfield_range(ld.get(py_layer(cfg))[kw['colorfield']])
        m['colour_range_python'] = [float(v) for v in py_range]
        srp = m.get('smap_render_parameters', {})
        if 'colorfield_min' in srp:
            sm = [float(srp['colorfield_min']), float(srp['colorfield_max'])]
            m['colour_range_smap'] = sm
            m['colour_range_identical'] = bool(np.allclose(py_range, sm, rtol=5e-4, atol=0))
            kw['crange'] = tuple(sm)   # stage isolation: render with SMAP's range
    render_cells(cells, ld, [py_layer(cfg)], se, kw)
    rel, rng, nloc, shape_ok = [], [], [], []
    for c, cr in zip(cells, cells_ref):
        im = c.image['layers'][0]
        ref = np.asarray(cr['image'], float)
        ok_shape = ref.shape == im['image'].shape
        shape_ok.append(ok_shape)
        rng.append(max(np.abs(np.asarray(c.image['rangex']) - np.atleast_1d(cr['rangex'])).max(),
                       np.abs(np.asarray(c.image['rangey']) - np.atleast_1d(cr['rangey'])).max()))
        if not ok_shape:
            rel.append(np.inf)
            continue
        mx = max(ref.max(), 1e-12)
        rel.append(float(np.abs(im['image'] - ref).max() / mx))
        nloc.append((int(im.get('numberOfLocs', -1)), cr.get('numberOfLocs', np.nan)))
    rel = np.array(rel)
    m.update(n_cells=len(cells_ref), shapes_identical=bool(all(shape_ok)),
             max_range_diff_nm=float(max(rng)) if rng else np.nan,
             max_rel_diff=float(rel.max()) if rel.size else np.nan,
             median_rel_diff=float(np.median(rel)) if rel.size else np.nan,
             n_layers_smap=sorted({int(c.get('nlayers', 1)) for c in cells_ref}),
             numberOfLocs_identical=bool(all(a == b for a, b in nloc)))
    m['pass'] = bool(rel.size and np.all(rel < 1e-3) and m['max_range_diff_nm'] < 1e-3 and
                     m.get('colour_range_identical', True))
    m['images_bit_identical'] = int(sum(1 for c, cr in zip(cells, cells_ref)
                                        if np.array_equal(np.asarray(c.image['layers'][0]['image'], np.float32),
                                                          np.asarray(cr['image'], np.float32))))
    if not m.get('smap_render_settings_as_configured', True):
        m['warning'] = ('SMAP rendered with %s, not with the configured settings; the comparison uses '
                        "SMAP's settings, but the SMAP run is not the analysis you configured"
                        % m['smap_render_settings_used'])
    return m


def _match(P, S, tol):
    if len(P) == 0 or len(S) == 0:
        return np.array([]), np.array([], int)
    d, j = cKDTree(S).query(P, distance_upper_bound=tol)
    return d, j


def stage_S5(cfg, run_dir, out):
    cells_ref = as_list(lm(ref_path(run_dir, 'S4_cells.mat'))['cells'])
    sites_ref = as_list(lm(ref_path(run_dir, 'S4_sites.mat'))['sites'])
    cells = []
    for c in cells_ref:
        img = np.asarray(c['image_sum'] if 'image_sum' in c else c['image'], float)
        cells.append(Cell(int(c['ID']), np.atleast_1d(c['pos']).astype(float), int(c['filenumber']),
                          image={'layers': [{'image': img}],
                                 'rangex': np.atleast_1d(c['rangex']).astype(float),
                                 'rangey': np.atleast_1d(c['rangey']).astype(float),
                                 'pixrec': float(c['pixrec'])}))
    s = cfg['segment']
    sites = segment_npc(cells, s['cutoff'], s['diameter'], s['rim'])
    P = np.array([x.pos[:2] for x in sites])
    S = np.array([np.atleast_1d(x['pos'])[:2] for x in sites_ref])
    d, j = _match(P, S, 0.5)
    nm = int(np.isfinite(d).sum()) if len(d) else 0
    m = dict(n_smap=len(sites_ref), n_python=len(sites), n_matched_0p5nm=nm,
             max_pos_diff=float(d[np.isfinite(d)].max()) if nm else np.nan)
    m['pass'] = bool(len(sites) == len(sites_ref) and nm == len(sites_ref))
    return m


def stage_S6(cfg, run_dir, out):
    ld = _locdata_ref(cfg, run_dir)
    clean_ref = as_list(lm(ref_path(run_dir, 'S5_cleanup.mat'))['cleanup'])
    layer, se, cp = py_layer(cfg), py_se(cfg), py_cleanup(cfg)
    sites = [Site(int(c['ID']), np.r_[np.atleast_1d(c['pos_before']).astype(float)[:2], 0.0])
             for c in clean_ref]
    npc_segment_cleanup(ld, sites, layer, se, cp)
    rows = []
    for s, c in zip(sites, clean_ref):
        e = s.evaluation['NPCsegmentCleanup']
        dpos = float(np.hypot(*(s.pos[:2] - np.atleast_1d(c['pos_after'])[:2])))
        dR0 = abs(e['R0'] - c['R0']) if np.isfinite(c['R0']) else 0.0
        counts_eq = all(int(e[k]) == int(c[k]) for k in ('inside', 'outside', 'inring') if np.isfinite(c[k]))
        rows.append(dict(ID=int(c['ID']), locs_python=int(e['locs']), locs_smap=int(c['locs']),
                         centre_diff_nm=dpos, R0_python=float(e['R0']), R0_smap=float(c['R0']), dR0=dR0,
                         counts_equal=counts_eq, use_python=bool(s.annotation['use']), use_smap=bool(c['use'])))
    better = {'python': 0, 'smap': 0, 'equal': 0}
    rad = se.site_roi / 2 * 1.2
    for r, c in zip(rows, clean_ref):
        if r['centre_diff_nm'] < 0.5:
            continue
        L = get_site_locs(ld, Site(r['ID'], np.r_[np.atleast_1d(c['pos_before'])[:2], 0.0]), layer, rad,
                          ['xnm', 'ynm'])
        x, y = L['xnm'].astype(float), L['ynm'].astype(float)
        sm = np.atleast_1d(c['pos_after'])[:2]
        px = [s.pos[:2] for s in sites if s.ID == r['ID']][0]
        cs = np.sum(np.abs(np.hypot(x - sm[0], y - sm[1]) - cp.R))
        cpy = np.sum(np.abs(np.hypot(x - px[0], y - px[1]) - cp.R))
        r['centring_cost_python'], r['centring_cost_smap'] = float(cpy), float(cs)
        better['python' if cpy < cs - 1e-6 else ('smap' if cs < cpy - 1e-6 else 'equal')] += 1
    _write_csv(os.path.join(out, 'S6_cleanup_sites.csv'), rows)
    n = len(rows)
    same_fit = [r for r in rows if r['centre_diff_nm'] < 0.5 and r['dR0'] < 0.5]
    m = dict(n_sites=n,
             locs_per_site_identical=float(np.mean([r['locs_python'] == r['locs_smap'] for r in rows])),
             n_accepted_smap=int(sum(r['use_smap'] for r in rows)),
             n_accepted_python=int(sum(r['use_python'] for r in rows)),
             use_agreement=float(np.mean([r['use_python'] == r['use_smap'] for r in rows])),
             sites_with_same_fits=len(same_fit),
             use_agreement_when_fits_agree=float(np.mean([r['use_python'] == r['use_smap'] for r in same_fit]))
             if same_fit else np.nan,
             counts_agreement_when_fits_agree=float(np.mean([r['counts_equal'] for r in same_fit]))
             if same_fit else np.nan,
             centre_diff=absdiff_stats([r['centre_diff_nm'] for r in rows], np.zeros(n)),
             R0_diff=absdiff_stats([r['dR0'] for r in rows], np.zeros(n)),
             centring_fit_lower_cost=better,
             decisions_differing=[r for r in rows if r['use_python'] != r['use_smap']])
    det_ok = m['locs_per_site_identical'] == 1.0 and m['use_agreement_when_fits_agree'] == 1.0
    m['status'] = 'PASS' if (det_ok and len(same_fit) == n) else ('EQUIVALENT' if det_ok and m['use_agreement'] >= 0.99
                                                                  else 'FAIL')
    m['pass'] = m['status'] != 'FAIL'
    m['note'] = ('NPCsegmentCleanup runs three ring fits per site (lsqnonlin in SMAP, least_squares here) that '
                 'minimise sum |r - R|, a non-smooth objective with several local minima for sparse sites. '
                 'Deterministic parts (localizations per site; decisions when the fits agree) must be identical. '
                 'centring_fit_lower_cost counts, for sites where the centres differ, which program found the '
                 'lower objective value. Per-site table: S6_cleanup_sites.csv.')
    return m


def _tolerance(pid):
    if pid.endswith('weight'):
        return 1e-3
    if pid.endswith('rot') or pid.endswith('theta'):
        return 0.1
    return 0.1


def stage_S7(cfg, run_dir, out):
    allm = {}
    ld = _locdata_ref(cfg, run_dir)
    layer = py_layer(cfg)
    se = py_se(cfg)
    for lc in cfg['locmofit']:
        p = ref_path(run_dir, 'S6_%s.mat' % lc['name'])
        if not os.path.exists(p):
            allm[lc['name']] = dict(pass_=False, error='no MATLAB result')
            continue
        res = [x for x in as_list(lm(p)['res']) if not has_error(x)]
        rows = []
        for x in res:
            locs = {'xnm': np.atleast_1d(x['xnm']).astype(float), 'ynm': np.atleast_1d(x['ynm']).astype(float),
                    'locprecnm': np.atleast_1d(x['locprecnm']).astype(float)}
            f = py_locmofit(lc, se.site_roi)
            out_py = f.fit(locs)
            ref_vals = dict(zip([str(s) for s in np.atleast_1d(x['parId'])], np.atleast_1d(x['value'])))
            ref_fix = dict(zip([str(s) for s in np.atleast_1d(x['parId'])], np.atleast_1d(x['fix'])))
            row = {'ID': x['ID'], 'nlocs': x['nlocs'], 'LLfit_smap': x['LLfit'], 'LLfit_python': out_py['LLfit']}
            ok = abs(out_py['LLfit'] - x['LLfit']) < 1e-4
            for pid, v in ref_vals.items():
                if ref_fix[pid] or pid not in out_py:
                    continue
                pv = out_py[pid]
                row[pid + '_smap'] = v
                row[pid + '_python'] = pv
                ok &= abs(pv - v) < _tolerance(pid) or (pid.endswith('zrot') and _angle_close(pv, v))
            s = Site(int(x['ID']), np.r_[np.atleast_1d(x['pos'])[:2], 0.0])
            sl = get_site_locs(ld, s, layer, se.site_roi / 2, ['locprecnm'])
            row['nlocs_python_selection'] = len(sl['xnm'])
            row['ok'] = bool(ok)
            rows.append(row)
        _write_csv(os.path.join(out, 'S7_locmofit_%s.csv' % lc['name']), rows)
        frac = float(np.mean([r['ok'] for r in rows])) if rows else 0.0
        sel_eq = float(np.mean([r['nlocs_python_selection'] == r['nlocs'] for r in rows])) if rows else 0.0
        mm = dict(n_sites=len(rows), frac_sites_within_tolerance=frac,
                  site_locs_selection_identical=sel_eq,
                  LLfit_diff=absdiff_stats([r['LLfit_python'] for r in rows], [r['LLfit_smap'] for r in rows]))
        for pid in [k[:-5] for k in (rows[0].keys() if rows else []) if k.endswith('_smap') and k != 'LLfit_smap']:
            mm[pid] = absdiff_stats([r[pid + '_python'] for r in rows], [r[pid + '_smap'] for r in rows])
        mm['pass'] = bool(frac >= 0.95 and sel_eq >= 0.99)
        allm[lc['name']] = mm
        if rows:
            _fig_scatter(out, 'S7_%s' % lc['name'], rows)
    allm['pass'] = bool(all(v.get('pass', False) for k, v in allm.items() if isinstance(v, dict)))
    return allm


def _angle_close(a, b, tol=0.1):
    d = (a - b + 180) % 360 - 180
    return abs(d) < tol


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(str(r.get(k, '')) for k in keys) + '\n')


def _plt():
    import matplotlib
    if 'matplotlib.pyplot' not in sys.modules:
        try:
            matplotlib.use('Agg')
        except Exception:
            pass
    import matplotlib.pyplot as plt
    return plt


def _fig_scatter(out, name, rows):
    try:
        plt = _plt()
        pids = [k[:-5] for k in rows[0] if k.endswith('_smap')]
        fig, axs = plt.subplots(1, len(pids), figsize=(3.5 * len(pids), 3.5), squeeze=False)
        for ax, pid in zip(axs[0], pids):
            a = [r.get(pid + '_smap', np.nan) for r in rows]
            b = [r.get(pid + '_python', np.nan) for r in rows]
            ax.plot(a, b, '.')
            lo, hi = np.nanmin(a + b), np.nanmax(a + b)
            ax.plot([lo, hi], [lo, hi], 'k-', lw=0.5)
            ax.set_xlabel('SMAP')
            ax.set_ylabel('Python')
            ax.set_title(pid, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, name + '.png'), dpi=120)
        plt.close(fig)
    except Exception:
        pass


def matlab_problems(st):
    """List of (stage, error message) for the MATLAB stages that failed."""
    out = []
    for k, v in st.items():
        if isinstance(v, dict) and v.get('ok') is False:
            out.append((k, v.get('error', '?')))
    return out


def print_matlab_status(run_dir, echo=print):
    st = matlab_status(run_dir)
    log = os.path.join(run_dir, 'matlab_reference', 'matlab_log.txt')
    if not st:
        echo('\nMATLAB did not write status.json -- it did not start or crashed. See %s' % log)
        return st
    echo('\nSMAP (MATLAB) stages:')
    names = dict(MATLAB_STAGES)
    keys = [k for k, _ in MATLAB_STAGES] + sorted(k for k in st if k.startswith('S6_'))
    for k in keys:
        v = st.get(k)
        if not isinstance(v, dict):
            echo('  %-28s not run' % names.get(k, k))
        elif v.get('ok'):
            echo('  %-28s ok (%.0f s)' % (names.get(k, k), v.get('seconds', 0)))
        else:
            echo('  %-28s FAILED: %s' % (names.get(k, k), v.get('error', '?')))
    if matlab_problems(st):
        echo('Full MATLAB error messages: %s' % log)
        echo('Stages after a failed MATLAB stage have no SMAP reference and are skipped below.')
    return st


def compare_all(run_dir, stages=None, echo=print):
    """Run every Python stage on SMAP's inputs and compare. Writes
    <run_dir>/comparison/summary.json, report.md and figures.
    """
    with open(os.path.join(run_dir, 'config.json')) as fh:
        cfg = json.load(fh)
    out = os.path.join(run_dir, 'comparison')
    os.makedirs(out, exist_ok=True)
    summary = {'matlab_status': matlab_status(run_dir), 'stages': {}}
    for sid, title, fun, needs in STAGES:
        if stages and sid not in stages:
            continue
        if needs and not os.path.exists(ref_path(run_dir, needs)):
            summary['stages'][sid] = {'title': title, 'skipped': 'no SMAP reference (%s missing)' % needs}
            echo('%s %s: skipped -- no SMAP reference (%s missing)' % (sid, title, needs))
            continue
        echo('%s %s ...' % (sid, title))
        t = time.time()
        try:
            m = fun(cfg, run_dir, out)
        except Exception as e:
            m = {'pass': False, 'error': str(e), 'traceback': traceback.format_exc()}
            echo(m['traceback'])
        m['title'] = title
        m['seconds'] = time.time() - t
        summary['stages'][sid] = m
        echo('    -> %s: %s (%.0f s)' % (_status(m), headline(sid, m), m['seconds']))
        with open(os.path.join(out, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=1, default=_jsonable)
    write_report(cfg, summary, out)
    return summary


def _status(m):
    if 'skipped' in m:
        return 'skipped'
    st = m.get('status') or {True: 'PASS', False: 'FAIL', None: 'info'}.get(m.get('pass'), '?')
    return {'PASS': 'PASS (identical)', 'EQUIVALENT': 'PASS (equivalent)', 'PARTIAL': 'PASS (equivalent)'}.get(st, st)


def _pct(x):
    return '%.2f %%' % (100 * x) if x < 1 else '100 %'


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def _rng(v):
    return [(-BIG if x == -np.inf else (BIG if x == np.inf else x)) for x in v]


def make_config(S, sml_file, smap_dir, locmofit_max_sites=None):
    """Validation config (config.json) from the SETTINGS of 01_extract.py."""
    filters = {'locprecnm': _rng(S['FILTER_LOCPREC_NM']), 'PSFxnm': _rng(S['FILTER_PSF_NM']),
               'frame': _rng(S['FILTER_FRAME'])}
    if S.get('FILTER_LLREL') is not None:
        filters['LLrel'] = _rng(S['FILTER_LLREL'])
    return {
        'smap_dir': smap_dir, 'input_sml': sml_file,
        'grouping': {'dx': S['GROUP_DX_NM'], 'dt': S['GROUP_DT_FRAMES']},
        'layer1': {'grouped': bool(S['LAYER_GROUPED']), 'filters': filters},
        'render': {'colormode': S['COLORMODE'], 'renderfield': S['COLORFIELD'], 'lut': S['LUT']},
        'se': {'cell_pixelsize': S['CELL_PIXELSIZE_NM'], 'cell_fov': S['CELL_FOV_NM'],
               'site_pixelsize': S['SITE_PIXELSIZE_NM'], 'site_fov': S['SITE_FOV_NM'],
               'site_roi': S['SITE_ROI_NM']},
        'segment': {'cutoff': S['SEGMENT_CUTOFF'], 'diameter': S['SEGMENT_DIAMETER_NM'],
                    'rim': S['SEGMENT_RIM_NM']},
        'cleanup': {'R': S['CLEANUP_R_NM'], 'dR': S['CLEANUP_DR_NM'], 'center': bool(S['CLEANUP_CENTER']),
                    'inside': S['CLEANUP_INSIDE'], 'outside': S['CLEANUP_OUTSIDE'],
                    'radiusrange': list(S['CLEANUP_RADIUS_RANGE_NM']), 'minsize': S['CLEANUP_MINSIZE'],
                    'minlocs': S['CLEANUP_MINLOCS'], 'maxPSF': S['CLEANUP_MAXPSF_NM']},
        'locmofit': [S['LOCMOFIT_MODEL']],
        'locmofit_sites': 'used',
        'locmofit_max_sites': locmofit_max_sites,
    }


def _locdata_ref(cfg, run_dir):
    """SMAP's loaded and grouped localizations (stage S2), as Python input."""
    r = lm(ref_path(run_dir, 'S2_loaded.mat'))
    return LocData(locdict(r['loc']), cfg['grouping']['dx'], cfg['grouping']['dt'],
                   grouploc=locdict(r['grouploc']))


def sites_config(cfg):
    """npc_pipeline.smap_sites.SmapSiteConfig equal to the validation config (no dedup).
    """
    from npc_pipeline.smap_sites import SmapSiteConfig
    f = {k: (v[0] if v[0] > -BIG / 10 else -np.inf, v[1] if v[1] < BIG / 10 else np.inf)
         for k, v in cfg['layer1']['filters'].items()}
    se, sg, c, r = cfg['se'], cfg['segment'], cfg['cleanup'], cfg['render']
    return SmapSiteConfig(
        group_dx_nm=cfg['grouping']['dx'], group_dt_frames=cfg['grouping']['dt'],
        layer_grouped=cfg['layer1']['grouped'], filter_locprec_nm=f['locprecnm'],
        filter_psf_nm=f['PSFxnm'], filter_frame=f['frame'], filter_llrel=f.get('LLrel'),
        colormode=r['colormode'], colorfield=r['renderfield'], lut=r['lut'],
        cell_pixelsize_nm=se['cell_pixelsize'], cell_fov_nm=se['cell_fov'],
        site_pixelsize_nm=se['site_pixelsize'], site_fov_nm=se['site_fov'], site_roi_nm=se['site_roi'],
        segment_cutoff=sg['cutoff'], segment_diameter_nm=sg['diameter'], segment_rim_nm=sg['rim'],
        cleanup_R_nm=c['R'], cleanup_dR_nm=c['dR'], cleanup_center=c['center'],
        cleanup_inside=c['inside'], cleanup_outside=c['outside'],
        cleanup_radius_range_nm=tuple(c['radiusrange']), cleanup_minsize=c['minsize'],
        cleanup_minlocs=c['minlocs'], cleanup_maxpsf_nm=c['maxPSF'], dedup_nm=0.0,
        locmofit_model=cfg['locmofit'][0])


def stage_S8(cfg, run_dir, out):
    """End to end: npc_pipeline.smap_sites on the same sml file, compared with SMAP's
    accepted sites (NPCsegmentCleanup) and LocMoFit results.
    """
    from npc_pipeline import smap_sites
    r = smap_sites.run_camr(cfg['input_sml'], sites_config(cfg))
    clean_ref = as_list(lm(ref_path(run_dir, 'S5_cleanup.mat'))['cleanup'])
    acc = np.array([np.atleast_1d(x['pos_after'])[:2] for x in clean_ref if x['use']]).reshape(-1, 2)
    P = np.array([[g['x0'], g['y0']] for g in r['geom']]).reshape(-1, 2)
    m = dict(n_accepted_smap=len(acc), n_accepted_python=len(P))
    for tol in (0.5, 5.0):
        d, _ = _match(P, acc, tol)
        m['n_matched_%gnm' % tol] = int(np.isfinite(d).sum()) if len(d) else 0
    lc = cfg['locmofit'][0]
    p = ref_path(run_dir, 'S6_%s.mat' % lc['name'])
    rows = []
    if os.path.exists(p) and len(P):
        ref = [x for x in as_list(lm(p)['res']) if not has_error(x)]
        S = np.array([np.atleast_1d(x['pos'])[:2] for x in ref]).reshape(-1, 2)
        d, j = _match(P, S, 0.5)
        mle = {mm['seed_index']: mm for mm in r['mle']}
        for i, (dd, jj) in enumerate(zip(d, j)):
            if not np.isfinite(dd) or i not in mle:
                continue
            x = ref[jj]
            vals = dict(zip([str(s) for s in np.atleast_1d(x['parId'])], np.atleast_1d(x['value'])))
            g = r['geom'][i]
            rows.append(dict(ID=int(x['ID']),
                             dx_smap=vals.get('m1.lPar.x', np.nan), dx_python=mle[i]['x0'] - g['x0'],
                             dy_smap=vals.get('m1.lPar.y', np.nan), dy_python=mle[i]['y0'] - g['y0'],
                             radius_smap=vals.get('m1.mPar.radius', np.nan), radius_python=mle[i]['R']))
        _write_csv(os.path.join(out, 'S8_end_to_end_locmofit.csv'), rows)
    if rows:
        cen = [np.hypot(q['dx_smap'] - q['dx_python'], q['dy_smap'] - q['dy_python']) for q in rows]
        m['locmofit_centre_diff_nm'] = absdiff_stats(cen, np.zeros(len(cen)))
        m['locmofit_radius_diff_nm'] = absdiff_stats([q['radius_python'] for q in rows],
                                                     [q['radius_smap'] for q in rows])
        m['locmofit_sites_compared'] = len(rows)
    m['pass'] = None
    return m


def run_matlab(cfg_path, matlab_exe, timeout_s=6 * 3600, echo=print, mode='batch'):
    """Run validation/matlab/smap_reference.m ('batch': matlab -batch; 'desktop': matlab -r).
    """
    mdir = os.path.join(HERE, 'matlab').replace('\\', '/')
    cp = cfg_path.replace('\\', '/')
    call = "addpath('%s'); smap_reference('%s')" % (mdir, cp)
    run_dir = os.path.dirname(cfg_path)
    log = os.path.join(run_dir, 'matlab_reference', 'matlab_log.txt')
    t0 = time.time()
    if mode == 'batch':
        cmd = [matlab_exe, '-batch', call]
    else:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = [matlab_exe, '-nosplash', '-wait', '-logfile', os.path.join(run_dir, 'matlab_console.txt'),
               '-r', "try, %s, catch e, disp(getReport(e)), end, exit" % call]
    echo('Running: ' + ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, errors='replace')
    try:
        for line in proc.stdout:
            echo(line.rstrip())
            if time.time() - t0 > timeout_s:
                proc.kill()
                echo('MATLAB stopped after timeout')
                break
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        raise
    if mode != 'batch' and os.path.exists(log):
        with open(log, errors='replace') as fh:
            echo(fh.read())
    echo('MATLAB finished (exit code %s) in %.1f min; log: %s' % (proc.returncode, (time.time() - t0) / 60, log))
    return proc.returncode


CRITERIA = {
    'S2': 'identical grouping (values within 0.01 nm / 0.01 %); identical layer-1 selection',
    'S4': 'cell images equal within 1e-3 of the image maximum, same colour range',
    'S5': "identical sites (same number, positions within 0.5 nm) from SMAP's cell images",
    'S6': 'localizations per site and decisions identical where the ring fits agree (PASS); '
          '>= 99 % of decisions identical (equivalent)',
    'S7': '>= 95 % of sites with every free parameter within tolerance '
          '(0.1 nm, 0.1 deg, 0.001 for weights) and |dLLfit| < 1e-4',
    'S8': 'informative: accepted sites and LocMoFit results of npc_pipeline.smap_sites vs SMAP',
}

STAGES = [('S2', 'Grouping and layer filters', stage_S2, 'S2_loaded.mat'),
          ('S4', 'Rendering of cell images', stage_S4, 'S4_cells.mat'),
          ('S5', 'segmentNPC', stage_S5, 'S4_sites.mat'),
          ('S6', 'NPCsegmentCleanup', stage_S6, 'S5_cleanup.mat'),
          ('S7', 'LocMoFit', stage_S7, 'S5_cleanup.mat'),
          ('S8', 'End to end (npc_pipeline.smap_sites)', stage_S8, 'S5_cleanup.mat')]

MATLAB_STAGES = [('init', 'start SMAP'), ('S2_load', 'S2 load + filters'),
                 ('S4_segment', 'S4/S5 cells + segmentNPC'), ('S5_cleanup', 'S6 NPCsegmentCleanup')]


def headline(sid, m):
    """One line with the agreement figures of a stage."""
    try:
        if 'skipped' in m or m.get('error'):
            return m.get('skipped') or m.get('error')
        if sid == 'S2':
            return '{:,} groups and {:,} layer-1 localizations'.format(m['n_groups_smap'], m['layer1_n_smap'])
        if sid == 'S4':
            return '%d of %d cell images bit-identical' % (m.get('images_bit_identical', 0), m['n_cells'])
        if sid == 'S5':
            return '%d of %d sites identical' % (m['n_matched_0p5nm'], m['n_smap'])
        if sid == 'S6':
            return ('%s of %d accept/reject decisions identical (accepted: SMAP %d, Python %d)'
                    % (_pct(m['use_agreement']), m['n_sites'], m['n_accepted_smap'], m['n_accepted_python']))
        if sid == 'S7':
            parts = []
            for k, v in m.items():
                if isinstance(v, dict) and 'n_sites' in v:
                    parts.append('%s %d of %d sites identical' % (k, round(v['frac_sites_within_tolerance'] *
                                                                         v['n_sites']), v['n_sites']))
            return '; '.join(parts)
        if sid == 'S8':
            s = ('accepted sites: SMAP %d, Python %d, matched within 0.5 nm %d'
                 % (m['n_accepted_smap'], m['n_accepted_python'], m['n_matched_0.5nm']))
            if 'locmofit_centre_diff_nm' in m:
                s += ('; LocMoFit on %d matched sites: centre diff median %.3f nm, radius diff median %.3f nm'
                      % (m['locmofit_sites_compared'], m['locmofit_centre_diff_nm']['median'],
                         m['locmofit_radius_diff_nm']['median']))
            return s
    except Exception as e:
        return '(%s)' % e
    return ''


def print_summary(summary, run_dir, echo=print):
    echo('')
    for sid, m in summary['stages'].items():
        echo('%-3s %-38s %-18s %s' % (sid, m['title'], _status(m), headline(sid, m)))
        if m.get('warning'):
            echo('      WARNING: %s' % m['warning'])
    echo('Report: %s' % os.path.join(run_dir, 'comparison', 'report.md'))


def write_report(cfg, summary, out):
    L = ['# Validation of npc_pipeline.smap against SMAP', '',
         'Run directory: `%s`' % os.path.dirname(out), '',
         'MATLAB: %s' % summary['matlab_status'].get('matlab_version', '?'), '',
         'Input: `%s`' % cfg.get('input_sml', ''), '',
         'Stages S2-S7 receive exactly the input SMAP used for that stage, so differences do not '
         'propagate. S8 runs npc_pipeline.smap_sites on the same file.', '']
    probs = matlab_problems(summary['matlab_status'])
    if not summary['matlab_status']:
        L += ['**MATLAB did not run or stopped before writing status.json.** See '
              '`matlab_reference/matlab_log.txt`.', '']
    for k, e in probs:
        L += ['**SMAP stage %s failed in MATLAB:** %s' % (k, e), '']
    L += ['| stage | | result | agreement | criterion |', '|---|---|---|---|---|']
    for sid, title, _, _ in STAGES:
        m = summary['stages'].get(sid)
        if m is None:
            continue
        res = _status(m)
        res = res if res in ('skipped', 'info') else '**%s**' % res
        L.append('| %s | %s | %s | %s | %s |' % (sid, title, res, headline(sid, m), CRITERIA[sid]))
    L += ['', '## Details', '']
    for sid, title, _, _ in STAGES:
        m = summary['stages'].get(sid)
        if m is None:
            continue
        L += ['### %s %s' % (sid, title), '```',
              json.dumps({k: v for k, v in m.items() if k != 'traceback'}, indent=1, default=_jsonable)[:6000],
              '```']
        if 'traceback' in m:
            L += ['```', m['traceback'], '```']
    L += ['', '## Settings used', '```', json.dumps(cfg, indent=1)[:8000], '```']
    with open(os.path.join(out, 'report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
