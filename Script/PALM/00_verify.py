# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check the coordinate conventions of a dataset: bead offset direction, drift sign,
CamT grid consistency (SMAP vs GDSC) and pixel size.

Usage: python 00_verify.py [--data DIR] [--limit N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() \
    else os.getcwd()
sys.path.insert(0, HERE)

from npc_pipeline import io_smap, drift, camt, register as R, roi as ROI
from npc_pipeline.config import RegisterConfig

PASS = '  PASS'
FAIL = '  FAIL'
WARN = '  WARN'


def resolve_dir(parent, name):
    direct = os.path.join(parent, name)
    if os.path.isdir(direct):
        return direct
    try:
        for e in os.listdir(parent):
            if e.lower() == name.lower() and os.path.isdir(
                    os.path.join(parent, e)):
                return os.path.join(parent, e)
    except OSError:
        pass
    return direct


def find_cells(data):
    out = []
    for pat in (os.path.join(data, '*', '*', '*', '*'),
                os.path.join(data, '*', '*', '*')):
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and 'alignment' not in d.lower() \
                    and glob.glob(os.path.join(d, '*_sml.mat')):
                out.append(d)
        if out:
            break
    return out


def check_offset_direction(data, limit=2):
    """Apply the fitted transform to CamT beads; the distance to CamR beads
    must fall. Applying its inverse must make things worse.
    """
    from scipy.spatial import cKDTree
    print('\n1. OFFSET DIRECTION — does the bead transform move CamT onto CamR?')
    cfg = RegisterConfig(mirror='none', model='affine', min_beads=6,
                         bead_threshold_sigma=8.0)
    n_ok = n_run = 0
    for session in sorted(glob.glob(os.path.join(data, '*'))):
        for d, _ in ROI.alignment_rois(session, 'R')[:limit]:
            try:
                A = R.beads_from_image(
                    R.sum_projection(os.path.join(d, 'ROI-T.tif'), 30),
                    cfg) * cfg.cam_pixelsize_nm
                B = R.beads_from_image(
                    R.sum_projection(os.path.join(d, 'ROI-R.tif'), 30),
                    cfg) * cfg.cam_pixelsize_nm
                a, b = R.pair_points(A, B, cfg.bead_max_pair_nm)
                if len(a) < cfg.min_beads:
                    continue
                T = R.solve_transform(a, b, 'affine')
            except Exception as e:
                print(f'   {os.path.basename(d)}: skipped ({type(e).__name__})')
                continue
            lin, tr = R.split_transform(T)
            inv = np.linalg.inv(lin)
            fx, fy = T.apply(A[:, 0], A[:, 1])
            gx = inv[0, 0] * (A[:, 0] - tr[0]) + inv[0, 1] * (A[:, 1] - tr[1])
            gy = inv[1, 0] * (A[:, 0] - tr[0]) + inv[1, 1] * (A[:, 1] - tr[1])
            tree = cKDTree(B)
            before = np.median(tree.query(A, k=1)[0])
            after = np.median(tree.query(np.c_[fx, fy], k=1)[0])
            reverse = np.median(tree.query(np.c_[gx, gy], k=1)[0])
            ok = after < before and after < reverse
            n_run += 1
            n_ok += ok
            print(f'   {os.path.basename(session)[:20]:20s} '
                  f'{os.path.basename(d):14s} '
                  f'before {before:6.1f}  after {after:6.1f}  '
                  f'inverse {reverse:7.1f} nm')
    if n_run == 0:
        print(WARN + ' no usable bead acquisition found')
        return None
    print((PASS if n_ok == n_run else FAIL) +
          f' {n_ok}/{n_run} bead runs: CamT moves onto CamR')
    return n_ok == n_run


def check_drift_sign(data, limit=3):
    """corrected = raw - dxt(frame)."""
    print('\n2. DRIFT SIGN — is corrected = raw - dxt(frame)?')
    n_ok = n_run = 0
    for c in find_cells(data):
        rawp = os.path.join(c, 'ROI-R_sml.mat')
        corp = io_smap.find_sml(c, prefer_driftc=True, channel='R')
        if not os.path.exists(rawp) or corp is None or 'driftc' not in corp:
            continue
        try:
            raw = io_smap.read_sml(rawp, fields=['xnm', 'ynm', 'frame'])
            cor = io_smap.read_sml(corp, fields=['xnm', 'ynm', 'frame'])
            di = io_smap.read_driftinfo(corp)
            if di is None:
                continue
            fr = raw['frame'].astype(int)
            fc = cor['frame'].astype(int)
            frames = np.intersect1d(np.unique(fr), np.unique(fc))
            frames = frames[(frames >= 1) & (frames <= len(di['dxt']))]
            obs, sto = [], []
            for f in frames:
                m, k = fr == f, fc == f
                if m.sum() < 25 or m.sum() != k.sum():
                    continue
                obs.append([cor['xnm'][k].mean() - raw['xnm'][m].mean(),
                            cor['ynm'][k].mean() - raw['ynm'][m].mean()])
                sto.append([di['dxt'][f - 1], di['dyt'][f - 1]])
            if len(obs) < 20:
                continue
            obs, sto = np.array(obs), np.array(sto)
            sx = np.polyfit(sto[:, 0], obs[:, 0], 1)[0]
            sy = np.polyfit(sto[:, 1], obs[:, 1], 1)[0]
            rx = np.corrcoef(sto[:, 0], obs[:, 0])[0, 1]
            ok = abs(sx + 1) < 0.05 and abs(sy + 1) < 0.05 and rx < -0.99
            n_run += 1
            n_ok += ok
            print(f'   {os.path.basename(c):16s} {len(obs):4d} frames   '
                  f'slope x {sx:+.4f}  y {sy:+.4f}   r {rx:+.5f}')
        except Exception as e:
            print(f'   {os.path.basename(c):16s} skipped ({type(e).__name__})')
        if n_run >= limit:
            break
    if n_run == 0:
        print(WARN + ' no acquisition has BOTH a raw and a driftc CamR file, '
                     'so the sign cannot be checked here')
        return None
    print((PASS if n_ok == n_run else FAIL) +
          f' {n_ok}/{n_run}: slope -1, corrected = raw - dxt')
    return n_ok == n_run


def _match_median(P, Q, pf, qf, n_frames=40, tol=200.0):
    """Median (P - Q) over molecules matched within a frame. Matching is done
    per frame so identical coordinates in different frames cannot cross-match.
    """
    from scipy.spatial import cKDTree
    dx, dy = [], []
    for f in np.unique(qf)[:n_frames]:
        a, b = pf == f, qf == f
        if not a.any() or not b.any():
            continue
        d, j = cKDTree(Q[b]).query(P[a], k=1)
        k = d <= tol
        if k.any():
            dx += list(P[a][k, 0] - Q[b][j[k], 0])
            dy += list(P[a][k, 1] - Q[b][j[k], 1])
    if not dx:
        return None
    return float(np.median(dx)), float(np.median(dy)), len(dx)


def check_camt_consistency(data, limit=3):
    """Every CamT product must land on SMAP's coordinate grid."""
    print('\n3. CamT COORDINATE CONSISTENCY — do all CamT products sit on '
          'SMAP\'s grid?')
    n_ok = n_run = 0
    for c in find_cells(data):
        if not os.path.exists(os.path.join(c, camt.SOURCES['smap'])):
            continue
        try:
            good = camt.load_camt(c)
            bad = camt.load_camt(c, pixel_offset=0.0)   # the mistake, for scale
        except Exception as e:
            print(f'   {os.path.basename(c):16s} skipped ({type(e).__name__})')
            continue
        if 'smap' not in good:
            continue
        S = np.c_[good['smap']['xnm'], good['smap']['ynm']]
        sf = good['smap']['frame']
        lines, ok_here, any_here = [], True, False
        for key, label in (('gdsc', 'GDSC single points'),
                           ('traces', 'GDSC traces')):
            if key not in good:
                lines.append(f'      {label:20s} not present')
                continue
            m = _match_median(np.c_[good[key]['xnm'], good[key]['ynm']], S,
                              good[key]['frame'], sf)
            b = _match_median(np.c_[bad[key]['xnm'], bad[key]['ynm']], S,
                              bad[key]['frame'], sf)
            if m is None:
                lines.append(f'      {label:20s} no molecules matched')
                continue
            any_here = True
            resid = max(abs(m[0]), abs(m[1]))
            hit = resid < 24.0
            ok_here &= hit
            note = '' if hit else '   <-- looks like a grid offset'
            lines.append(f'      {label:20s} dx {m[0]:+6.1f}  dy {m[1]:+6.1f} nm'
                         f'  (n {m[2]:4d}){note}')
            if b is not None:
                lines.append(f'      {"":20s} without the +0.5 px it would be '
                             f'dx {b[0]:+6.1f}  dy {b[1]:+6.1f}')
        if not any_here:
            continue
        n_run += 1
        n_ok += ok_here
        print(f'   {os.path.basename(c)}')
        for ln in lines:
            print(ln)
        if n_run >= limit:
            break
    if n_run == 0:
        print(WARN + ' no acquisition has both a SMAP and a GDSC CamT export')
        return None
    print((PASS if n_ok == n_run else FAIL) +
          f' {n_ok}/{n_run}: every GDSC CamT product agrees with SMAP to well '
          f'inside the {camt.GDSC_PIXEL_OFFSET * camt.DEFAULT_NM_PER_PX:.1f} nm '
          f'half-pixel')
    return n_ok == n_run


def check_pixel_size(data, limit=3):
    """Do the two FITTERS agree on nm per pixel?"""
    print('\n4. PIXEL SIZE — do the two fitters agree, and match the pipeline?')
    used = camt.DEFAULT_NM_PER_PX
    seen = {}
    for c in find_cells(data)[:limit]:
        p = os.path.join(c, 'modality.json')
        if os.path.exists(p):
            try:
                v = json.load(open(p, encoding='utf-8', errors='replace')).get(
                    'pixel size')
                if v:
                    seen.setdefault('acquisition (modality.json)', set()).add(
                        round(float(v), 3))
            except Exception:
                pass
        g = os.path.join(c, camt.SOURCES['gdsc'])
        for path in (g, os.path.join(c, camt.SOURCES['traces'])):
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    head = f.read(8192)
                i = head.find('nmPerPixel')
                if i > 0:
                    v = float(head[i:i + 40].split(':')[1].split(',')[0])
                    seen.setdefault('GDSC calibration', set()).add(round(v, 3))
            except Exception:
                pass
        s = os.path.join(c, camt.SOURCES['smap'])
        if os.path.exists(s):
            try:
                L = io_smap.read_sml(s, fields=['xnm', 'xpix'])
                if 'xpix' in L and len(L) > 100:
                    k = np.polyfit(L['xpix'], L['xnm'], 1)[0]
                    seen.setdefault('SMAP xnm/xpix', set()).add(round(k, 3))
            except Exception:
                pass
    if not seen:
        print(WARN + ' no pixel-size source found')
        return None
    for k, v in seen.items():
        note = '   (nominal, not what the fits used)' if 'modality' in k else ''
        print(f'   {k:28s} {[round(x, 3) for x in sorted(v)]}{note}')
    print(f'   {"pipeline uses":28s} [{used}]')

    fitters = {x for k, v in seen.items() if 'modality' not in k for x in v}
    if not fitters:
        print(WARN + ' neither fitter reported a pixel size')
        return None
    if len(fitters) > 1:
        print(FAIL + f' the fitters disagree: {sorted(fitters)}. One channel '
                     f'is scaled relative to the other.')
        return False
    v = fitters.pop()
    if abs(v - used) > 1e-6:
        print(FAIL + f' the fitters say {v} but the pipeline applies {used}')
        return False
    print(PASS + f' SMAP and GDSC both {v} nm/px, matching the pipeline')
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=None)
    ap.add_argument('--limit', type=int, default=3,
                    help='acquisitions to check per test')
    a = ap.parse_args(argv)
    data = a.data or resolve_dir(HERE, 'data')

    print('=' * 74)
    print(f'  conventions check on {data}')
    print('=' * 74)
    if not os.path.isdir(data):
        print(f'no data folder at {data}')
        return 1

    results = [check_offset_direction(data, a.limit),
               check_drift_sign(data, a.limit),
               check_camt_consistency(data, a.limit),
               check_pixel_size(data, a.limit)]

    print('\n' + '=' * 74)
    bad = [i + 1 for i, r in enumerate(results) if r is False]
    skipped = [i + 1 for i, r in enumerate(results) if r is None]
    if bad:
        print(f'  FAILED checks: {bad} — do not trust the output until these '
              f'are resolved')
    else:
        print('  no failed checks')
    if skipped:
        print(f'  inconclusive: {skipped} (see the notes above)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
