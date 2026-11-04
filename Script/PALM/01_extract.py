# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-colour PALM NPC analysis: data/ -> results/ (CSV) + interactive_html/.

Per acquisition: bead registration (camera offset), CamR drift applied to CamT,
SMAP segmentNPC / NPCsegmentCleanup / LocMoFit on CamR, CamT (SMAP and GDSC
localizations and traces) assigned to pores, radial and 8-fold analysis.

Usage: python 01_extract.py [options]   (or edit SETTINGS and run in Spyder)
"""
import argparse
import glob
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() \
    else os.getcwd()
sys.path.insert(0, HERE)

import npc_pipeline as P
from npc_pipeline import (io_smap, radial, register as R, camt, traces,
                          eightfold, roi as ROI, explorer, smap_sites, drift)
from npc_pipeline.config import RegisterConfig, Molecules


# =============================================================================
#  SETTINGS  (command-line options override these; see --help)
# =============================================================================
# Molecules
CAMT_MOLECULE = 'Nup62'      # target channel (CamT)
CAMR_MOLECULE = 'Nup96'      # reference scaffold (CamR), segmented into rings
REF_RADIUS_NM = 53.7         # literature ring radius of the scaffold

# Folders, relative to this file (case-insensitive)
DATA_DIR = 'data'
RESULTS_DIR = 'results'
HTML_DIR = 'interactive_html'

# CamR frame merging: N raw ROI-R frames summed into one frame before the SMAP
# fit (1 = not merged). The CamR drift is applied to CamT by time.
ROI_R_MERGE_FRAMES = 1
MERGE_CHECK_TOL = 0.05       # raw / CamR frames must round to N, within 5 %
ROI_T_FILE = 'ROI-T.tif'     # used for its frame count
TIMESTAMP_FILE = 'Timestamp.txt'

# Registration (camera offset from beads)
ROI_ANCHOR = 'centred'       # 'centred' | 'corner' | 'unknown'
REG_METHOD = 'joint'         # 'joint' | 'best'

# Two-colour analysis
MIN_TRACE_LENGTH = 6
ASSIGN_RADIUS_NM = 80.0
OCCUPANCY = 0.5
LINK_DISTANCE_NM = 200.0     # linking SMAP CamT localizations into traces
MAX_FRAME_GAP = 1
MIN_LOCS_CELL = 2000         # CamR localizations after filtering, else skipped
MERGE_RADIUS_NM = 120.0
FOLD = 8
PHASE_RMIN_NM = 25.0
PHASE_RMAX_NM = 90.0
LIMIT = 0                    # max acquisitions per session; 0 = all

# SMAP: grouping and render layer 1
GROUP_DX_NM = 35.0
GROUP_DT_FRAMES = 1          # in CamR frames
LAYER_GROUPED = True
FILTER_LOCPREC_NM = (0, 25)
FILTER_PSF_NM = (0, 150)
FILTER_FRAME = (0, float('inf'))
FILTER_LLREL = None          # e.g. (-1.5, 0)
COLORMODE = 'field'          # 'field' or 'normal'
COLORFIELD = 'PSFxnm'
LUT = 'jet'

# SMAP: ROI manager, segmentNPC, NPCsegmentCleanup
CELL_PIXELSIZE_NM = 15.0     # the segmentNPC cutoff is absolute on this render
CELL_FOV_NM = 5000.0
SITE_PIXELSIZE_NM = 3.0
SITE_FOV_NM = 400.0
SITE_ROI_NM = 220.0
SEGMENT_CUTOFF = 0.06
SEGMENT_DIAMETER_NM = 100.0
SEGMENT_RIM_NM = 20.0
CLEANUP_R_NM = 55.0
CLEANUP_DR_NM = 15.0
CLEANUP_CENTER = True
CLEANUP_INSIDE = 0.3
CLEANUP_OUTSIDE = 0.75
CLEANUP_RADIUS_RANGE_NM = (45.0, 65.0)
CLEANUP_MINSIZE = 25.0
CLEANUP_MINLOCS = 10
CLEANUP_MAXPSF_NM = 150.0
DEDUP_NM = 30.0              # merge pore centres closer than this; 0 = off

# SMAP: LocMoFit (parameter table as in the LocMoFit GUI; lb/ub relative)
LOCMOFIT_MODEL = {
    'name': 'ring2D',
    'models': [{'class': 'ring2D', 'layer': 1}],
    'pars': [{'id': 'm1.mPar.radius', 'fix': False, 'value': 53.7, 'lb': -20,
              'ub': 20, 'min': 20, 'max': 100},
             {'id': 'm1.lPar.zrot', 'fix': True, 'value': 0},
             {'id': 'm1.lPar.weight', 'fix': True, 'value': 1}]}
# =============================================================================


def smap_site_config(args):
    """SETTINGS (+ command-line overrides) -> smap_sites.SmapSiteConfig."""
    import copy
    model = copy.deepcopy(LOCMOFIT_MODEL)
    if args.fix_radius:
        for p in model['pars']:
            if p['id'] == 'm1.mPar.radius':
                p['fix'] = True
                p['value'] = float(args.ref_radius)
    return smap_sites.SmapSiteConfig(
        group_dx_nm=GROUP_DX_NM, group_dt_frames=GROUP_DT_FRAMES,
        layer_grouped=LAYER_GROUPED, filter_locprec_nm=FILTER_LOCPREC_NM,
        filter_psf_nm=FILTER_PSF_NM, filter_frame=FILTER_FRAME,
        filter_llrel=FILTER_LLREL, colormode=COLORMODE, colorfield=COLORFIELD,
        lut=LUT,
        cell_pixelsize_nm=(args.pixrec if args.pixrec is not None
                           else CELL_PIXELSIZE_NM),
        cell_fov_nm=CELL_FOV_NM, site_pixelsize_nm=SITE_PIXELSIZE_NM,
        site_fov_nm=SITE_FOV_NM, site_roi_nm=SITE_ROI_NM,
        segment_cutoff=(args.cutoff if args.cutoff is not None
                        else SEGMENT_CUTOFF),
        segment_diameter_nm=SEGMENT_DIAMETER_NM, segment_rim_nm=SEGMENT_RIM_NM,
        cleanup_R_nm=CLEANUP_R_NM, cleanup_dR_nm=CLEANUP_DR_NM,
        cleanup_center=CLEANUP_CENTER, cleanup_inside=CLEANUP_INSIDE,
        cleanup_outside=CLEANUP_OUTSIDE,
        cleanup_radius_range_nm=CLEANUP_RADIUS_RANGE_NM,
        cleanup_minsize=CLEANUP_MINSIZE, cleanup_minlocs=CLEANUP_MINLOCS,
        cleanup_maxpsf_nm=CLEANUP_MAXPSF_NM, dedup_nm=args.dedup,
        locmofit_model=model)


def resolve_dir(parent, name, create=False):
    """Case-insensitive lookup, so 'data' finds 'Data' on any filesystem."""
    direct = os.path.join(parent, name)
    if os.path.isdir(direct):
        return direct
    try:
        for entry in os.listdir(parent):
            if entry.lower() == name.lower():
                p = os.path.join(parent, entry)
                if os.path.isdir(p):
                    return p
    except OSError:
        pass
    if create:
        os.makedirs(direct, exist_ok=True)
    return direct


def find_sessions(data_dir):
    """Every session folder in data/."""
    out = []
    for d in sorted(glob.glob(os.path.join(data_dir, '*'))):
        if not os.path.isdir(d):
            continue
        if find_cells(d):
            out.append(d)
    return out


def find_cells(session):
    """Acquisition folders holding CamR localizations."""
    seen, out = set(), []
    for pattern in (os.path.join(session, '*', '*', '*'),
                    os.path.join(session, '*', '*')):
        for d in sorted(glob.glob(pattern)):
            if not os.path.isdir(d) or d in seen:
                continue
            parts = {p.lower() for p in d.split(os.sep)}
            if 'alignment' in parts:
                continue
            if glob.glob(os.path.join(d, '*_sml.mat')):
                seen.add(d)
                out.append(d)
    return sorted(out)


def find_alignment_dirs(session):
    """Bead acquisitions, paired with their ROI-R and ROI-T image crops."""
    out = []
    for d, gR in ROI.alignment_rois(session, channel='R'):
        if not (os.path.exists(os.path.join(d, 'ROI-T.tif'))
                and os.path.exists(os.path.join(d, 'ROI-R.tif'))):
            continue
        out.append((d, gR, ROI.acquisition_roi(d, channel='T')))
    return out


class SessionRegistration:
    """The bead registration for one session, kept ROI-aware."""

    def __init__(self, linear, residual_nm, n_sets, trans_by_roi, rois,
                 rois_t=None, pixel_nm=97.0, optics_spread_nm=None):
        self.linear = linear
        self.residual_nm = residual_nm
        self.n_sets = n_sets
        self.trans_by_roi = trans_by_roi     # {(w, h): mean translation (2,)}
        self.rois = rois                     # {(w, h): [RoiGeometry, ...]} R
        self.rois_t = rois_t or {}           # {(w, h): [RoiGeometry, ...]} T
        self.pixel_nm = pixel_nm
        self.optics_spread_nm = optics_spread_nm

    @property
    def bead_rois(self):
        return sorted(self.rois)

    def representative_roi(self):
        """One ROI to report when nothing more specific is asked for."""
        if not self.rois:
            return ROI.RoiGeometry(source='unknown')
        size = max(self.rois, key=lambda k: len(self.rois[k]))
        return self.rois[size][0]

    def for_cell(self, cell_roi, cell_roi_t, cfg_reg):
        """(bead affine, bead ROI, mode, reason) for one acquisition."""
        if self.linear is None or not self.rois:
            return None, ROI.RoiGeometry(source='unknown'), \
                ROI.UNREGISTERED, 'no usable bead run in this session'

        match = None
        if cell_roi.known:
            for size in self.rois:
                if abs(size[0] - cell_roi.width) <= cfg_reg.roi_tolerance_px \
                        and abs(size[1] - cell_roi.height) \
                        <= cfg_reg.roi_tolerance_px:
                    match = size
                    break

        bead_roi = (self.rois[match][0] if match is not None
                    else self.representative_roi())
        mode, reason = ROI.decide_mode(bead_roi, cell_roi, cfg_reg)

        if mode == ROI.UNREGISTERED:
            return None, bead_roi, mode, reason

        if mode == ROI.DIRECT:
            trans = self.trans_by_roi[match]
            reason += (f'; from the {len(self.rois[match])} bead '
                       f'acquisition(s) at that crop')
        else:
            size = max(self.rois, key=lambda k: len(self.rois[k]))
            bead_roi = self.rois[size][0]
            trans, corr = ROI.transport_translation(
                self.trans_by_roi[size], self.linear,
                bead_roi, self.rois_t.get(size, [None])[0],
                cell_roi, cell_roi_t, self.pixel_nm)
            if trans is None:
                return None, bead_roi, ROI.UNREGISTERED, (
                    reason + '; but a channel crop is unknown so the '
                             'transport could not be computed')
            reason += (f'; moved from the {size[0]}x{size[1]} bead crop by '
                       f'({corr[0]:+.1f}, {corr[1]:+.1f}) nm')
        return R.make_affine(self.linear, trans), bead_roi, mode, reason


def build_session_registration(session, cfg_reg, method='joint', verbose=True):
    """Fit the shared linear part and the per-crop translations for a session."""
    sets, singles, geoms, geoms_t, sizes = [], [], [], [], []
    for al, geom, geom_t in find_alignment_dirs(session):
        try:
            src = R.sum_projection(os.path.join(al, 'ROI-T.tif'), 30)
            dst = R.sum_projection(os.path.join(al, 'ROI-R.tif'), 30)
        except Exception:
            continue
        best = None
        for thr in (8.0, 5.0, 3.0, 2.0):
            try:
                a, b = R.bead_pairs(src, dst,
                                    replace(cfg_reg, bead_threshold_sigma=thr))
                T = R.solve_transform(a, b, cfg_reg.model)
            except Exception:
                continue
            if len(a) >= cfg_reg.min_beads and (best is None
                                                or T.residual_nm < best[2]):
                best = (a, b, T.residual_nm, T)
        if best is not None:
            sets.append((best[0], best[1]))
            singles.append(best[3])
            geoms.append(geom)
            geoms_t.append(geom_t)
            sizes.append(geom.size if geom.known else None)

    if not sets:
        if verbose:
            print('  ! no usable bead registration in this session; '
                  'assuming an identity linear part and relying entirely on '
                  'the per-acquisition translation')
        return SessionRegistration(np.eye(2), np.nan, 0, {}, {},
                                   pixel_nm=cfg_reg.cam_pixelsize_nm)

    if method == 'joint':
        J = R.fit_shared_linear(sets)
        lin, res, n = J['linear'], J['residual_nm'], J['n_sets']
        translations = [np.asarray(t, float) for t in J['translations']]
        if verbose:
            print(f'  bead registration: JOINT over {n} acquisitions, '
                  f'{J["n_pairs"]} pairs, residual {res:.1f} nm '
                  f'(p90 {J["residual_p90_nm"]:.1f})')
    else:
        k = int(np.argmin([t.residual_nm for t in singles]))
        lin, t = R.split_transform(singles[k])
        res, n = singles[k].residual_nm, 1
        translations = [np.zeros(2)] * len(sets)
        translations[k] = np.asarray(t, float)
        sizes = [sizes[k] if i == k else None for i in range(len(sizes))]
        geoms = [geoms[k] if i == k else geoms[i] for i in range(len(geoms))]
        if verbose:
            print(f'  bead registration: best single, residual {res:.1f} nm, '
                  f'{singles[k].n_pairs} pairs')

    by_roi, roi_map, roi_map_t = {}, {}, {}
    for size, t, g, gt in zip(sizes, translations, geoms, geoms_t):
        if size is None:
            continue
        by_roi.setdefault(size, []).append(t)
        roi_map.setdefault(size, []).append(g)
        roi_map_t.setdefault(size, []).append(gt)
    by_roi = {k: np.mean(np.array(v), axis=0) for k, v in by_roi.items()}

    px = cfg_reg.cam_pixelsize_nm
    optics, spread = [], None
    for t, g, gt in zip(translations, geoms, geoms_t):
        term = ROI.crop_term(g, gt, lin, px)
        if term is not None:
            optics.append(np.asarray(t, float) - term)
    if len(optics) > 1:
        o = np.array(optics)
        raw = np.array([np.asarray(t, float) for t in translations])
        spread = (float(np.ptp(o[:, 0])), float(np.ptp(o[:, 1])))
        raw_spread = (float(np.ptp(raw[:, 0])), float(np.ptp(raw[:, 1])))
        if verbose and len(roi_map) > 1:
            print(f'  centred-crop check: removing the crop term collapses '
                  f'the spread in the bead translations from '
                  f'({raw_spread[0]:.1f}, {raw_spread[1]:.1f}) to '
                  f'({spread[0]:.1f}, {spread[1]:.1f}) nm, against a bead-fit '
                  f'residual of {res:.1f} nm')

    if verbose and len(roi_map) > 1:
        crops = ', '.join(f'{w}x{h} ({len(roi_map[(w, h)])})'
                          for w, h in sorted(roi_map))
        print(f'  bead acquisitions span {len(roi_map)} image crops: {crops}. '
              f'The linear part is shared; each crop keeps its own '
              f'translation, and a cell either matches one or has it '
              f'transported onto its own crop.')

    return SessionRegistration(lin, res, n, by_roi, roi_map, roi_map_t,
                               pixel_nm=px, optics_spread_nm=spread)


def _emit_points(rows, meta, meth, pore_idx, source, x, y, cx, cy, phase,
                 frame=None, prec=None, trace_id=None):
    """One row per localization, with everything needed to re-derive anything."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0:
        return
    dx, dy = x - cx, y - cy
    c, s_ = np.cos(-phase), np.sin(-phase)
    rdx, rdy = c * dx - s_ * dy, s_ * dx + c * dy
    r = np.hypot(dx, dy)
    for k in range(len(x)):
        rows.append(dict(
            **meta, centre_method=meth, pore=pore_idx, source=source,
            x_nm=float(x[k]), y_nm=float(y[k]),
            dx_nm=float(dx[k]), dy_nm=float(dy[k]),
            rdx_nm=float(rdx[k]), rdy_nm=float(rdy[k]),
            r_nm=float(r[k]),
            frame=(float(frame[k]) if frame is not None else np.nan),
            locprec_nm=(float(prec[k]) if prec is not None else np.nan),
            trace_id=(float(trace_id) if trace_id is not None else np.nan),
            phase_rad=float(phase)))


def process_cell(cell, sreg, cfg, args):
    """Returns (rows, summary dict) or raises."""
    name = os.path.basename(cell)
    dish = os.path.basename(os.path.dirname(cell))

    dc = io_smap.find_sml(cell, prefer_driftc=True, channel='R')
    if dc is None:
        raise FileNotFoundError('no CamR sml file')

    N = int(args.merge_frames)
    ts = drift.read_timestamps(cell, args.timestamp_file)
    curve = drift.from_stored(dc, n_merge=N, timestamps=ts)
    if curve is not None:
        n_camr = curve.n_camr_frames
    else:
        fr = io_smap.read_sml(dc, fields=['frame']).get('frame')
        n_camr = int(np.nanmax(fr)) if fr is not None and len(fr) else 0
    t_path = drift.find_file(cell, args.roi_t_file)
    n_raw = drift.tif_frames(t_path)
    raw_from = 'ROI-T.tif' if n_raw else None
    if not n_raw and ts is not None:
        n_raw, raw_from = ts.n, 'Timestamp.txt'
    ok, implied_n, msg = drift.check_merge(n_camr, N, n_raw, args.merge_tol)
    if not ok:
        raise ValueError(msg)

    camr = smap_sites.run_camr(dc, args.smap_cfg, min_locs=args.min_locs_cell)
    Lr, geom, mle = camr['Lr'], camr['geom'], camr['mle']
    xr, yr = Lr['xnm'], Lr['ynm']
    mle_ids = [int(m['seed_index']) for m in mle]

    cell_roi = ROI.acquisition_roi(cell, channel='R', sml_path=dc)
    cell_roi_t = ROI.acquisition_roi(cell, channel='T')
    bead_T, bead_roi, mode, reason = sreg.for_cell(cell_roi, cell_roi_t,
                                                   cfg.register)
    if bead_T is None:
        raise ValueError(f'camera offset unavailable — {reason}')

    raw = camt.load_camt(cell)
    ct, align_info, overlap = {}, {}, {}
    for key, L in raw.items():
        if 'frame' not in L:
            continue
        x, y, info = R.align_camt_to_camr(
            L['xnm'], L['ynm'], bead_transform=bead_T,
            drift_curve=curve, frame=L['frame'], mode=mode)
        ct[key] = L.copy_with(xnm=x, ynm=y)
        align_info[key] = info
        if args.report_overlap:
            try:
                overlap[key] = R.channel_overlap_offset(x, y, xr, yr)
            except Exception:
                pass

    tcfg = traces.TraceConfig(min_length=args.min_trace_length,
                              assign_radius_nm=args.assign_radius,
                              min_occupancy=args.occupancy)
    lcfg = traces.LinkConfig(distance_threshold_nm=args.link_distance,
                             max_frame_gap=args.max_frame_gap)
    tracesets = {}
    if 'smap' in ct:
        tracesets['smap_traces'] = traces.build_traces(
            traces.link_traces(ct['smap'], lcfg), tcfg)
    if 'traces' in ct:
        tracesets['gdsc_traces'] = traces.build_traces(ct['traces'], tcfg)

    rows = {'pores': [], 'radial': [], 'traces': [], 'points': [],
            'phases': []}
    meta = dict(session=args._session, dish=dish, cell=name)
    for i, g in enumerate(geom):
        rows['pores'].append(dict(
            **meta, pore=i, method='geometric',
            x_nm=g['x0'], y_nm=g['y0'], R_nm=g['R0'], n_locs=g['n_locs'],
            inside=g['inside'], inring=g['inring'], outside=g['outside'],
            ratio_in=g['ratio_in'], ratio_out=g['ratio_out'],
            size_nm=g['size'], psf_nm=g['psf'], smap_site_ID=g['site_ID']))
    radius_free = args.smap_cfg.radius_free
    for m in mle:
        rows['pores'].append(dict(
            **meta, pore=int(m['seed_index']), method='locmofit',
            x_nm=m['x0'], y_nm=m['y0'], R_nm=m['R'], R_was_fitted=radius_free,
            n_locs=m['n'], sigma_nm=m['sigma'], background=m['background'],
            loglik=m['loglik'], AIC=m['AIC'], lr_vs_null=m['lr_vs_null'],
            shift_from_seed_nm=m['shift_from_seed']))

    kept_counts = {}
    for meth, centres, pore_ids in (('geometric', geom, list(range(len(geom)))),
                                    ('locmofit', mle, mle_ids)):
        if not centres:
            continue
        phases = {}
        for i, p in enumerate(centres):
            m = ((xr - p['x0']) ** 2 + (yr - p['y0']) ** 2) \
                <= args.merge_radius ** 2
            est = eightfold.estimate_phase(xr[m], yr[m], p['x0'], p['y0'],
                                           fold=args.fold,
                                           r_min=args.phase_rmin,
                                           r_max=args.phase_rmax)
            phases[i] = est
            _emit_points(rows['points'], meta, meth, pore_ids[i], 'ref_nup96',
                         xr[m], yr[m], p['x0'], p['y0'], est['phase'],
                         frame=Lr['frame'][m] if 'frame' in Lr else None,
                         prec=Lr['locprecnm'][m] if 'locprecnm' in Lr else None)

        for src, key in (('smap', 'smap_locs'), ('gdsc', 'gdsc_locs')):
            if src not in ct:
                continue
            L = ct[src]
            for i, p in enumerate(centres):
                m = ((L['xnm'] - p['x0']) ** 2 +
                     (L['ynm'] - p['y0']) ** 2) <= args.merge_radius ** 2
                if not m.any():
                    continue
                _emit_points(rows['points'], meta, meth, pore_ids[i], key,
                             L['xnm'][m], L['ynm'][m], p['x0'], p['y0'],
                             phases[i]['phase'],
                             frame=L['frame'][m] if 'frame' in L else None,
                             prec=L['locprecnm'][m] if 'locprecnm' in L else None)
                for v in np.hypot(L['xnm'][m] - p['x0'],
                                  L['ynm'][m] - p['y0']):
                    if v <= cfg.radial.r_keep_nm:
                        rows['radial'].append(dict(**meta, centre_method=meth,
                                                   source=key, pore=pore_ids[i],
                                                   r_nm=float(v)))

        for key, tset in tracesets.items():
            kept, allt = traces.assign_traces_to_pores(tset, centres, tcfg)
            kept_counts[(meth, key)] = (len(tset), len(kept))
            for t in kept:
                pi = t.get('pore_index', -1)
                if pi < 0 or pi >= len(centres):
                    continue
                p = centres[pi]
                _emit_points(rows['points'], meta, meth, pore_ids[int(pi)], key,
                             t['x'], t['y'], p['x0'], p['y0'],
                             phases[int(pi)]['phase'],
                             frame=t['frame'], prec=t.get('prec'),
                             trace_id=t['id'])
                for v in np.hypot(t['x'] - p['x0'], t['y'] - p['y0']):
                    if v <= cfg.radial.r_keep_nm:
                        rows['radial'].append(dict(**meta, centre_method=meth,
                                                   source=key,
                                                   pore=pore_ids[int(pi)],
                                                   r_nm=float(v)))
            if meth == 'geometric':
                for t in allt:
                    rows['traces'].append(dict(
                        **meta, trace_source=key, trace_id=t['id'],
                        length=t['length'], rg_nm=t['rg_nm'],
                        span_nm=t['span_nm'],
                        occupancy=t.get('occupancy', np.nan),
                        pore_index=t.get('pore_index', -1),
                        dist_to_pore_nm=t.get('dist_to_pore_nm', np.nan),
                        kept=bool(t.get('keep', False))))

        for i, p in enumerate(centres):
            rows['phases'].append(dict(**meta, centre_method=meth,
                                       pore=pore_ids[i],
                                       phase_rad=phases[i]['phase'],
                                       order=phases[i]['order'],
                                       null_order=phases[i]['null_order'],
                                       order_ratio=phases[i]['order_ratio'],
                                       n_ref_locs=phases[i]['n']))

    bead_t = next(iter(align_info.values()))['bead_translation_nm'] \
        if align_info else (np.nan, np.nan)
    drift_extrap = max((v.get('n_drift_extrapolated', 0)
                        for v in align_info.values()), default=0)
    ci = camr['info']

    summary = dict(
        **meta,
        n_locs_camR=len(Lr), n_locs_camR_ungrouped=ci['n_locs_camR_ungrouped'],
        n_candidates=ci['n_candidates'], n_sites_cleanup=ci['n_sites_cleanup'],
        n_pores_geometric=len(geom), n_pores_locmofit=len(mle),
        qc_pass_rate=ci['n_sites_cleanup'] / max(1, ci['n_candidates']),
        n_duplicate_pores=ci['n_duplicate_pores'], dedup_tol_nm=ci['dedup_tol_nm'],
        n_locmofit_rejected=ci['n_locmofit_rejected'],
        locmofit_reject_reasons=ci['locmofit_reject_reasons'],
        locmofit_R_was_fitted=ci['locmofit_R_was_fitted'],
        reg_mode=mode, reg_reason=reason,
        bead_translation_nm='{:.0f},{:.0f}'.format(*bead_t),
        drift_applied=bool(curve is not None),
        drift_by=(curve.unit if curve is not None else ''),
        camr_merge_frames=N, camr_frames=n_camr,
        raw_frames=n_raw if n_raw else np.nan, raw_frames_from=raw_from or '',
        implied_merge=implied_n, merge_check=msg,
        timestamp_file=bool(ts is not None),
        timestamp_rows=(ts.n if ts is not None else np.nan),
        camR_camT_sync_ms=(ts.sync_ms() if ts is not None else np.nan),
        drift_extrapolated_locs=drift_extrap,
        **cell_roi.as_dict('cell_roi_'),
        **bead_roi.as_dict('bead_roi_'),
        drift_span_x_nm=curve.span_nm[0] if curve else np.nan,
        drift_span_y_nm=curve.span_nm[1] if curve else np.nan,
        **{f'n_{k}': len(v) for k, v in tracesets.items()},
        **{f'on_pore_{m}_{k}': v[1] for (m, k), v in kept_counts.items()},
        median_R_geometric=float(np.median([g['R0'] for g in geom]))
        if geom else np.nan,
        median_R_locmofit=float(np.median([m['R'] for m in mle]))
        if mle else np.nan,
        **{f'overlap_{k}_nm': '{:.0f},{:.0f}'.format(*v)
           for k, v in overlap.items()},
        **{f'n_camT_{k}': len(v) for k, v in ct.items()})
    return rows, summary


def get_config(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=None,
                    help='override the data folder (default: ./data)')
    ap.add_argument('--out', default=None,
                    help='override the results folder (default: ./results)')
    ap.add_argument('--html', default=None,
                    help='override the HTML folder (default: ./interactive_html)')

    ap.add_argument('--camt-name', default=CAMT_MOLECULE,
                    help='what CamT is tracking (the target channel)')
    ap.add_argument('--camr-name', default=CAMR_MOLECULE,
                    help='what CamR is tracking (the reference scaffold)')
    ap.add_argument('--ref-radius', type=float, default=REF_RADIUS_NM,
                    help='literature ring radius of the CamR scaffold (nm)')

    ap.add_argument('--reg-method', default=REG_METHOD,
                    choices=('joint', 'best'),
                    help='how the bead acquisitions in a session combine')
    ap.add_argument('--report-overlap', action='store_true',
                    help='also report the cross-correlation offset between the '
                         'two channels as a diagnostic; never applied')
    ap.add_argument('--roi-anchor', default=ROI_ANCHOR,
                    choices=('centred', 'corner', 'unknown'),
                    help='how the camera anchors a crop; centred allows a bead '
                         'translation to be transported onto another crop size')
    ap.add_argument('--limit', type=int, default=LIMIT,
                    help='max cells per session; 0 = all')
    ap.add_argument('--merge-frames', type=int, default=ROI_R_MERGE_FRAMES,
                    help='raw ROI-R frames summed into one CamR frame before '
                         'the SMAP fit (1 = not merged)')
    ap.add_argument('--merge-tol', type=float, default=MERGE_CHECK_TOL)
    ap.add_argument('--timestamp-file', default=TIMESTAMP_FILE)
    ap.add_argument('--roi-t-file', default=ROI_T_FILE)
    ap.add_argument('--cutoff', type=float, default=None,
                    help='override SEGMENT_CUTOFF (segmentNPC)')
    ap.add_argument('--pixrec', type=float, default=None,
                    help='override CELL_PIXELSIZE_NM (segmentNPC render); '
                         'the cutoff is absolute on this render')
    ap.add_argument('--dedup', type=float, default=DEDUP_NM,
                    help='merge pore centres closer than this (nm); 0 = SMAP output')
    ap.add_argument('--min-trace-length', type=int, default=MIN_TRACE_LENGTH)
    ap.add_argument('--assign-radius', type=float, default=ASSIGN_RADIUS_NM)
    ap.add_argument('--occupancy', type=float, default=OCCUPANCY)
    ap.add_argument('--link-distance', type=float, default=LINK_DISTANCE_NM,
                    help='nm threshold for linking SMAP localizations into '
                         'traces')
    ap.add_argument('--max-frame-gap', type=int, default=MAX_FRAME_GAP,
                    help='1 = consecutive frames only; 2 tolerates one blink off')
    ap.add_argument('--min-locs-cell', type=int, default=MIN_LOCS_CELL)
    ap.add_argument('--merge-radius', type=float, default=MERGE_RADIUS_NM,
                    help='nm around each pore centre collected for merging')
    ap.add_argument('--fold', type=int, default=FOLD)
    ap.add_argument('--phase-rmin', type=float, default=PHASE_RMIN_NM)
    ap.add_argument('--phase-rmax', type=float, default=PHASE_RMAX_NM)
    ap.add_argument('--fix-radius', action='store_true',
                    help='fix the LocMoFit ring2D radius at --ref-radius '
                         '(default: LOCMOFIT_MODEL, radius free from 53.7 nm)')
    ap.add_argument('--max-html-points', type=int, default=15000,
                    help='points per explorer panel; larger files open slower')
    ap.add_argument('--no-html', action='store_true')
    args = ap.parse_args(argv)
    if args.merge_frames < 1:
        ap.error('--merge-frames must be an integer >= 1')
    args.smap_cfg = smap_site_config(args)

    args.data = args.data or resolve_dir(HERE, DATA_DIR)
    args.out = args.out or resolve_dir(HERE, RESULTS_DIR, create=True)
    args.html = args.html or resolve_dir(HERE, HTML_DIR, create=True)
    return args


def main(argv=None):
    args = get_config(argv)
    t0 = time.time()

    molecules = Molecules(camt=args.camt_name, camr=args.camr_name,
                          ref_radius_nm=args.ref_radius)
    cfg = P.DEFAULT.with_(molecules=molecules)
    reg_cfg = replace(RegisterConfig(mirror='none', model='affine',
                                     min_beads=6),
                      roi_anchor=args.roi_anchor)
    cfg = cfg.with_(register=reg_cfg)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.html, exist_ok=True)

    print('=' * 74)
    print(f'  CamT (target)    : {molecules.camt}')
    print(f'  CamR (reference) : {molecules.camr}  '
          f'(ring {molecules.ref_radius_nm:g} nm)')
    sc = args.smap_cfg
    print(f'  CamR merge       : {args.merge_frames}:1 '
          f'(drift moved to CamT by time; checked per acquisition)')
    print(f'  CamR pores       : SMAP port, cutoff {sc.segment_cutoff:g} on '
          f'{sc.cell_pixelsize_nm:g} nm render, LocMoFit radius '
          f'{"free" if sc.radius_free else "fixed"}, dedup {sc.dedup_nm:g} nm')
    print(f'  data             : {args.data}')
    print(f'  results          : {args.out}')
    print(f'  interactive html : {args.html}')
    print('=' * 74)

    if not os.path.isdir(args.data):
        print(f'\nNo data folder at {args.data}.')
        print('Create it and drop the acquisition sessions inside, e.g.')
        print(f'  {os.path.join(args.data, "2026-07-28 Session", "Alignment", "...")}')
        return 1

    sessions = find_sessions(args.data)
    if not sessions:
        print(f'\nNo sessions found under {args.data}.')
        print('A session is a folder containing acquisition folders with '
              '*_sml.mat files.')
        return 1
    print(f'\nsessions: {[os.path.basename(s) for s in sessions]}\n')

    all_rows = {'pores': [], 'radial': [], 'traces': [], 'points': [],
                'phases': []}
    summaries, failures, mode_log = [], [], []

    for sess in sessions:
        sname = os.path.basename(sess)
        args._session = sname
        print(f'=== {sname} ===')

        sreg = build_session_registration(sess, reg_cfg,
                                          method=args.reg_method)

        cells = find_cells(sess)
        if args.limit:
            cells = cells[:args.limit]
        print(f'  {len(cells)} acquisitions, {sreg.n_sets} usable bead '
              f'acquisitions')

        if cells:
            probe_sml = io_smap.find_sml(cells[0], channel='R')
            probe_roi = ROI.acquisition_roi(cells[0], 'R', probe_sml)
            probe_roi_t = ROI.acquisition_roi(cells[0], 'T')
            _, bead_roi, mode, reason = sreg.for_cell(probe_roi, probe_roi_t,
                                                      reg_cfg)
            print(ROI.format_report(bead_roi, probe_roi, mode, reason,
                                    label=sname))
            mode_log.append((sname, mode, reason,
                             bead_roi.describe(), probe_roi.describe()))

        for cell in cells:
            label = (f'{os.path.basename(os.path.dirname(cell))}/'
                     f'{os.path.basename(cell)}')
            try:
                rows, summ = process_cell(cell, sreg, cfg, args)
                summ['bead_residual_nm'] = sreg.residual_nm
                summ['n_bead_acquisitions'] = sreg.n_sets
                for k in all_rows:
                    all_rows[k] += rows[k]
                summaries.append(summ)
                flag = ('' if summ['reg_mode'] == ROI.DIRECT
                        else '  [offset transported]')
                print(f'  {label:38s} pores {summ["n_pores_geometric"]:4d}  '
                      f'traces {summ.get("n_smap_traces", 0):4d}smap/'
                      f'{summ.get("n_gdsc_traces", 0):<4d}gdsc  '
                      f'({summ.get("on_pore_geometric_smap_traces", 0)}+'
                      f'{summ.get("on_pore_geometric_gdsc_traces", 0)} on pore)'
                      f'{flag}')
            except Exception as e:
                failures.append(dict(cell=label,
                                     error=f'{type(e).__name__}: {e}'))
                print(f'  {label:38s} SKIPPED — {type(e).__name__}: {e}')
        print()

    if not summaries:
        print('No acquisition was processed successfully. See failures.csv.')

    import pandas as pd
    written = []
    for k, v in all_rows.items():
        p = os.path.join(args.out, f'{k}.csv')
        pd.DataFrame(v).to_csv(p, index=False)
        written.append((p, len(v)))
    for nm, data in (('summary', summaries), ('failures', failures)):
        p = os.path.join(args.out, f'{nm}.csv')
        pd.DataFrame(data).to_csv(p, index=False)
        written.append((p, len(data)))

    p = os.path.join(args.out, 'registration_modes.csv')
    pd.DataFrame(mode_log, columns=['session', 'mode', 'reason',
                                    'bead_roi', 'cell_roi']).to_csv(p,
                                                                    index=False)
    written.append((p, len(mode_log)))

    p = os.path.join(args.out, 'molecules.csv')
    pd.DataFrame([dict(camt=molecules.camt, camr=molecules.camr,
                       ref_radius_nm=molecules.ref_radius_nm)]).to_csv(
        p, index=False)
    written.append((p, 1))

    if all_rows['radial']:
        df = pd.DataFrame(all_rows['radial'])
        prof = []
        for (meth, src), g in df.groupby(['centre_method', 'source']):
            r = g['r_nm'].to_numpy()
            ctr, dens, cnt = radial.radial_density(r, cfg.radial)
            fit = radial.fit_peak(r, cfg.radial) or {}
            for c_, d_, n_ in zip(ctr, dens, cnt):
                prof.append(dict(centre_method=meth, source=src, r_nm=c_,
                                 density=d_, count=int(n_), n_total=len(r),
                                 fit_peak_nm=fit.get('peak_nm', np.nan),
                                 fit_sigma_nm=fit.get('sigma_nm', np.nan),
                                 fit_R2=fit.get('R2', np.nan)))
        p = os.path.join(args.out, 'radial_profiles.csv')
        pd.DataFrame(prof).to_csv(p, index=False)
        written.append((p, len(prof)))

    html_path = None
    if not args.no_html and all_rows['points']:
        html_path = build_explorer(args, cfg, molecules, mode_log)

    print('=== written ===')
    for p, n in written:
        print(f'  {n:7d} rows  {p}')
    if html_path:
        print(f'          html  {html_path}')

    if summaries:
        s = pd.DataFrame(summaries)
        print(f'\ncells processed: {len(s)}   failed: {len(failures)}')
        print(f'total pores (geometric): {int(s.n_pores_geometric.sum())}   '
              f'(locmofit): {int(s.n_pores_locmofit.sum())}')
        print(f'QC pass rate: {100 * s.qc_pass_rate.median():.1f}% median')
        print('\ncamera offset (from beads):')
        for mode, n in s.reg_mode.value_counts().items():
            print(f'  {mode:18s} {n} acquisitions')
        print(f'  bead-fit residual: {s.bead_residual_nm.median():.1f} nm '
              f'median — this is the precision of the offset correction')
        print(f'drift (from {molecules.camr} per acquisition): '
              f'{int(s.drift_applied.sum())}/{len(s)} acquisitions corrected, '
              f'CamR merge {args.merge_frames}:1, by '
              f'{", ".join(sorted(set(s.drift_by.astype(str))))}')
        nx = int(s.drift_extrapolated_locs.sum())
        if nx:
            print(f'  ! {nx} CamT localizations fell outside the CamR drift '
                  f'curve and were corrected with its end value')
        print(f'duplicate pores removed: {int(s.n_duplicate_pores.sum())}   '
              f'LocMoFit failures: {int(s.n_locmofit_rejected.sum())}')
        for col in [c for c in s.columns if c.startswith('n_smap_traces')
                    or c.startswith('n_gdsc_traces')]:
            print(f'{col}: {int(s[col].sum())}')
        for col in [c for c in s.columns if c.startswith('on_pore_')]:
            print(f'{col}: {int(s[col].sum())}')

    print(f'\ndone in {time.time() - t0:.0f} s')
    print('next: python 02_figures.py')
    return 0


def build_explorer(args, cfg, molecules, mode_log):
    """Join the per-pore quality columns onto points.csv and write the HTML."""
    import pandas as pd

    df = pd.read_csv(os.path.join(args.out, 'points.csv'))
    key = ['session', 'dish', 'cell', 'centre_method', 'pore']

    php = os.path.join(args.out, 'phases.csv')
    if os.path.exists(php):
        ph = pd.read_csv(php)
        if len(ph):
            df = df.merge(ph[key + ['n_ref_locs', 'order_ratio']], on=key,
                          how='left')
    prp = os.path.join(args.out, 'pores.csv')
    if os.path.exists(prp):
        pr = pd.read_csv(prp)
        if len(pr):
            pr = pr.rename(columns={'method': 'centre_method'})
            df = df.merge(pr[key + ['R_nm']].drop_duplicates(key, keep='last'),
                          on=key, how='left')
    for c in ('n_ref_locs', 'order_ratio', 'R_nm', 'locprec_nm'):
        if c not in df.columns:
            df[c] = np.nan

    note = None
    if os.path.exists(php):
        per_pore = pd.read_csv(php)
        if len(per_pore):
            ratio = per_pore['order_ratio'].median()
            print('\n--- 8-fold phase quality ---')
            print(f'  pores: {len(per_pore):,}')
            print(f'  median order / null-floor ratio: {ratio:.2f}')
            if ratio < 1.2:
                print('  WARNING: at or below the uniform floor. The per-pore '
                      'phase is not measurable, so rotating by it RANDOMIZES '
                      'rather than aligns. Rotated panels will destroy real '
                      'structure, not create fake structure — a false-negative '
                      'risk.')

    modes = sorted({m for _, m, _, _, _ in mode_log})
    if modes:
        detail = '; '.join(f'{s}: {m}' for s, m, _, _, _ in mode_log)
        note = (f'<b>Two-camera registration:</b> {" and ".join(modes)}. '
                f'{detail}. Full reasoning in '
                f'<code>results/registration_modes.csv</code>.')

    out = os.path.join(args.html, 'merged_explorer.html')
    explorer.build_html(df, out, molecules,
                        max_pts=args.max_html_points,
                        registration_note=note)
    return out


if __name__ == '__main__':
    sys.exit(main())
