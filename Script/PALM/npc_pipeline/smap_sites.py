# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""CamR pore centres with the SMAP port: segmentNPC -> NPCsegmentCleanup -> LocMoFit."""
import os
from dataclasses import dataclass, field

import numpy as np

from .io_smap import Localizations


class SiteResult(dict):
    """Geometric site: centre, radius and NPCsegmentCleanup counts."""


def _default_locmofit():
    return {'name': 'ring2D',
            'models': [{'class': 'ring2D', 'layer': 1}],
            'pars': [{'id': 'm1.mPar.radius', 'fix': False, 'value': 53.7,
                      'lb': -20, 'ub': 20, 'min': 20, 'max': 100},
                     {'id': 'm1.lPar.zrot', 'fix': True, 'value': 0},
                     {'id': 'm1.lPar.weight', 'fix': True, 'value': 1}]}


@dataclass
class SmapSiteConfig:
    """SMAP settings for the CamR stage."""
    group_dx_nm: float = 35.0
    group_dt_frames: int = 1             # in CamR (merged) frames
    layer_grouped: bool = True
    filter_locprec_nm: tuple = (0, 25)
    filter_psf_nm: tuple = (0, 150)
    filter_frame: tuple = (0, np.inf)
    filter_llrel: tuple = None           # e.g. (-1.5, 0)
    colormode: str = 'field'
    colorfield: str = 'PSFxnm'
    lut: str = 'jet'
    cell_pixelsize_nm: float = 15.0
    cell_fov_nm: float = 5000.0
    site_pixelsize_nm: float = 3.0
    site_fov_nm: float = 400.0
    site_roi_nm: float = 220.0
    segment_cutoff: float = 0.06
    segment_diameter_nm: float = 100.0
    segment_rim_nm: float = 20.0
    cleanup_R_nm: float = 55.0
    cleanup_dR_nm: float = 15.0
    cleanup_center: bool = True
    cleanup_inside: float = 0.3
    cleanup_outside: float = 0.75
    cleanup_radius_range_nm: tuple = (45.0, 65.0)
    cleanup_minsize: float = 25.0
    cleanup_minlocs: int = 10
    cleanup_maxpsf_nm: float = 150.0
    dedup_nm: float = 30.0
    locmofit_model: dict = field(default_factory=_default_locmofit)

    @property
    def radius_free(self):
        return not any(p.get('id') == 'm1.mPar.radius' and p.get('fix')
                       for p in self.locmofit_model.get('pars', []))


def pipeline_config(c: SmapSiteConfig):
    """npc_pipeline.smap PipelineConfig for the post-fit steps."""
    from npc_pipeline.smap.pipeline import PipelineConfig
    from npc_pipeline.smap.roimanager import SESettings, CleanupParameters
    from npc_pipeline.smap.locmofit import LocMoFit
    p = PipelineConfig()
    p.group_dx, p.group_dt = float(c.group_dx_nm), int(c.group_dt_frames)
    p.layer_grouped = bool(c.layer_grouped)
    p.layer_filters = {'locprecnm': tuple(c.filter_locprec_nm),
                       'PSFxnm': tuple(c.filter_psf_nm),
                       'frame': tuple(c.filter_frame)}
    if c.filter_llrel is not None:
        p.layer_filters['LLrel'] = tuple(c.filter_llrel)
    p.render_kw = ({'colorfield': c.colorfield, 'lut': c.lut}
                   if c.colormode == 'field' else {})
    p.se = SESettings(site_pixelsize=c.site_pixelsize_nm, site_fov=c.site_fov_nm,
                      site_roi=c.site_roi_nm, cell_pixelsize=c.cell_pixelsize_nm,
                      cell_fov=c.cell_fov_nm)
    p.segment_cutoff = c.segment_cutoff
    p.segment_diameter = c.segment_diameter_nm
    p.segment_rim = c.segment_rim_nm
    p.cleanup = CleanupParameters(
        R=c.cleanup_R_nm, dR=c.cleanup_dR_nm, center=c.cleanup_center,
        inside=c.cleanup_inside, outside=c.cleanup_outside,
        radiusrange=tuple(c.cleanup_radius_range_nm), minsize=c.cleanup_minsize,
        minlocs=c.cleanup_minlocs, maxPSF=c.cleanup_maxpsf_nm)
    p.keep_only_good_sites = True
    model = c.locmofit_model
    p.locmofit_factory = lambda roi, _m=model: LocMoFit.from_dict(_m, roi)
    return p


def dedupe_sites(sites, tol_nm=30.0, R_nm=55.0):
    """Merge accepted sites that re-centred onto the same pore."""
    if not sites:
        return [], []
    order = sorted(range(len(sites)),
                   key=lambda i: (-int(sites[i].get('n_locs', 0)),
                                  abs(float(sites[i].get('R0', R_nm)) - R_nm)))
    kept, kept_xy, dropped = [], [], []
    t2 = float(tol_nm) ** 2
    for i in order:
        s = sites[i]
        x0, y0 = float(s['x0']), float(s['y0'])
        if any((x0 - kx) ** 2 + (y0 - ky) ** 2 < t2 for kx, ky in kept_xy):
            dropped.append(s)
            continue
        kept.append(s)
        kept_xy.append((x0, y0))
    keep_ids = {id(s) for s in kept}
    return [s for s in sites if id(s) in keep_ids], dropped


def geometric_sites(sites, dedup_nm, R_nm):
    """SMAP sites accepted by NPCsegmentCleanup -> SiteResult dicts (+ dedup)."""
    geom = []
    for s in sites:
        ev = s.evaluation.get('NPCsegmentCleanup', {})
        inring = float(ev.get('inring', 0))
        g = SiteResult(
            x0=float(s.pos[0]), y0=float(s.pos[1]), R0=float(ev.get('R0', np.nan)),
            n_locs=int(ev.get('locs', 0)), inside=int(ev.get('inside', 0)),
            inring=int(inring), outside=int(ev.get('outside', 0)),
            ratio_in=float(ev.get('inside', 0)) / inring if inring else np.nan,
            ratio_out=float(ev.get('outside', 0)) / inring if inring else np.nan,
            size=float(ev.get('sizeav', np.nan)), psf=float(ev.get('meanPSF', np.nan)),
            keep=True, site_ID=int(s.ID))
        g['_site'] = s
        geom.append(g)
    n0 = len(geom)
    if dedup_nm and dedup_nm > 0:
        geom, _ = dedupe_sites(geom, dedup_nm, R_nm=R_nm)
    return geom, n0 - len(geom)


def locmofit_sites(geom):
    """LocMoFit result of each geometric site; seed_index = geometric pore index."""
    out = []
    for i, g in enumerate(geom):
        lm = g['_site'].evaluation.get('LocMoFit', {}) or {}
        if 'error' in lm or 'm1.lPar.x' not in lm:
            out.append(dict(seed_index=i, keep=False,
                            reason=lm.get('error', 'no LocMoFit result')))
            continue
        dx, dy = float(lm['m1.lPar.x']), float(lm['m1.lPar.y'])
        n = lm.get('numOfLocsPerLayer', [np.nan])
        out.append(dict(seed_index=i, keep=True, x0=g['x0'] + dx, y0=g['y0'] + dy,
                        R=float(lm.get('m1.mPar.radius', np.nan)),
                        n=float(np.ravel(n)[0]), sigma=np.nan,
                        background=float(lm.get('m91.offset.weight', np.nan)),
                        loglik=float(lm.get('LLfit', np.nan)),
                        AIC=float(lm.get('AIC', np.nan)),
                        lr_vs_null=np.nan, shift_from_seed=float(np.hypot(dx, dy))))
    return out


def run_camr(sml_path, c: SmapSiteConfig, min_locs=0, verbose=False):
    """SMAP segmentation -> cleanup -> LocMoFit on an existing CamR sml."""
    from npc_pipeline.smap.io import read_sml
    from npc_pipeline.smap.grouping import LocData
    from npc_pipeline.smap.pipeline import NPCPipeline

    p = pipeline_config(c)
    pipe = NPCPipeline(p)
    r = read_sml(sml_path)
    locs = r['loc']
    if 'xnm' not in locs or len(locs['xnm']) == 0:
        raise ValueError(f'{os.path.basename(sml_path)}: no localizations')
    ld = LocData(locs, p.group_dx, p.group_dt)
    layer = pipe.layer
    sel = ld.get(layer)
    keep = [k for k in ('xnm', 'ynm', 'frame', 'locprecnm', 'PSFxnm', 'phot',
                        'LLrel') if k in sel]
    Lr = Localizations({k: np.asarray(sel[k], float) for k in keep}, sml_path)
    if len(Lr) < min_locs:
        raise ValueError(f'only {len(Lr)} CamR locs after grouping and the '
                         f'layer filters (MIN_LOCS_CELL = {min_locs})')

    all_sites = pipe.segment(ld)
    pipe.cleanup(ld, all_sites)
    good = [s for s in all_sites if s.annotation.get('use')]
    pipe.fit_sites(ld, good, verbose=verbose)

    geom, n_dup = geometric_sites(good, c.dedup_nm, c.cleanup_R_nm)
    mle_all = locmofit_sites(geom)
    mle = [m for m in mle_all if m.get('keep')]
    rejects = {}
    for m in mle_all:
        if not m.get('keep'):
            k = str(m.get('reason', 'unknown'))[:40]
            rejects[k] = rejects.get(k, 0) + 1
    info = dict(n_locs_camR_ungrouped=int(len(locs['xnm'])),
                n_candidates=len(all_sites), n_sites_cleanup=len(good),
                n_duplicate_pores=n_dup, dedup_tol_nm=c.dedup_nm,
                n_locmofit_rejected=len(mle_all) - len(mle),
                locmofit_reject_reasons='; '.join(
                    f'{k}:{v}' for k, v in sorted(rejects.items())),
                locmofit_R_was_fitted=c.radius_free)
    return dict(Lr=Lr, geom=geom, mle=mle, info=info, locdata=ld, layer=layer,
                all_sites=all_sites)
