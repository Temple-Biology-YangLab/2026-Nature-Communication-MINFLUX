# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""SMAP NPC workflow after localization: grouping, layer filters, makeCellGrid +
segmentNPC, NPCsegmentCleanup, and LocMoFit on every accepted site."""
from dataclasses import dataclass, field
import time

import numpy as np

from .grouping import LocData, Layer
from .roimanager import (SESettings, CleanupParameters, make_cell_grid, render_cells,
                         segment_npc, npc_segment_cleanup, get_site_locs)
from .locmofit import LocMoFit, Ring2D


def default_locmofit(roi_size):
    """Ring2D: centre (x, y) and radius free (start 53.7 nm), background weight free."""
    f = LocMoFit(roi_size=roi_size)
    f.add_model(Ring2D(), layer=1)
    f.set_par('m1.mPar.radius', fix=False, value=53.7, lb=-20, ub=20, min=20, max=100)
    f.set_par('m1.lPar.zrot', fix=True, value=0)
    f.set_par('m1.lPar.weight', fix=True, value=1)
    return f


@dataclass
class PipelineConfig:
    group_dx: float = 35.0
    group_dt: int = 1
    layer_filters: dict = field(default_factory=lambda: {'locprecnm': (0, 25), 'PSFxnm': (0, 150),
                                                         'frame': (0, np.inf)})
    layer_grouped: bool = True
    se: SESettings = field(default_factory=SESettings)
    segment_cutoff: float = 0.06
    segment_diameter: float = 100.0
    segment_rim: float = 20.0
    render_kw: dict = field(default_factory=lambda: {'colorfield': 'PSFxnm'})
    cleanup: CleanupParameters = field(default_factory=CleanupParameters)
    keep_only_good_sites: bool = True
    locmofit_factory: object = None
    locmofit_fields: tuple = ('locprecnm', 'PSFxnm', 'xnm', 'ynm', 'frame')


class NPCPipeline:
    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()

    @property
    def layer(self):
        return Layer(self.cfg.layer_filters, self.cfg.layer_grouped)

    def locdata(self, locs):
        """Group localizations (dict of arrays, SMAP field names)."""
        return LocData(locs, self.cfg.group_dx, self.cfg.group_dt)

    def segment(self, locdata):
        c = self.cfg
        cells = make_cell_grid(locdata.loc, c.se)
        render_cells(cells, locdata, [self.layer], c.se, c.render_kw)
        return segment_npc(cells, c.segment_cutoff, c.segment_diameter, c.segment_rim)

    def cleanup(self, locdata, sites):
        return npc_segment_cleanup(locdata, sites, self.layer, self.cfg.se, self.cfg.cleanup)

    def fit_sites(self, locdata, sites, verbose=False):
        c = self.cfg
        factory = c.locmofit_factory or default_locmofit
        template = factory(c.se.site_roi)
        t0 = time.time()
        for i, s in enumerate(sites):
            locs = get_site_locs(locdata, s, self.layer, c.se.site_roi / 2, list(c.locmofit_fields))
            if len(locs['xnm']) < 2:
                s.evaluation['LocMoFit'] = {'error': 'too few localizations'}
                continue
            fl = {'xnm': locs['xnmrot'], 'ynm': locs['ynmrot'], 'locprecnm': locs['locprecnm'],
                  'layer': np.ones(len(locs['xnmrot']), int)}
            f = template.copy()
            try:
                s.evaluation['LocMoFit'] = f.fit(fl)
            except Exception as e:
                s.evaluation['LocMoFit'] = {'error': str(e)}
            if verbose and (i % 20 == 0 or i == len(sites) - 1):
                print('\rLocMoFit %d/%d sites, %.1f s' % (i + 1, len(sites), time.time() - t0),
                      end='', flush=True)
        if verbose:
            print()
        return sites

    def run(self, locs, verbose=False):
        """Returns (locdata, all sites, accepted sites with LocMoFit results)."""
        ld = self.locdata(locs)
        sites = self.segment(ld)
        self.cleanup(ld, sites)
        good = [s for s in sites if s.annotation.get('use')] if self.cfg.keep_only_good_sites else sites
        self.fit_sites(ld, good, verbose)
        return ld, sites, good
