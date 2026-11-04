# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configuration dataclasses."""
from dataclasses import dataclass, field, replace
from typing import Tuple


@dataclass
class Molecules:
    """Names of the proteins on each camera."""
    camt: str = 'Nup62'          # target channel
    camr: str = 'Nup96'          # reference / scaffold channel
    ref_radius_nm: float = 53.7  # literature ring radius of the CamR scaffold

    @property
    def camt_slug(self):
        return _slugify(self.camt)

    @property
    def camr_slug(self):
        return _slugify(self.camr)


def _slugify(name):
    """Filename-safe version of a molecule name."""
    import re
    s = re.sub(r'[^A-Za-z0-9]+', '_', str(name)).strip('_')
    return s or 'molecule'


@dataclass
class RegisterConfig:
    """Image-based two-camera registration."""
    method: str = 'beads'            # 'beads' | 'crosscorr'
    mirror: str = 'none'             # 'none' | 'updown' | 'leftright'
    model: str = 'affine'            # 'translation' | 'rigid' | 'affine' | 'poly2'
    cam_pixelsize_nm: float = 97.0
    bead_min_sep_px: float = 5.0
    bead_threshold_sigma: float = 5.0
    bead_fit_roi_px: int = 7
    bead_max_pair_nm: float = 2000.0
    cc_pixrec_nm: Tuple[float, float] = (100.0, 20.0)
    cc_max_shift_nm: Tuple[float, float] = (3000.0, 500.0)
    max_residual_nm: float = 30.0    # warn above this
    min_beads: int = 6
    roi_anchor: str = 'centred'      # 'centred' | 'corner' | 'unknown'
    roi_tolerance_px: int = 0


@dataclass
class RadialConfig:
    """Radial averaging of a target channel about NPC centres."""
    r_keep_nm: float = 120.0
    n_bins: int = 24
    use_density: bool = True         # area-corrected
    fit_center_bounds_nm: Tuple[float, float] = (0.0, 150.0)
    fit_sigma_bounds_nm: Tuple[float, float] = (2.0, 100.0)


@dataclass
class Config:
    molecules: Molecules     = field(default_factory=Molecules)
    register: RegisterConfig = field(default_factory=RegisterConfig)
    radial:   RadialConfig   = field(default_factory=RadialConfig)

    def with_(self, **kw):
        """Modified copy, e.g. cfg.with_(radial=replace(cfg.radial, n_bins=30))."""
        return replace(self, **kw)


DEFAULT = Config()
