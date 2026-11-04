# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image-ROI geometry of acquisitions and transport of bead translations between crops.
"""
import glob
import json
import os
import re

import numpy as np


class RoiGeometry:
    """One acquisition's camera crop."""

    def __init__(self, width=None, height=None, x0=None, y0=None,
                 sensor=None, source='unknown', origin_known=False,
                 detail=None):
        self.width = width              # IMAGE roi — the coordinate frame
        self.height = height
        self.x0 = x0
        self.y0 = y0
        self.sensor = sensor            # (w, h) of the camera readout, context
        self.source = source            # 'sml' | 'gdsc' | 'tif' | 'abl'
        self.origin_known = origin_known
        self.detail = detail or {}

    @property
    def known(self):
        return self.width is not None and self.height is not None

    @property
    def size(self):
        return (self.width, self.height)

    def describe(self):
        if not self.known:
            return 'unknown ROI'
        txt = f'image {self.width}x{self.height} px'
        if self.origin_known:
            txt += f' at ({self.x0}, {self.y0})'
        txt += f' [{self.source}]'
        if self.sensor and tuple(self.sensor) != (self.width, self.height):
            txt += f', camera readout {self.sensor[0]}x{self.sensor[1]}'
        return txt

    def as_dict(self, prefix=''):
        return {f'{prefix}width_px': self.width,
                f'{prefix}height_px': self.height,
                f'{prefix}x0_px': self.x0,
                f'{prefix}y0_px': self.y0,
                f'{prefix}sensor_w_px': self.sensor[0] if self.sensor else None,
                f'{prefix}sensor_h_px': self.sensor[1] if self.sensor else None,
                f'{prefix}source': self.source}

    def __repr__(self):
        return f'<RoiGeometry {self.describe()}>'


_ROI_RE = re.compile(r'xy\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*'
                     r'wh\(\s*(\d+)\s*,\s*(\d+)\s*\)')


def _parse_roi_string(text):
    """'xy(0, 0), wh(2304, 2304)' -> (0, 0, 2304, 2304), or None."""
    if not isinstance(text, str):
        return None
    m = _ROI_RE.search(text)
    if not m:
        return None
    x0, y0, w, h = (int(v) for v in m.groups())
    return x0, y0, w, h


def find_abl(acq_dir, max_up=4):
    """Nearest *.abl at or above an acquisition folder."""
    d = os.path.abspath(acq_dir)
    for _ in range(max_up):
        hits = sorted(glob.glob(os.path.join(d, '*.abl')))
        if hits:
            return hits[0]
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def read_abl_roi(acq_dir, channel='R'):
    """CAMERA ROI for one acquisition, from the nearest .abl."""
    abl = find_abl(acq_dir)
    if abl is None:
        return None
    try:
        with open(abl, encoding='utf-8', errors='replace') as f:
            doc = json.load(f)
    except Exception:
        return None

    want = os.path.basename(os.path.abspath(acq_dir))
    wanted_names = (f'ROI-{channel}', f'Cam{channel}')

    best = None
    for seq in doc.get('Sequences', []) or []:
        for acq in seq.get('Acquisition', []) or []:
            if acq.get('Name') != want and best is not None:
                continue
            cams = acq.get('Cameras', []) or []
            sensor = None
            if cams:
                sw = cams[0].get('Sensor width (px)')
                sh = cams[0].get('Sensor height (px)')
                if sw and sh:
                    sensor = (int(sw), int(sh))
            channels = acq.get('Channels', []) or []
            picked = None
            for name in wanted_names:
                for ch in channels:
                    if ch.get('Name') == name:
                        picked = ch
                        break
                if picked is not None:
                    break
            if picked is None and channels:
                picked = channels[0]
            if picked is None:
                continue
            parsed = _parse_roi_string(picked.get('ROI (px)'))
            if parsed is None:
                continue
            x0, y0, w, h = parsed
            g = RoiGeometry(width=w, height=h, x0=x0, y0=y0, sensor=sensor,
                            source='abl',
                            origin_known=False,
                            detail=dict(abl=abl, acquisition=acq.get('Name'),
                                        channel=picked.get('Name')))
            if acq.get('Name') == want:
                return g
            if best is None:
                best = g
    return best


def read_tif_roi(acq_dir, channel='R'):
    """IMAGE ROI size from the tif actually written for this acquisition."""
    path = os.path.join(acq_dir, f'ROI-{channel}.tif')
    if not os.path.exists(path):
        return None
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            h, w = t.pages[0].shape[:2]
    except Exception:
        try:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            return None
    return RoiGeometry(width=int(w), height=int(h), source='tif',
                       origin_known=False, detail=dict(path=path))


_BOUNDS_RE = re.compile(r'#Bounds\s+x(-?\d+)\s+y(-?\d+)\s+w(\d+)\s+h(\d+)')
_IJ_ORIGIN_RE = re.compile(
    r'<xOrigin>(-?\d+)</xOrigin>\s*<yOrigin>(-?\d+)</yOrigin>'
    r'\s*<width>(\d+)</width>\s*<height>(\d+)</height>')

GDSC_FILES = ('Single_Point_CamT.csv', 'Trace_CamT.xls')


def read_gdsc_roi(acq_dir):
    """IMAGE ROI from a GDSC results header."""
    for fn in GDSC_FILES:
        path = os.path.join(acq_dir, fn)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                head = f.read(8192)
        except OSError:
            continue
        m = _IJ_ORIGIN_RE.search(head) or _BOUNDS_RE.search(head)
        if not m:
            continue
        if m.re is _IJ_ORIGIN_RE:
            x0, y0, w, h = (int(v) for v in m.groups())
        else:
            x0, y0, w, h = (int(v) for v in m.groups())
        return RoiGeometry(width=w, height=h, x0=x0, y0=y0, source='gdsc',
                           origin_known=True, detail=dict(path=path))
    return None


def read_sml_roi(sml_path):
    """ROI from a SMAP _sml.mat file info block, when present."""
    if sml_path is None or not os.path.exists(sml_path):
        return None
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(sml_path, 'r') as h:
            g = h.get('saveloc')
            if g is None:
                return None
            f = g.get('file')
            if f is None or 'roi' not in f:
                return None
            v = np.array(f['roi']).ravel().astype(float)
    except Exception:
        return None
    if v.size < 4:
        return None
    x0, y0, w, h = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return RoiGeometry(width=int(round(w)), height=int(round(h)),
                       x0=int(round(x0)), y0=int(round(y0)),
                       source='sml', origin_known=True,
                       detail=dict(path=sml_path))


def acquisition_roi(acq_dir, channel='R', sml_path=None):
    """Best available IMAGE ROI for one acquisition folder."""
    cands = {}
    for name, g in (('sml', read_sml_roi(sml_path)),
                    ('gdsc', read_gdsc_roi(acq_dir)),
                    ('tif', read_tif_roi(acq_dir, channel))):
        if g is not None and g.known:
            cands[name] = g
    camera = read_abl_roi(acq_dir, channel)

    if not cands:
        if camera is not None and camera.known:
            camera.detail = dict(camera.detail, fallback='no image ROI found')
            return camera
        return RoiGeometry(source='unknown')

    for name in ('sml', 'gdsc', 'tif'):
        if name in cands:
            best = cands[name]
            break
    best.detail = dict(best.detail)
    best.detail['alternatives'] = {k: v.size for k, v in cands.items()}
    if camera is not None and camera.known:
        best.sensor = (camera.width, camera.height)
        best.detail['camera_roi'] = camera.size
        best.detail['camera_sensor'] = camera.sensor
    return best


def alignment_rois(session_dir, channel='R'):
    """[(acquisition dir, RoiGeometry)] for every bead run in a session."""
    out = []
    for pattern in (os.path.join(session_dir, 'Alignment', '*', '*'),
                    os.path.join(session_dir, 'Alignment', '*')):
        dirs = [d for d in sorted(glob.glob(pattern)) if os.path.isdir(d)
                and os.path.exists(os.path.join(d, f'ROI-{channel}.tif'))]
        if dirs:
            out = [(d, acquisition_roi(d, channel)) for d in dirs]
            break
    return out


def session_bead_roi(session_dir, channel='R'):
    """Representative bead ROI for a session."""
    pairs = alignment_rois(session_dir, channel)
    geoms = [g for _, g in pairs if g.known]
    if not geoms:
        return RoiGeometry(source='unknown'), 0, True
    sizes = {g.size for g in geoms}
    return geoms[0], len(geoms), len(sizes) == 1


def group_by_size(pairs):
    """{(w, h): [acquisition dir, ...]} for a list from alignment_rois."""
    out = {}
    for d, g in pairs:
        if g.known:
            out.setdefault(g.size, []).append(d)
    return out


def half_extent(geom, pixel_nm):
    """v = (W/2, H/2) * pixel_nm, in nm. None if the crop is unknown."""
    if geom is None or not geom.known:
        return None
    return np.array([geom.width / 2.0 * pixel_nm,
                     geom.height / 2.0 * pixel_nm], float)


def crop_term(roi_ref, roi_tgt, linear, pixel_nm):
    """v_R - L @ v_T for one run. None if either crop is unknown."""
    vR = half_extent(roi_ref, pixel_nm)
    vT = half_extent(roi_tgt, pixel_nm)
    if vR is None or vT is None:
        return None
    return vR - np.asarray(linear, float) @ vT


def transport_translation(t_bead, linear, bead_ref, bead_tgt,
                          cell_ref, cell_tgt, pixel_nm):
    """Move a bead-run translation onto a different crop."""
    cb = crop_term(bead_ref, bead_tgt, linear, pixel_nm)
    cc = crop_term(cell_ref, cell_tgt, linear, pixel_nm)
    if cb is None or cc is None:
        return None, None
    correction = cc - cb
    return np.asarray(t_bead, float) + correction, correction


DIRECT = 'bead-direct'
TRANSPORTED = 'bead-transported'
UNREGISTERED = 'unregistered'


def sizes_match(a, b, tol_px=0):
    if not (a.known and b.known):
        return False
    return (abs(a.width - b.width) <= tol_px and
            abs(a.height - b.height) <= tol_px)


def origins_match(a, b, tol_px=0):
    """True only when BOTH sources actually knew their origin and they agree."""
    if not (a.origin_known and b.origin_known):
        return False
    if a.x0 is None or b.x0 is None:
        return False
    return abs(a.x0 - b.x0) <= tol_px and abs(a.y0 - b.y0) <= tol_px


def decide_mode(bead_roi, cell_roi, cfg=None):
    """Can the bead transform be applied to this acquisition? How?"""
    tol = getattr(cfg, 'roi_tolerance_px', 0) if cfg is not None else 0
    anchor = (getattr(cfg, 'roi_anchor', 'centred') if cfg is not None
              else 'centred')

    if not bead_roi.known or not cell_roi.known:
        return UNREGISTERED, ('image ROI unknown for '
                              + ('the bead run' if not bead_roi.known
                                 else 'this acquisition')
                              + ' — cannot place the bead transform')

    if sizes_match(bead_roi, cell_roi, tol):
        return DIRECT, (
            f'bead run and acquisition share a {bead_roi.width}x'
            f'{bead_roi.height} crop — the bead affine applies as fitted')

    if anchor == 'centred':
        return TRANSPORTED, (
            f'crop differs (beads {bead_roi.width}x{bead_roi.height}, '
            f'acquisition {cell_roi.width}x{cell_roi.height}) but crops are '
            f'centred, so the translation is moved onto this crop '
            f'analytically')

    extra = ''
    if (bead_roi.sensor and cell_roi.sensor
            and tuple(bead_roi.sensor) == tuple(cell_roi.sensor)):
        extra = (f' (the camera readout DOES match at '
                 f'{bead_roi.sensor[0]}x{bead_roi.sensor[1]} — it is the '
                 f'written crop that differs)')
    return UNREGISTERED, (
        f'crop differs (beads {bead_roi.width}x{bead_roi.height}, acquisition '
        f'{cell_roi.width}x{cell_roi.height}) and roi_anchor={anchor!r}, so '
        f'the origin shift cannot be computed' + extra)


def format_report(bead_roi, cell_roi, mode, reason, label=''):
    """One-line-per-fact block for the console, so the mode is never a mystery."""
    head = f'  camera offset: {mode.upper()}'
    if label:
        head += f'   [{label}]'
    return '\n'.join([
        head,
        f'    bead ROI : {bead_roi.describe()}',
        f'    cell ROI : {cell_roi.describe()}',
        f'    reason   : {reason}',
    ])
