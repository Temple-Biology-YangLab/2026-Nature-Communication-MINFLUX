# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Drift: SMAP's stored CamR drift curve evaluated at CamT frames (by time, ROI-R merged
N:1 aware), timestamps, and the cross-correlation shift used by registration."""
import os

import numpy as np

from .io_smap import read_driftinfo


class Timestamps:
    """Abbelight Timestamp.txt: 'Frame,CamR,CamT', one row per raw frame (s)."""

    def __init__(self, frame, t_r, t_t, path=None):
        o = np.argsort(frame)
        self.frame = np.asarray(frame, int)[o]
        self.t_r = np.asarray(t_r, float)[o]
        self.t_t = np.asarray(t_t, float)[o]
        self.path = path
        t0 = min(self.t_r[0], self.t_t[0])
        self.t_r = self.t_r - t0
        self.t_t = self.t_t - t0

    @classmethod
    def read(cls, path):
        a = np.genfromtxt(path, delimiter=',', skip_header=1)
        a = np.atleast_2d(a)
        a = a[np.all(np.isfinite(a[:, :3]), axis=1)]
        return cls(a[:, 0], a[:, 1], a[:, 2], path)

    @property
    def n(self):
        return len(self.frame)

    def interval_ms(self, cam='R'):
        t = self.t_r if cam == 'R' else self.t_t
        return float(np.median(np.diff(t)) * 1000.0)

    def sync_ms(self):
        """Largest |CamR - CamT| time of the same raw frame (ms)."""
        return float(np.max(np.abs(self.t_r - self.t_t)) * 1000.0)

    def frame_time(self, frames, cam='T'):
        """Time (s) of 1-based raw frames; NaN outside the file."""
        t = self.t_r if cam == 'R' else self.t_t
        f = np.asarray(frames, float)
        out = np.full(f.shape, np.nan)
        ok = np.isfinite(f) & (f >= 1) & (f <= len(t))
        out[ok] = t[f[ok].astype(int) - 1]
        return out

    def merged_time(self, n_merge, n_groups, cam='R'):
        """Time of merged frame k = mean time of its raw frames (k-1)N+1 .. kN."""
        t = self.t_r if cam == 'R' else self.t_t
        m = min(int(n_groups), len(t) // int(n_merge))
        out = np.full(int(n_groups), np.nan)
        out[:m] = t[:m * n_merge].reshape(m, n_merge).mean(axis=1)
        return out


def find_file(folder, name):
    """Case-insensitive lookup of one file inside a folder."""
    p = os.path.join(folder, name)
    if os.path.exists(p):
        return p
    try:
        for f in os.listdir(folder):
            if f.lower() == name.lower():
                return os.path.join(folder, f)
    except OSError:
        pass
    return None


def read_timestamps(folder, name='Timestamp.txt'):
    p = find_file(folder, name)
    if p is None:
        return None
    try:
        ts = Timestamps.read(p)
        return ts if ts.n > 1 else None
    except Exception:
        return None


def tif_frames(path):
    """Number of frames in a tif, without reading the pixels. None if unreadable."""
    if not path or not os.path.exists(path):
        return None
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            shape = t.series[0].shape
            return 1 if len(shape) == 2 else int(np.prod(shape[:-2]))
    except Exception:
        return None


class DriftCurve:
    """CamR drift (per CamR frame, possibly merged N:1) evaluated at CamT frames."""

    def __init__(self, camr_frames, dx, dy, n_merge=1, timestamps=None,
                 source=''):
        f = np.asarray(camr_frames, float)
        o = np.argsort(f)
        self.camr_frames = f[o]                     # merged CamR frame numbers
        self.n_camr_frames = int(round(float(np.nanmax(f)))) if len(f) else 0
        self.dx = np.asarray(dx, float)[o]
        self.dy = np.asarray(dy, float)[o]
        self.n_merge = int(n_merge)
        self.ts = timestamps
        self.source = source
        if timestamps is not None:
            K = int(np.ceil(self.camr_frames.max()))
            k = np.arange(1, K + 1, dtype=float)
            tk = timestamps.merged_time(self.n_merge, K, cam='R')
            good = np.isfinite(tk)
            if good.sum() >= 2:
                self.t = np.interp(self.camr_frames, k[good], tk[good],
                                   left=np.nan, right=np.nan)
            else:
                self.t = np.full(len(self.camr_frames), np.nan)
            self.unit = 's'
        else:
            self.t = (self.camr_frames - 0.5) * self.n_merge + 0.5
            self.unit = 'raw frame'
        ok = np.isfinite(self.t)
        self.n_curve_beyond_timestamps = int((~ok).sum())
        self.t, self.dx, self.dy = self.t[ok], self.dx[ok], self.dy[ok]
        self.camr_frames = self.camr_frames[ok]
        if len(self.t) < 2:
            raise ValueError('drift curve has fewer than 2 usable points '
                             '(Timestamp.txt shorter than the CamR movie?)')
        step = float(np.median(np.diff(self.t)))
        self.tol = 0.5 * step       # CamT frame 1 sits half a group before t_R(1)
        self.frame_bin = float(self.n_merge)

    def target_time(self, frame):
        f = np.asarray(frame, float)
        if self.ts is not None:
            return self.ts.frame_time(f, cam='T')
        return f

    def covers(self, frame):
        t = self.target_time(frame)
        return (np.isfinite(t) & (t >= self.t[0] - self.tol)
                & (t <= self.t[-1] + self.tol))

    def at(self, frame):
        t = self.target_time(frame)
        t = np.where(np.isfinite(t), t, self.t[-1])
        return np.interp(t, self.t, self.dx), np.interp(t, self.t, self.dy)

    def apply(self, x, y, frame, return_flags=False):
        dx, dy = self.at(frame)
        xc = np.asarray(x, float) - dx
        yc = np.asarray(y, float) - dy
        if return_flags:
            return xc, yc, ~self.covers(frame)
        return xc, yc

    @property
    def span_nm(self):
        return float(np.ptp(self.dx)), float(np.ptp(self.dy))

    def __repr__(self):
        return (f'<DriftCurve n={len(self.dx)} merge={self.n_merge} by '
                f'{self.unit} span x={self.span_nm[0]:.1f} '
                f'y={self.span_nm[1]:.1f} nm>')


def from_stored(sml_path, n_merge=1, timestamps=None):
    """SMAP's stored drift of a CamR sml, mapped for CamT. None if absent."""
    d = read_driftinfo(sml_path)
    if d is None:
        return None
    if 'dxt' in d and len(d['dxt']):
        k = np.arange(1, len(d['dxt']) + 1)
        return DriftCurve(k, d['dxt'], d['dyt'], n_merge, timestamps,
                                f'stored:{sml_path}')
    if 'dx' in d and 'binframes' in d:
        return DriftCurve(d['binframes'], d['dx'], d['dy'], n_merge,
                                timestamps, f'stored-binned:{sml_path}')
    return None


def check_merge(n_camr_frames, n_merge, n_raw_frames, tol=0.05):
    """Does the CamR movie length agree with N and the raw (CamT) length?"""
    if not n_raw_frames or not n_camr_frames:
        return True, np.nan, 'raw frame count unavailable; merge factor not checked'
    r = float(n_raw_frames) / float(n_camr_frames)
    ok = (int(round(r)) == int(n_merge)) and abs(r - n_merge) <= tol * n_merge
    msg = (f'CamR {n_camr_frames} frames x N={n_merge} vs {n_raw_frames} raw '
           f'frames (implied N = {r:.3f})')
    if not ok:
        msg += (f' — ROI_R_MERGE_FRAMES does not match this acquisition '
                f'(looks like N = {max(1, int(round(r)))})')
    return ok, r, msg


def _xcorr_shift(a, b, max_shift_px):
    """Sub-pixel shift of b relative to a by FFT cross-correlation."""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    cc = np.fft.irfft2(A * np.conj(B), s=a.shape)
    cc = np.fft.fftshift(cc)
    cy, cx = np.array(cc.shape) // 2
    m = int(max(1, max_shift_px))
    sub = cc[max(0, cy - m):cy + m + 1, max(0, cx - m):cx + m + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    def refine(v, i):
        if 0 < i < len(v) - 1:
            d = v[i - 1] - 2 * v[i] + v[i + 1]
            if d != 0:
                return i + 0.5 * (v[i - 1] - v[i + 1]) / d
        return float(i)
    fy = refine(sub[:, ix], iy)
    fx = refine(sub[iy, :], ix)
    return (fx - (sub.shape[1] - 1) / 2.0), (fy - (sub.shape[0] - 1) / 2.0)
