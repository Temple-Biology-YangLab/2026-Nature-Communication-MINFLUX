# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trace linking, trace statistics and assignment of traces to pores."""
from dataclasses import dataclass

import numpy as np


@dataclass
class LinkConfig:
    """Parameters for building traces from raw localizations."""
    distance_threshold_nm: float = 200.0
    max_frame_gap: int = 1      # 1 = consecutive frames only; 2 allows one blink off
    max_locs_per_frame_link: int = 1   # nearest-neighbour, one-to-one


@dataclass
class TraceConfig:
    """Trace assembly and NPC-association criteria."""
    min_length: int = 6            # minimum localizations (frames) per trace
    assign_radius_nm: float = 80.0 # how close a localization counts as "on" a pore
    min_occupancy: float = 0.5     # fraction of the trace that must be on ONE pore
    max_rg_nm: float = None        # optional cap on radius of gyration
    max_span_nm: float = None      # optional cap on max displacement
    require_contiguous: bool = False  # drop traces with frame gaps


class Trace(dict):
    @property
    def n(self):
        return len(self['x'])


def link_traces(locs, cfg=None):
    """Build traces from raw localizations by frame-to-frame nearest-neighbour
    linking. Use this for SMAP CamT, which ships localizations but no traces.
    """
    cfg = cfg or LinkConfig()
    x = np.asarray(locs['xnm'], float)
    y = np.asarray(locs['ynm'], float)
    fr = np.asarray(locs['frame'], float)

    order = np.argsort(fr, kind='stable')
    xs, ys, fs = x[order], y[order], fr[order]
    n = len(xs)
    tid = np.full(n, -1.0)

    frames = np.unique(fs)
    idx_by_frame = {f: np.flatnonzero(fs == f) for f in frames}
    next_id = 0
    active = {}          # trace_id -> (last_index, last_frame)

    from scipy.spatial import cKDTree
    for f in frames:
        cur = idx_by_frame[f]
        tails = [(t, i) for t, (i, lf) in active.items()
                 if 0 < f - lf <= cfg.max_frame_gap]
        if tails:
            tail_idx = np.array([i for _, i in tails])
            tree = cKDTree(np.c_[xs[tail_idx], ys[tail_idx]])
            pairs = tree.query_ball_point(np.c_[xs[cur], ys[cur]],
                                          cfg.distance_threshold_nm)
            cand = []
            for k, plist in enumerate(pairs):
                for p in plist:
                    d = np.hypot(xs[cur[k]] - xs[tail_idx[p]],
                                 ys[cur[k]] - ys[tail_idx[p]])
                    cand.append((d, k, p))
            cand.sort()
            used_cur, used_tail = set(), set()
            for d, k, p in cand:
                if k in used_cur or p in used_tail:
                    continue
                used_cur.add(k); used_tail.add(p)
                t = tails[p][0]
                tid[cur[k]] = t
                active[t] = (cur[k], f)
        for k in range(len(cur)):
            if tid[cur[k]] < 0:
                tid[cur[k]] = next_id
                active[next_id] = (cur[k], f)
                next_id += 1
        active = {t: v for t, v in active.items()
                  if f - v[1] <= cfg.max_frame_gap}

    out = np.empty(n)
    out[order] = tid
    return locs.copy_with(trace_id=out)


def build_traces(locs, cfg=None, id_field='trace_id'):
    """Group localizations into traces using GDSC's Id column."""
    cfg = cfg or TraceConfig()
    if id_field not in locs:
        raise ValueError(f'no {id_field} column — is this a trace file?')
    tid = np.asarray(locs[id_field], float)
    x = np.asarray(locs['xnm'], float)
    y = np.asarray(locs['ynm'], float)
    fr = np.asarray(locs['frame'], float)
    prec = np.asarray(locs['locprecnm'], float) if 'locprecnm' in locs else None

    order = np.argsort(tid, kind='stable')
    tid, x, y, fr = tid[order], x[order], y[order], fr[order]
    if prec is not None:
        prec = prec[order]
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    ends = np.r_[starts[1:], len(tid)]

    out = []
    for s, e in zip(starts, ends):
        if e - s < cfg.min_length:
            continue
        f = fr[s:e]
        if cfg.require_contiguous and np.any(np.diff(np.sort(f)) > 1):
            continue
        tx, ty = x[s:e], y[s:e]
        cx, cy = float(tx.mean()), float(ty.mean())
        rg = float(np.sqrt(np.mean((tx - cx) ** 2 + (ty - cy) ** 2)))
        span = float(np.max(np.hypot(tx - cx, ty - cy))) if len(tx) else 0.0
        if cfg.max_rg_nm is not None and rg > cfg.max_rg_nm:
            continue
        if cfg.max_span_nm is not None and span > cfg.max_span_nm:
            continue
        out.append(Trace(id=float(tid[s]), x=tx, y=ty, frame=f,
                         prec=None if prec is None else prec[s:e],
                         cx=cx, cy=cy, rg_nm=rg, span_nm=span,
                         length=int(e - s)))
    return out


def assign_traces_to_pores(traces, centres, cfg=None):
    """Assign each trace to its best NPC and test occupancy."""
    cfg = cfg or TraceConfig()
    if not traces or len(centres) == 0:
        return [], []
    C = np.array([[c['x0'], c['y0']] if isinstance(c, dict) else [c[0], c[1]]
                  for c in centres], float)
    r2 = cfg.assign_radius_nm ** 2

    kept, allt = [], []
    for t in traces:
        d2 = ((t['x'][:, None] - C[None, :, 0]) ** 2 +
              (t['y'][:, None] - C[None, :, 1]) ** 2)
        on = d2 <= r2                       # (n_locs, n_pores)
        counts = on.sum(axis=0)
        best = int(np.argmax(counts))
        occ = float(counts[best] / t.n)
        t = Trace(dict(t))
        t['pore_index'] = best
        t['occupancy'] = occ
        t['n_on_pore'] = int(counts[best])
        t['dist_to_pore_nm'] = float(np.hypot(t['cx'] - C[best, 0],
                                              t['cy'] - C[best, 1]))
        t['n_pores_touched'] = int((counts > 0).sum())
        t['keep'] = occ >= cfg.min_occupancy
        allt.append(t)
        if t['keep']:
            kept.append(t)
    return kept, allt


def trace_locs(traces, pore_only=True, cfg=None):
    """Flatten traces into localization arrays."""
    cfg = cfg or TraceConfig()
    xs, ys, pid, tid = [], [], [], []
    for t in traces:
        if pore_only and 'pore_index' in t:
            keep = np.hypot(t['x'] - t['cx'], t['y'] - t['cy']) >= -1  # all
        m = np.ones(t.n, bool)
        xs.append(t['x'][m]); ys.append(t['y'][m])
        pid.append(np.full(m.sum(), t.get('pore_index', -1), float))
        tid.append(np.full(m.sum(), t['id'], float))
    if not xs:
        return dict(x=np.array([]), y=np.array([]),
                    pore_index=np.array([]), trace_id=np.array([]))
    return dict(x=np.concatenate(xs), y=np.concatenate(ys),
                pore_index=np.concatenate(pid), trace_id=np.concatenate(tid))


def radial_from_traces(traces, centres, r_keep_nm=120.0, pore_only=True):
    """Radial distances of trace localizations about their ASSIGNED pore."""
    C = np.array([[c['x0'], c['y0']] if isinstance(c, dict) else [c[0], c[1]]
                  for c in centres], float)
    out = []
    for t in traces:
        i = t.get('pore_index')
        if i is None or i < 0 or i >= len(C):
            continue
        d = np.hypot(t['x'] - C[i, 0], t['y'] - C[i, 1])
        out.append(d[d <= r_keep_nm])
    return np.concatenate(out) if out else np.array([])


def summarize(all_traces):
    """Quick diagnostic over a trace set."""
    if not all_traces:
        return dict(n=0)
    L = np.array([t['length'] for t in all_traces])
    occ = np.array([t.get('occupancy', np.nan) for t in all_traces])
    rg = np.array([t['rg_nm'] for t in all_traces])
    return dict(n=len(all_traces),
                length_median=float(np.median(L)), length_max=int(L.max()),
                occupancy_median=float(np.nanmedian(occ)),
                kept=int(sum(bool(t.get('keep')) for t in all_traces)),
                rg_median_nm=float(np.median(rg)))
