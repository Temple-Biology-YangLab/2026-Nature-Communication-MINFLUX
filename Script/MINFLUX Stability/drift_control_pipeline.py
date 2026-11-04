# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""MINFLUX static-bead drift control (panels A-F): localization precision,
long-term drift and precision-corrected MSD.

Raw data are read from Data/. Each panel is saved as a PNG with its plotted
data as CSV in Panel_<letter>/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# SETTINGS

# Paths
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # e.g. pasted into a console
    BASE_DIR = os.getcwd()

DATA_DIR   = os.path.join(BASE_DIR, "Data")
RAW_NPY    = os.path.join(DATA_DIR, "260623-182050_minflux.npy")  # Abberior export
CACHE_DIR  = os.path.join(DATA_DIR, "_cache")   # compact per-axis arrays live here
USE_CACHE  = True        # reuse cached final-iteration arrays if present
FINAL_ITER = 4           # iteration index that carries the tracking localization

# Extraction (only used if cache must be built)
EXTRACT_CHUNK = 1_000_000   # records per chunk when reading the 2 GB raw file

# Bead detection / isolation
SELECT_DOWNSAMPLE = 3       # use every Nth localization for peak finding + stats
HIST_BIN_NM       = 50      # 2D histogram bin size for peak finding (nm)
FIELD_HALF_NM     = 20000   # half field-of-view for the peak-finding histogram (nm)
PEAK_MAXFILT      = 8       # local-maximum neighborhood (bins)
PEAK_MIN_COUNTS   = 1000    # min localizations at a peak (on downsampled data)
PEAK_MIN_SEP_NM   = 700     # discard peaks with a neighbor closer than this (nm)
MAX_BEADS         = 25      # cap on number of isolated beads analysed
BEAD_WIN_NM       = 150     # +/- window around a peak that defines one bead (nm)
BEAD_MIN_PTS      = 8000    # min localizations (downsampled) to keep a bead
BURST_GAP_S       = 2.0     # time gap that separates two tracking bursts (s)
BURST_MIN_PTS     = 40      # min localizations for a burst to count
BEAD_MIN_BURSTS   = 3       # min bursts for a bead to be usable

# Global figure style
DPI            = 200
SAVE_COMBINED  = True
matplotlib.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                            "figure.autolayout": True})

# Panel A: single-bead localization cloud
A_BEAD_RANK   = 0          # 0 = tightest-precision bead (rank in sorted list)
A_FULLRES     = True       # draw the cloud from full-resolution localizations
A_LIM_NM      = 40         # axis limit, +/- nm
A_PT_SIZE     = 1.0
A_PT_ALPHA    = 0.05
A_PT_COLOR    = "#1f4e79"
A_RINGS_NM    = (11, 22)   # reference circles (e.g. 1 sigma, 2 sigma) in nm
A_RING_COLOR  = "#d00000"

# Panel B: residual drift of one bead
B_BEAD_RANK   = 0
B_YLIM_NM     = 10
B_BAND_NM     = 2          # shaded +/- reference band (nm)
B_COLORS      = {"x": "#c00000", "y": "#008000", "z": "#0000c0"}

# Panel C: 3D drift of many beads
C_NBEADS             = 25
C_MERGE_WINDOW_MIN   = 5.0  # average points within a non-overlapping 5-min window
C_YLIM_NM            = 12
C_ALPHA              = 0.6
C_LW                 = 0.7

# Panel D: localization-precision histogram
D_BINS_NM     = np.arange(0, 30, 2)
D_COLORS      = {"x": "#4c78a8", "y": "#f58518", "z": "#54a24b"}
D_ALPHA       = 0.6

# Panel E: peak-to-peak drift histogram
E_BINS_NM     = np.arange(0, 12, 1)
E_COLOR       = "#555555"
E_MEDLINE     = "#d00000"

# Panel F: static-bead 1D MSD per axis, precision-corrected (- 2*sigma^2)
F_NBEADS         = 12                      # beads averaged into the ensemble MSD
F_MAXLAG_MS      = 1000.0                   # longest lag (ms)
F_DT_S           = None                     # None = auto (median within-burst dt); else seconds
F_AXES           = ("x", "y", "z")          # 1D MSD is computed separately per axis
F_PRECISION_CORR = True                     # subtract 2*sigma^2 offset per axis
F_OFFSET_LAG_MS  = (5.0, 50.0)              # lag window (ms) used to estimate 2*sigma^2
F_SHOW_RAW       = True                     # also draw the uncorrected (raw) 1D MSD (dashed)
F_XLOG           = True                     # log x-axis (lags span ~1-1000 ms)
F_YLIM_NM2       = (-60, 320)               # y-limits (nm^2), tuple (lo, hi)
F_COLORS         = {"x": "#c00000", "y": "#008000", "z": "#0000c0"}

# END SETTINGS


# %% Step 1 - load final-iteration localizations (build cache on first run)
def _cache_paths():
    return {k: os.path.join(CACHE_DIR, f"final_{k}.npy")
            for k in ("x_nm", "y_nm", "z_nm", "t_s")}


def build_cache():
    """Extract itr==FINAL_ITER localizations from the raw .npy in chunks
    (memory-safe via mmap) and cache compact float32 per-axis arrays.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    a = np.load(RAW_NPY, mmap_mode="r")
    n = len(a)
    xs, ys, zs, ts = [], [], [], []
    for i in range(0, n, EXTRACT_CHUNK):
        b = a[i:i + EXTRACT_CHUNK]
        m = b["itr"] == FINAL_ITER
        loc = np.asarray(b["loc"][m])
        xs.append(loc[:, 0]); ys.append(loc[:, 1]); zs.append(loc[:, 2])
        ts.append(np.asarray(b["tim"][m]))
    x = (np.concatenate(xs) * 1e9).astype("f4")
    y = (np.concatenate(ys) * 1e9).astype("f4")
    z = (np.concatenate(zs) * 1e9).astype("f4")
    t = np.concatenate(ts).astype("f8"); t = (t - t.min()).astype("f4")
    p = _cache_paths()
    np.save(p["x_nm"], x); np.save(p["y_nm"], y)
    np.save(p["z_nm"], z); np.save(p["t_s"], t)
    return x, y, z, t


def load_locs():
    p = _cache_paths()
    if USE_CACHE and all(os.path.exists(v) for v in p.values()):
        return (np.load(p["x_nm"]), np.load(p["y_nm"]),
                np.load(p["z_nm"]), np.load(p["t_s"]))
    return build_cache()


# %% Step 2 - detect isolated single beads and compute per-bead statistics
def find_isolated_peaks(xd, yd):
    edges = np.arange(-FIELD_HALF_NM, FIELD_HALF_NM, HIST_BIN_NM)
    H, xe, ye = np.histogram2d(xd, yd, bins=[edges, edges])
    peaks = (H == maximum_filter(H, size=PEAK_MAXFILT)) & (H > PEAK_MIN_COUNTS)
    pi, pj = np.where(peaks)
    px, py, ph = xe[pi] + HIST_BIN_NM / 2, ye[pj] + HIST_BIN_NM / 2, H[pi, pj]
    keep = []
    for i in range(len(px)):
        d = np.hypot(px - px[i], py - py[i]); d[i] = np.inf
        if d.min() > PEAK_MIN_SEP_NM:
            keep.append(i)
    px, py, ph = px[keep], py[keep], ph[keep]
    order = np.argsort(ph)[::-1][:MAX_BEADS]
    return px[order], py[order]


def segment_bursts(ts):
    gaps = np.where(np.diff(ts) > BURST_GAP_S)[0]
    seg = np.split(np.arange(len(ts)), gaps + 1)
    return [s for s in seg if len(s) > BURST_MIN_PTS]


def analyse_beads(x, y, z, t):
    s = slice(None, None, SELECT_DOWNSAMPLE)
    xd, yd, zd, td = x[s], y[s], z[s], t[s]
    px, py = find_isolated_peaks(xd, yd)
    beads = []
    for cx, cy in zip(px, py):
        m = (np.abs(xd - cx) < BEAD_WIN_NM) & (np.abs(yd - cy) < BEAD_WIN_NM)
        if m.sum() < BEAD_MIN_PTS:
            continue
        xs, ys, zs, ts = xd[m], yd[m], zd[m], td[m]
        o = np.argsort(ts); xs, ys, zs, ts = xs[o], ys[o], zs[o], ts[o]
        seg = segment_bursts(ts)
        if len(seg) < BEAD_MIN_BURSTS:
            continue
        sx = np.mean([xs[q].std() for q in seg])
        sy = np.mean([ys[q].std() for q in seg])
        sz = np.mean([zs[q].std() for q in seg])
        bt = np.array([ts[q].mean() for q in seg])
        bx = np.array([np.median(xs[q]) for q in seg])
        by = np.array([np.median(ys[q]) for q in seg])
        bz = np.array([np.median(zs[q]) for q in seg])
        rad = np.hypot(np.hypot(bx - bx[0], by - by[0]), bz - bz[0])
        beads.append(dict(
            cx=float(cx), cy=float(cy), n=int(m.sum() * SELECT_DOWNSAMPLE),
            nbursts=len(seg), sx=sx, sy=sy, sz=sz,
            drift_ptp=float(rad.max()),
            drift_rms=float(np.sqrt(bx.std()**2 + by.std()**2 + bz.std()**2)),
            dur_min=float((bt.max() - bt.min()) / 60.0),
            bt=bt, bx=bx, by=by, bz=bz, seg=seg,
            xs=xs, ys=ys, zs=zs, ts=ts))
    beads.sort(key=lambda b: b["sx"] + b["sy"])   # tightest precision first
    for i, b in enumerate(beads):
        b["bead_id"] = i
    return beads


# %% Step 3 - helpers for saving each panel + its data
def panel_dir(letter):
    d = os.path.join(BASE_DIR, f"Panel_{letter}")
    os.makedirs(d, exist_ok=True)
    return d


def save_panel(fig, letter, df):
    d = panel_dir(letter)
    fig.savefig(os.path.join(d, f"Panel_{letter}.png"), dpi=DPI,
                bbox_inches="tight")
    df.to_csv(os.path.join(d, f"Panel_{letter}_data.csv"), index=False)
    return d


# %% Step 4 - panels
def rep_cloud(beads, rank, x=None, y=None, z=None):
    """Return recentred (dx,dy,dz) for the requested bead's localization cloud."""
    b = beads[rank]
    if A_FULLRES and x is not None:
        m = (np.abs(x - b["cx"]) < BEAD_WIN_NM) & (np.abs(y - b["cy"]) < BEAD_WIN_NM)
        xs, ys, zs = x[m], y[m], z[m]
    else:
        xs, ys, zs = b["xs"], b["ys"], b["zs"]
    return (xs - np.median(xs), ys - np.median(ys), zs - np.median(zs))


def draw_A(ax, beads, x, y, z):
    dx, dy, dz = rep_cloud(beads, A_BEAD_RANK, x, y, z)
    ax.scatter(dx, dy, s=A_PT_SIZE, alpha=A_PT_ALPHA, c=A_PT_COLOR,
               edgecolors="none")
    th = np.linspace(0, 2 * np.pi, 200)
    for r in A_RINGS_NM:
        ax.plot(r * np.cos(th), r * np.sin(th), "-", lw=0.7,
                color=A_RING_COLOR, alpha=0.7)
    ax.set_xlim(-A_LIM_NM, A_LIM_NM); ax.set_ylim(-A_LIM_NM, A_LIM_NM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.set_title(f"A  Static-bead localizations (n={dx.size})", loc="left")
    return pd.DataFrame({"x_nm": dx, "y_nm": dy, "z_nm": dz})


def draw_B(ax, beads):
    b = beads[B_BEAD_RANK]
    tmin = b["bt"] / 60.0
    data = {"time_min": tmin}
    for axis, col in (("x", "bx"), ("y", "by"), ("z", "bz")):
        d = b[col] - b[col][0]
        ax.plot(tmin, d, "-o", ms=3, lw=1, color=B_COLORS[axis], label=axis)
        data[f"drift_{axis}_nm"] = d
    ax.axhspan(-B_BAND_NM, B_BAND_NM, color="gray", alpha=0.12)
    ax.set_ylim(-B_YLIM_NM, B_YLIM_NM)
    ax.set_xlabel("time (min)"); ax.set_ylabel("drift (nm)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("C  Residual drift, one bead", loc="left")
    return pd.DataFrame(data)


def _merge_close_timepoints(tmin, values, window_min):
    """Average consecutive samples within non-overlapping time windows."""
    tmin = np.asarray(tmin, dtype=float)
    values = np.asarray(values, dtype=float)
    if tmin.size != values.size:
        raise ValueError("tmin and values must have the same length")
    if tmin.size == 0:
        return tmin, values, np.array([], dtype=int)

    order = np.argsort(tmin)
    tmin, values = tmin[order], values[order]
    merged_t, merged_values, merged_counts = [], [], []
    start = 0
    for stop in range(1, tmin.size + 1):
        at_end = stop == tmin.size
        outside_window = (
            not at_end and tmin[stop] - tmin[start] > window_min
        )
        if at_end or outside_window:
            merged_t.append(float(np.mean(tmin[start:stop])))
            merged_values.append(float(np.mean(values[start:stop])))
            merged_counts.append(stop - start)
            start = stop

    return (np.asarray(merged_t), np.asarray(merged_values),
            np.asarray(merged_counts, dtype=int))


def draw_C(ax, beads):
    rows = []
    for b in beads[:C_NBEADS]:
        tmin = b["bt"] / 60.0
        r = np.hypot(np.hypot(b["bx"] - b["bx"][0], b["by"] - b["by"][0]),
                     b["bz"] - b["bz"][0])
        tplot, rplot, counts = _merge_close_timepoints(
            tmin, r, C_MERGE_WINDOW_MIN
        )
        ax.plot(tplot, rplot, "-", lw=C_LW, alpha=C_ALPHA)
        for tm, rr, count in zip(tplot, rplot, counts):
            rows.append({"bead_id": b["bead_id"], "time_min": tm,
                         "disp_3d_nm": rr,
                         "n_merged": count})
    ax.set_ylim(0, C_YLIM_NM)
    ax.set_xlabel("time (min)"); ax.set_ylabel("3D displacement (nm)")
    ax.set_title(f"D  Drift, {C_NBEADS} beads", loc="left")
    return pd.DataFrame(rows)


def draw_D(ax, beads):
    sx = [b["sx"] for b in beads]; sy = [b["sy"] for b in beads]
    sz = [b["sz"] for b in beads]
    ax.hist(sx, bins=D_BINS_NM, alpha=D_ALPHA, color=D_COLORS["x"],
            label=f"x (med {np.median(sx):.1f})")
    ax.hist(sy, bins=D_BINS_NM, alpha=D_ALPHA, color=D_COLORS["y"],
            label=f"y (med {np.median(sy):.1f})")
    ax.hist(sz, bins=D_BINS_NM, alpha=D_ALPHA, color=D_COLORS["z"],
            label=f"z (med {np.median(sz):.1f})")
    ax.set_xlabel("localization precision σ (nm)"); ax.set_ylabel("# beads")
    ax.legend(fontsize=8)
    ax.set_title(f"B  Precision, {len(beads)} beads", loc="left")
    return pd.DataFrame({"bead_id": [b["bead_id"] for b in beads],
                         "sigma_x_nm": sx, "sigma_y_nm": sy, "sigma_z_nm": sz})


def draw_E(ax, beads):
    dp = [b["drift_ptp"] for b in beads]
    ax.hist(dp, bins=E_BINS_NM, color=E_COLOR, alpha=0.85)
    ax.axvline(np.median(dp), color=E_MEDLINE, lw=1)
    ax.set_xlabel("peak-to-peak 3D drift (nm)"); ax.set_ylabel("# beads")
    ax.set_title(f"E  Drift, median {np.median(dp):.1f} nm", loc="left")
    return pd.DataFrame({"bead_id": [b["bead_id"] for b in beads],
                         "drift_ptp_nm": dp,
                         "drift_rms_nm": [b["drift_rms"] for b in beads],
                         "duration_min": [b["dur_min"] for b in beads]})


def _autocorr_fft(x):
    """Autocorrelation of a 1D array via FFT, normalised per lag."""
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * f.conjugate())[:n].real
    return acf / (n - np.arange(n))


def _msd_1d_fft(r):
    """Per-lag 1D MSD via the Fast Correlation Algorithm (lags 0..N-1).
    O(N log N) per trajectory, so lags out to ~1 s stay cheap.
    """
    n = len(r)
    d = np.append(r ** 2, 0.0)
    s2 = _autocorr_fft(r)
    q = 2.0 * d.sum()
    s1 = np.zeros(n)
    for m in range(n):
        q = q - d[m - 1] - d[n - m]
        s1[m] = q / (n - m)
    return s1 - 2.0 * s2


def _msd_dt(beads):
    dts = [np.median(np.diff(b["ts"][b["seg"][0]])) for b in beads[:F_NBEADS]]
    return F_DT_S if F_DT_S is not None else float(np.median(dts))


def static_msd_1d(beads, coord, maxlag):
    """Pair-weighted ensemble 1D within-burst MSD for one axis.
    Every tracking burst contributes (n_burst - lag) pairs at each lag; the
    statistics thin out beyond the median burst length (see 'n_pairs').
    """
    key = coord + "s"                       # 'xs' / 'ys' / 'zs'
    M = np.zeros(maxlag); C = np.zeros(maxlag)
    for b in beads[:F_NBEADS]:
        A = b[key]
        for q in b["seg"]:
            X = np.asarray(A[q], dtype=float)
            n = len(X)
            if n < 4:
                continue
            m = _msd_1d_fft(X)
            k = min(maxlag, n)
            w = (n - np.arange(k))          # pairs contributing at each lag
            M[:k] += m[:k] * w
            C[:k] += w
    return M / np.maximum(C, 1), C


def draw_F(ax, beads):
    dt = _msd_dt(beads)
    maxlag = int(round(F_MAXLAG_MS / (dt * 1000.0))) + 1
    tau_ms = np.arange(maxlag) * dt * 1000.0
    lo, hi = F_OFFSET_LAG_MS
    owin = (tau_ms >= lo) & (tau_ms <= hi)
    data = {"lag_ms": tau_ms[1:]}
    npairs = None
    for coord in F_AXES:
        M, C = static_msd_1d(beads, coord, maxlag)
        npairs = C
        offset = float(np.mean(M[owin]))         # 2*sigma^2 (flat short-lag plateau)
        sigma = np.sqrt(max(offset, 0.0) / 2.0)
        c = F_COLORS[coord]
        if F_SHOW_RAW:
            ax.plot(tau_ms[1:], M[1:], "--", lw=1, color=c, alpha=0.45)
        if F_PRECISION_CORR:
            Mc = M - offset
            ax.plot(tau_ms[1:], Mc[1:], "-", lw=1.3, color=c,
                    label=f"{coord}  2σ²={offset:.0f} nm² (σ={sigma:.1f} nm)")
            data[f"msd_{coord}_raw_nm2"] = M[1:]
            data[f"msd_{coord}_corrected_nm2"] = Mc[1:]
        else:
            ax.plot(tau_ms[1:], M[1:], "-", lw=1.3, color=c, label=coord)
            data[f"msd_{coord}_raw_nm2"] = M[1:]
    data["n_pairs"] = npairs[1:].astype(int)
    ax.axhline(0, color="k", lw=0.7)
    if F_XLOG:
        ax.set_xscale("log")
    ax.set_ylim(*F_YLIM_NM2)
    ax.set_xlabel("lag τ (ms)"); ax.set_ylabel("1D MSD (nm²)")
    ax.legend(fontsize=7, title="dashed = raw, solid = corrected")
    ax.set_title("F  Static-bead 1D MSD (precision-corrected)", loc="left")
    return pd.DataFrame(data)


# %% Burst-timing summary
def report_burst_timing(beads):
    durs, gaps, nb = [], [], []
    for b in beads:
        ts, seg = b["ts"], b["seg"]
        nb.append(len(seg))
        starts, ends = [], []
        for q in seg:
            durs.append(float(ts[q].max() - ts[q].min()))
            starts.append(float(ts[q].min())); ends.append(float(ts[q].max()))
        starts, ends = np.array(starts), np.array(ends)
        gaps.extend((starts[1:] - ends[:-1]).tolist())
    durs, gaps, nb = np.array(durs), np.array(gaps), np.array(nb)
    summary = pd.DataFrame([
        {"quantity": "bursts (repeats) per bead",
         "median": np.median(nb), "min": nb.min(), "max": nb.max(), "unit": "count"},
        {"quantity": "burst duration",
         "median": np.median(durs), "min": durs.min(), "max": durs.max(), "unit": "s"},
        {"quantity": "time between bursts (revisit interval)",
         "median": np.median(gaps), "min": gaps.min(), "max": gaps.max(), "unit": "s"},
    ])
    summary.to_csv(os.path.join(BASE_DIR, "burst_timing_summary.csv"), index=False)
    print("  Burst timing:")
    print(f"    repeats/bead : median {int(np.median(nb))} "
          f"(range {nb.min()}-{nb.max()})")
    print(f"    burst length : median {np.median(durs):.2f} s "
          f"(range {durs.min():.2f}-{durs.max():.2f} s)")
    print(f"    revisit gap  : median {np.median(gaps):.0f} s "
          f"(~{np.median(gaps)/60:.1f} min)")
    return summary


# %% Step 5 - run everything
def main():
    print("Loading localizations ...")
    x, y, z, t = load_locs()
    print(f"  {x.size:,} final-iteration localizations, "
          f"{(t.max()-t.min())/60:.0f} min acquisition")

    print("Detecting isolated single beads ...")
    beads = analyse_beads(x, y, z, t)
    print(f"  {len(beads)} isolated beads analysed")

    # master summary table
    summ = pd.DataFrame([{k: b[k] for k in
                          ("bead_id", "cx", "cy", "n", "nbursts",
                           "sx", "sy", "sz", "drift_ptp", "drift_rms",
                           "dur_min")} for b in beads])
    summ.to_csv(os.path.join(BASE_DIR, "beads_summary.csv"), index=False)

    # burst-timing summary (context for the long-term drift panels C/D/E)
    report_burst_timing(beads)

    panels = [("A", lambda ax: draw_A(ax, beads, x, y, z)),
              ("B", lambda ax: draw_D(ax, beads)),
              ("C", lambda ax: draw_B(ax, beads)),
              ("D", lambda ax: draw_C(ax, beads)),
              ("E", lambda ax: draw_E(ax, beads)),
              ("F", lambda ax: draw_F(ax, beads))]

    combined = plt.figure(figsize=(12, 8)) if SAVE_COMBINED else None
    for i, (letter, fn) in enumerate(panels):
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        df = fn(ax)
        d = save_panel(fig, letter, df)
        plt.close(fig)
        print(f"  Panel {letter}  ->  {os.path.relpath(d, BASE_DIR)}"
              f"  ({len(df)} rows)")
        if combined is not None:
            axc = combined.add_subplot(2, 3, i + 1)
            fn(axc)
    if combined is not None:
        combined.savefig(os.path.join(BASE_DIR,
                         "MINFLUX_drift_control_combined.png"), dpi=DPI)
        plt.close(combined)
    print("Done.")


if __name__ == "__main__":
    main()
