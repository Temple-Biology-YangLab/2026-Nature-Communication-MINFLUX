# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the OU-confined cluster model for each cluster number, analyse simulated and
experimental traces with the manuscript pipeline, and write results to results/.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_core as sc

# SETTINGS
CLUSTER_NUMBERS = [4, 5, 6, 7, 8]
RING_RADIUS_NM  = 10.0
DWELL_S         = 0.025          # mean time at one cluster before switching
N_TRACES        = 600            # simulated traces per construct per cluster number
N_WORKERS       = min(8, os.cpu_count() or 1)

W_CENTRAL       = 10.0
W_ANNULUS       = 5.0
ANNULUS_BAND_NM = 3.0
EXAMPLE_TRACES  = {4: ('HaloTag-Nup54-GFP', 264),
                   5: ('HaloTag-Nup58-GFP', 54),
                   6: ('HaloTag-Nup62-GFP', 111),
                   7: ('GFP-Nup58-HaloTag', 497),
                   8: ('HaloTag-Nup62-GFP', 41)}
EXAMPLES_ONLY   = False

DATA         = os.path.join(HERE, 'data')
RESULTS      = os.path.join(HERE, 'results')
EXAMPLE_DIRS = [os.path.join(HERE, '..', 'traces_AtoE'), os.path.join(HERE, '..', 'localmax_FtoJ')]


def run_trace(args):
    c, rep = args
    obs, truth = sc.simulate_trace(c, np.random.default_rng(sc.trace_seed(c, rep)), return_truth=True)
    res, pk = sc.analyze_trace(sc.preprocess(obs, c), c)
    off = np.nanmean(np.asarray(obs, dtype=np.float32), axis=0).astype(float)
    tc = -off[:2]                                    # true ring centre in the centred frame
    d = np.hypot(pk.x_refined_nm - c['CENTER_NM'] - tc[0], pk.y_refined_nm - c['CENTER_NM'] - tc[1]).to_numpy() if len(pk) else np.array([])
    R = c['RING_RADIUS_NM']
    res.update(replicate=rep,
               centre_error_nm=float(np.hypot(res['centre_x_nm'] - tc[0], res['centre_y_nm'] - tc[1])),
               frac_central=float(np.mean(d < R / 2)) if d.size else np.nan,
               frac_annulus=float(np.mean(np.abs(d - R) <= ANNULUS_BAND_NM)) if d.size else np.nan,
               clusters_visited=int(len(np.unique(truth['state']))))
    return res


def save_example(c, rep, K, metrics):
    """Write example_K<K>.npz (everything needed to draw the trace, nm, centred on the trace mean)
    and example_K<K>.csv (one-row summary) into each panel folder.
    """
    obs, truth = sc.simulate_trace(c, np.random.default_rng(sc.trace_seed(c, rep)), return_truth=True)
    pts = sc.preprocess(obs, c)
    res, pk = sc.analyze_trace(pts, c)
    off = np.nanmean(np.asarray(obs, dtype=np.float32), axis=0).astype(float)
    for d in EXAMPLE_DIRS:
        save_npz(os.path.join(d, f'example_K{K}.npz'), pts, truth, off, pk, res, c, rep)
        pd.DataFrame([dict(n_clusters=K, construct=c['CONSTRUCT'], replicate=rep, **metrics)]).to_csv(
            os.path.join(d, f'example_K{K}.csv'), index=False)


def save_npz(path, pts, truth, off, pk, res, c, rep):
    np.savez(path, localizations=pts.astype(float) - c['CENTER_NM'], true_path=truth['latent'] - off,
             cluster_centres=truth['centres'] - off, ring_centre=-off[:2], ring_radius=c['RING_RADIUS_NM'],
             maxima_xyz=np.c_[pk.x_refined_nm, pk.y_refined_nm, pk.z_nm] - c['CENTER_NM'] if len(pk) else np.zeros((0, 3)),
             maxima_density=pk.density.to_numpy() if len(pk) else np.zeros(0),
             fit_centre=np.array([res['centre_x_nm'], res['centre_y_nm']]), fit_radius=res['radius_nm'],
             construct=c['CONSTRUCT'], replicate=rep, n_clusters=c['N_CLUSTERS'])


def write_example(K, sim, config=None):
    best = sim.sort_values('score').iloc[0]
    con, rep = EXAMPLE_TRACES.get(K, (best.construct, int(best.replicate)))
    row = sim[(sim.construct == con) & (sim.replicate == rep)].iloc[0]
    metrics = {k: row[k] for k in ('radius_nm', 'centre_error_nm', 'frac_central', 'frac_annulus', 'n_maxima', 'score')}
    save_example(config or sc.make_config(con, K, RING_RADIUS_NM, DWELL_S), rep, K, metrics)
    print(f'K={K}: example {sc.SHORT[con]} replicate {rep}', flush=True)


def main():
    if EXAMPLES_ONLY:
        for K in CLUSTER_NUMBERS:
            write_example(K, pd.read_csv(os.path.join(RESULTS, f'sim_per_trace_K{K}.csv')))
        return
    os.makedirs(RESULTS, exist_ok=True)
    configs = {}
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for K in CLUSTER_NUMBERS:
            frames = []
            for con in sc.CONSTRUCTS:
                c = sc.make_config(con, K, RING_RADIUS_NM, DWELL_S); configs[(K, con)] = c
                df = pd.DataFrame(list(ex.map(run_trace, [(c, r) for r in range(N_TRACES)], chunksize=4)))
                df.insert(0, 'construct', con); df.insert(0, 'n_clusters', K)
                frames.append(df)
                print(f'K={K} {sc.SHORT[con]}: cluster SD {c["CLUSTER_SD_NM"][0]:.2f}/{c["CLUSTER_SD_NM"][1]:.2f} nm, '
                      f'median radius {df.radius_nm.median():.2f} nm', flush=True)
            sim = pd.concat(frames, ignore_index=True)
            sim['score'] = (sim.centre_error_nm + (sim.radius_nm - RING_RADIUS_NM).abs()
                            + W_CENTRAL * sim.frac_central - W_ANNULUS * sim.frac_annulus)
            sim.to_csv(os.path.join(RESULTS, f'sim_per_trace_K{K}.csv'), index=False)
            best = sim.sort_values('score').iloc[0]
            con = EXAMPLE_TRACES.get(K, (best.construct, 0))[0]
            write_example(K, sim, configs[(K, con)])

    # experimental traces, same analysis (cropped to the 120 nm cube)
    ref = pd.read_csv(os.path.join(DATA, 'experimental_reference_radii.csv'))
    Z = np.load(os.path.join(DATA, 'experimental_traces.npz'))
    ce = dict(sc.ANALYSIS, POINT_HANDLING='crop')
    rows = []
    for r in ref.itertuples():
        res, _ = sc.analyze_trace(sc.preprocess(Z[r.trace_id], ce), ce)
        res.update(construct=r.construct, trace_id=r.trace_id); rows.append(res)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'experiment_per_trace.csv'), index=False)

    # settings and calibrated parameters
    cal = [dict(n_clusters=K, construct=con, cluster_sd_x_nm=c['CLUSTER_SD_NM'][0], cluster_sd_y_nm=c['CLUSTER_SD_NM'][1],
                axial_scale=c['Z_SCALE'], achieved_sd_nm=str(c['ACHIEVED_SD_NM']),
                target_sd_nm=str(sc.CONSTRUCT[con]['target_sd'])) for (K, con), c in configs.items()]
    pd.DataFrame(cal).to_csv(os.path.join(RESULTS, 'calibration.csv'), index=False)
    with open(os.path.join(RESULTS, 'run_settings.json'), 'w') as f:
        json.dump(dict(cluster_numbers=CLUSTER_NUMBERS, ring_radius_nm=RING_RADIUS_NM, dwell_s=DWELL_S,
                       n_traces=N_TRACES, score=dict(w_central=W_CENTRAL, w_annulus=W_ANNULUS, annulus_band_nm=ANNULUS_BAND_NM),
                       example_traces={str(k): v for k, v in EXAMPLE_TRACES.items()},
                       construct_parameters=sc.CONSTRUCT, analysis=sc.ANALYSIS), f, indent=2)
    print('Results:', RESULTS)


if __name__ == '__main__':
    main()
