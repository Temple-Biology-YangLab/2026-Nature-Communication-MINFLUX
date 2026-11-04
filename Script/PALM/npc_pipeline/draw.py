# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared rendering primitives for the figures."""
import numpy as np

SCALEBAR_NM = 50.0
RED = '#ff5a3c'            # target-channel scatter fill
FIRE_LINE = '#f0883e'      # target-channel trace line
FIRE_DOT = '#ff5a3c'       # target-channel trace marker
RING_COLOR = '#58a6ff'
SECTOR_COLOR = '#d29922'


def ice_cmap():
    """black -> blue -> cyan -> white. The reference channel."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('ice', [
        (0.00, (0.00, 0.00, 0.00)),
        (0.33, (0.10, 0.20, 1.00)),
        (0.66, (0.20, 0.90, 1.00)),
        (1.00, (1.00, 1.00, 1.00))])


def fire_cmap():
    """black -> red -> orange -> white. The target channel."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('fire', [
        (0.00, (0.00, 0.00, 0.00)),
        (0.33, (1.00, 0.15, 0.05)),
        (0.66, (1.00, 0.60, 0.10)),
        (1.00, (1.00, 1.00, 0.90))])


_ICE = None
_FIRE = None


def ICE():
    global _ICE
    if _ICE is None:
        _ICE = ice_cmap()
    return _ICE


def FIRE():
    global _FIRE
    if _FIRE is None:
        _FIRE = fire_cmap()
    return _FIRE


def blur_image(x, y, extent, pixel_nm=2.0, bw_nm=4.0):
    """PALM-style render: localization histogram at `pixel_nm` resolution,
    Gaussian-blurred by `bw_nm`.
    """
    from scipy.ndimage import gaussian_filter
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    bins = max(8, int(round(2.0 * extent / pixel_nm)))
    H, _, _ = np.histogram2d(x, y, bins=bins,
                             range=[[-extent, extent], [-extent, extent]])
    if bw_nm <= 0:
        return H.T
    return gaussian_filter(H.T, bw_nm / pixel_nm)


def show(ax, Z, extent, cmap):
    ax.imshow(Z, origin='lower', extent=[-extent, extent, -extent, extent],
              cmap=cmap, interpolation='nearest')


def channel_vmax(ref_xy, tgt_xy, extent, pixel_nm, bw_ref, bw_tgt, pct=99.0):
    """Percentile-based scale so a few hot pixels don't dim the bulk."""
    ref_xy = np.asarray(ref_xy, float).reshape(-1, 2)
    tgt_xy = np.asarray(tgt_xy, float).reshape(-1, 2)
    Zb = blur_image(ref_xy[:, 0], ref_xy[:, 1], extent, pixel_nm, bw_ref)
    Zr = blur_image(tgt_xy[:, 0], tgt_xy[:, 1], extent, pixel_nm, bw_tgt)
    vb = np.percentile(Zb[Zb > 0], pct) if (Zb > 0).any() else 1.0
    vr = np.percentile(Zr[Zr > 0], pct) if (Zr > 0).any() else 1.0
    return (vb or 1.0, vr or 1.0)


def two_color(ref_xy, tgt_xy, extent, pixel_nm, bw_ref, bw_tgt, vmax=None,
              ref_floor=0.18, ref_gamma=1.0, tgt_gamma=0.55):
    """Two-colour composite: reference through ICE, target through FIRE."""
    ref_xy = np.asarray(ref_xy, float).reshape(-1, 2)
    tgt_xy = np.asarray(tgt_xy, float).reshape(-1, 2)
    Zb = blur_image(ref_xy[:, 0], ref_xy[:, 1], extent, pixel_nm, bw_ref)
    Zr = blur_image(tgt_xy[:, 0], tgt_xy[:, 1], extent, pixel_nm, bw_tgt)
    vb = (np.percentile(Zb[Zb > 0], 99) if vmax is None else vmax[0]) or 1.0
    vr = (np.percentile(Zr[Zr > 0], 99) if vmax is None else vmax[1]) or 1.0

    tb = np.clip(Zb / vb, 0, 1)
    tb = np.clip((tb - ref_floor) / (1 - ref_floor), 0, 1)
    tb = tb ** ref_gamma
    tr = np.clip(Zr / vr, 0, 1) ** tgt_gamma

    rgb_ice = ICE()(tb)[..., :3]
    rgb_fire = FIRE()(tr)[..., :3]
    a = tr[..., None]
    return np.clip(rgb_ice * (1 - a) + rgb_fire, 0, 1)


def draw_scalebar(ax, extent, length_nm=SCALEBAR_NM, pad_frac=0.10,
                  color='white'):
    x0 = extent - pad_frac * 2 * extent - length_nm
    y0 = -extent + pad_frac * 2 * extent
    ax.plot([x0, x0 + length_nm], [y0, y0], color=color, lw=2.4,
            solid_capstyle='butt')


def draw_ring(ax, R, color=RING_COLOR, lw=1.0):
    if not R or not np.isfinite(R):
        return
    t = np.linspace(0, 2 * np.pi, 240)
    ax.plot(R * np.cos(t), R * np.sin(t), color=color, lw=lw, alpha=0.9)


def draw_sectors(ax, theta_deg, extent, fold=8, color=SECTOR_COLOR):
    """Radial spokes at the measured scaffold angle."""
    if theta_deg is None or not np.isfinite(theta_deg):
        return
    for k in range(int(fold)):
        a = np.radians(theta_deg + k * 360.0 / fold)
        ax.plot([0, np.cos(a) * extent], [0, np.sin(a) * extent],
                color=color, lw=0.5, alpha=0.55)


def style_axis(ax, extent):
    ax.set_aspect('equal')
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor('black')


def _new_axes(size=2.6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(size, size))
    return fig, ax


def _finish(fig, out, dpi=400):
    import matplotlib.pyplot as plt
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out, dpi=dpi, facecolor='black')
    plt.close(fig)
    return out


def save_single(xy, out, extent, pixel_nm, bw, ring_nm, cmap, scalebar_nm,
                gamma=1.0, floor=0.0, vmax=None, sector_angle_deg=None,
                fold=8, sector_color=SECTOR_COLOR, dpi=400):
    """One channel alone. floor=0 and gamma=1 show the raw distribution."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    Z = blur_image(xy[:, 0], xy[:, 1], extent, pixel_nm, bw)
    v = (np.percentile(Z[Z > 0], 99) if vmax is None else vmax) or 1.0
    t = np.clip(Z / v, 0, 1)
    if floor > 0:
        t = np.clip((t - floor) / (1 - floor), 0, 1)
    if gamma != 1.0:
        t = t ** gamma
    fig, ax = _new_axes()
    ax.imshow(cmap(t)[..., :3], origin='lower',
              extent=[-extent, extent, -extent, extent],
              interpolation='nearest')
    draw_ring(ax, ring_nm)
    if sector_angle_deg is not None:
        draw_sectors(ax, sector_angle_deg, extent, fold, sector_color)
    style_axis(ax, extent)
    draw_scalebar(ax, extent, length_nm=scalebar_nm)
    return _finish(fig, out, dpi)


def save_composite(ref, tgt, out, extent, pixel_nm, bw_ref, bw_tgt, ring_nm,
                   scalebar_nm, vmax=None, ref_floor=0.18, ref_gamma=1.0,
                   tgt_gamma=0.55, sector_angle_deg=None, fold=8,
                   sector_color=SECTOR_COLOR, dpi=400):
    """Reference (ICE) with the target (FIRE) composited on top."""
    rgb = two_color(ref, tgt, extent, pixel_nm, bw_ref, bw_tgt, vmax=vmax,
                    ref_floor=ref_floor, ref_gamma=ref_gamma,
                    tgt_gamma=tgt_gamma)
    fig, ax = _new_axes()
    ax.imshow(rgb, origin='lower', extent=[-extent, extent, -extent, extent],
              interpolation='nearest')
    draw_ring(ax, ring_nm)
    if sector_angle_deg is not None:
        draw_sectors(ax, sector_angle_deg, extent, fold, sector_color)
    style_axis(ax, extent)
    draw_scalebar(ax, extent, length_nm=scalebar_nm)
    return _finish(fig, out, dpi)


def save_points(ref, tgt, out, extent, pixel_nm, bw_ref, ring_nm, scalebar_nm,
                point_size=20.0, sector_angle_deg=None, fold=8,
                sector_color=SECTOR_COLOR, ref_floor=0.0, ref_gamma=1.0,
                vmax=None, dpi=400):
    """Reference rendered, target drawn as individual localizations."""
    ref = np.asarray(ref, float).reshape(-1, 2)
    tgt = np.asarray(tgt, float).reshape(-1, 2)
    fig, ax = _new_axes()
    Zb = blur_image(ref[:, 0], ref[:, 1], extent, pixel_nm, bw_ref)
    v = (np.percentile(Zb[Zb > 0], 99) if vmax is None else vmax) or 1.0
    tb = np.clip(Zb / v, 0, 1)
    tb = np.clip((tb - ref_floor) / (1 - ref_floor), 0, 1) ** ref_gamma
    ax.imshow(ICE()(tb)[..., :3], origin='lower',
              extent=[-extent, extent, -extent, extent],
              interpolation='nearest')
    if len(tgt):
        ax.scatter(tgt[:, 0], tgt[:, 1], s=point_size, c=RED,
                   edgecolor='white', linewidths=0.3, zorder=3)
    draw_ring(ax, ring_nm)
    if sector_angle_deg is not None:
        draw_sectors(ax, sector_angle_deg, extent, fold, sector_color)
    style_axis(ax, extent)
    draw_scalebar(ax, extent, length_nm=scalebar_nm)
    return _finish(fig, out, dpi)


def save_trace(ref, tx, ty, out, extent, pixel_nm, bw, ring_nm, scalebar_nm,
               labeled=False, ref_r=(30.0, 70.0), sector_angle_deg=None,
               fold=8, sector_color=SECTOR_COLOR, dpi=300):
    """One trace on its pore: reference behind, trace connected in time order."""
    ref = np.asarray(ref, float).reshape(-1, 2)
    if ref_r is not None and len(ref):
        rr = np.hypot(ref[:, 0], ref[:, 1])
        ref = ref[(rr >= ref_r[0]) & (rr <= ref_r[1])]

    fig, ax = _new_axes(size=2.2)
    Zn = blur_image(ref[:, 0], ref[:, 1], extent, pixel_nm, bw)
    show(ax, Zn, extent, ICE())

    Z = blur_image(tx, ty, extent, pixel_nm, bw)
    if Z.max() > 0:
        t = Z / Z.max()
        rgba = FIRE()(t)
        rgba[..., 3] = np.clip(t * 1.6, 0, 1)
        ax.imshow(rgba, origin='lower',
                  extent=[-extent, extent, -extent, extent],
                  interpolation='nearest', zorder=2)
    ax.plot(tx, ty, '-', color=FIRE_LINE, lw=0.9, alpha=0.9, zorder=3)
    ax.scatter(tx, ty, s=14, facecolor=FIRE_DOT, edgecolor='white',
               linewidths=0.4, zorder=4)
    if labeled:
        for i, (xx, yy) in enumerate(zip(tx, ty)):
            ax.annotate(str(i), (xx, yy), color='white', fontsize=5,
                        ha='center', va='center', zorder=5)
    draw_ring(ax, ring_nm)
    if sector_angle_deg is not None:
        draw_sectors(ax, sector_angle_deg, extent, fold, sector_color)
    style_axis(ax, extent)
    draw_scalebar(ax, extent, length_nm=scalebar_nm)
    return _finish(fig, out, dpi)


def save_merged(ref, tgt, out, extent, pixel_nm, bw_ref, bw_tgt, ring_nm,
                scalebar_nm, title=None, n_points=None, n_pores=None,
                mode='composite', tgt_gamma=0.55, ref_floor=0.10,
                fold=None, sector_color=SECTOR_COLOR, dpi=300):
    """A pooled panel with a caption carrying the counts it was built from."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ref = np.asarray(ref, float).reshape(-1, 2)
    tgt = np.asarray(tgt, float).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(3.2, 3.6))

    if mode == 'reference' or len(tgt) == 0:
        Z = blur_image(ref[:, 0], ref[:, 1], extent, pixel_nm, bw_ref)
        v = np.percentile(Z[Z > 0], 99) if (Z > 0).any() else 1.0
        ax.imshow(ICE()(np.clip(Z / (v or 1.0), 0, 1))[..., :3], origin='lower',
                  extent=[-extent, extent, -extent, extent],
                  interpolation='nearest')
    elif mode == 'target':
        Z = blur_image(tgt[:, 0], tgt[:, 1], extent, pixel_nm, bw_tgt)
        v = np.percentile(Z[Z > 0], 99) if (Z > 0).any() else 1.0
        t = np.clip(Z / (v or 1.0), 0, 1) ** tgt_gamma
        ax.imshow(FIRE()(t)[..., :3], origin='lower',
                  extent=[-extent, extent, -extent, extent],
                  interpolation='nearest')
    else:
        rgb = two_color(ref, tgt, extent, pixel_nm, bw_ref, bw_tgt,
                        ref_floor=ref_floor, tgt_gamma=tgt_gamma)
        ax.imshow(rgb, origin='lower',
                  extent=[-extent, extent, -extent, extent],
                  interpolation='nearest')

    draw_ring(ax, ring_nm)
    if fold:
        draw_sectors(ax, 0.0, extent, fold, sector_color)
    style_axis(ax, extent)
    draw_scalebar(ax, extent, length_nm=scalebar_nm)

    if title:
        ax.set_title(title, fontsize=8, color='white', pad=4)
    caption = []
    if n_points is not None:
        caption.append(f'n = {n_points:,} localizations')
    if n_pores is not None:
        caption.append(f'{n_pores:,} pores')
    if caption:
        ax.set_xlabel('   '.join(caption), fontsize=7.5, color='#c9d1d9',
                      labelpad=5)
    fig.patch.set_facecolor('black')
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=dpi, facecolor='black')
    plt.close(fig)
    return out


def radial_density(r, r_max, bin_nm):
    edges = np.arange(0, r_max + bin_nm, bin_nm)
    h, e = np.histogram(np.asarray(r, float), bins=edges)
    area = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    return 0.5 * (e[1:] + e[:-1]), h / area, h
