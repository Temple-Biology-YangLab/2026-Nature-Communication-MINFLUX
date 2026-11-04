# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Binned Gaussian KDE of XY slices for _boot_core."""
import numpy as np
from scipy.ndimage import gaussian_filter
import _boot_core as bc
def fast_kde_slices(pts,cfg):
    bw=bc.estimate_global_kde_bandwidth(pts,cfg.k_band,cfg.min_bandwidth_nm)
    L=121
    V=np.zeros((L,L,L))
    p=np.clip(pts,0,120.0)
    i0=np.floor(p).astype(int); f=p-i0; i1=np.minimum(i0+1,120)
    wx=[1-f[:,0],f[:,0]]; wy=[1-f[:,1],f[:,1]]; wz=[1-f[:,2],f[:,2]]
    ix=[i0[:,0],i1[:,0]]; iy=[i0[:,1],i1[:,1]]; iz=[i0[:,2],i1[:,2]]
    for a in range(2):
      for b in range(2):
        for c in range(2):
          np.add.at(V,(ix[a],iy[b],iz[c]),wx[a]*wy[b]*wz[c])
    V=gaussian_filter(V,sigma=bw/cfg.vox_nm,mode='constant')
    xy=bc.values_for_zoom_grid(cfg).astype(int)
    proj={}; gmax=0.0
    for zc in cfg.z_centers_nm:
        mat=V[np.ix_(xy,xy,[int(round(zc))])][:,:,0].T
        proj[float(zc)]=mat; gmax=max(gmax,float(np.nanmax(mat)))
    if gmax<=0: gmax=1.0
    return proj,bw,gmax
