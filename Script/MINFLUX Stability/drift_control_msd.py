# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np, pickle
beads,rows=pickle.load(open('beads2.pkl','rb'))
# static-bead MSD within continuous bursts, ensemble over clean beads
def burst_msd(xs,ys,ts,seg,maxlag=50):
    # use median dt to define lag in points; assume ~uniform within burst
    msd=np.zeros(maxlag); cnt=np.zeros(maxlag)
    for s in seg:
        X=xs[s];Y=ys[s]
        for L in range(1,maxlag):
            if len(X)>L:
                d=(X[L:]-X[:-L])**2+(Y[L:]-Y[:-L])**2
                msd[L]+=d.sum(); cnt[L]+=len(d)
    return msd/np.maximum(cnt,1)
# typical dt
b=beads[0]; ts=b[5]; dt=np.median(np.diff(ts[b[6][0]]))
print('median dt within burst (s):',dt)
# ensemble MSD over first 12 clean beads
lags=40
M=np.zeros(lags);n=0
for b in beads[:12]:
    cx,cy,xs,ys,zs,ts,seg,bt,bx,by,bz=b
    m=burst_msd(xs,ys,ts,seg,lags); M+=m; n+=1
M/=n
tau=np.arange(lags)*dt
for L in [1,5,10,20,39]:
    print('tau=%.3fs MSD=%.1f nm^2  sqrt=%.1f nm'%(tau[L],M[L],np.sqrt(M[L])))
np.savez('static_msd.npz',tau=tau,M=M,dt=dt)
