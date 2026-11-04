# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np, pickle
from scipy.ndimage import maximum_filter
x=np.load('cx.npy');y=np.load('cy.npy');z=np.load('cz.npy');t=np.load('ct.npy')
# downsample for peak finding + selection (precision estimate robust to this)
s=slice(None,None,3)
xd,yd,zd,td=x[s],y[s],z[s],t[s]
H,xe,ye=np.histogram2d(xd,yd,bins=[np.arange(-20000,20000,50)]*2)
mx=maximum_filter(H,size=8); peaks=(H==mx)&(H>1000)
pi,pj=np.where(peaks); px=xe[pi]+25; py=ye[pj]+25; ph=H[pi,pj]
keep=[]
for i in range(len(px)):
    d=np.hypot(px-px[i],py-py[i]); d[i]=1e9
    if d.min()>700: keep.append(i)
px,py,ph=px[keep],py[keep],ph[keep]
o=np.argsort(ph)[::-1][:25]; px,py=px[o],py[o]
print('isolated single-bead peaks used:',len(px))
beads=[];rows=[]
for cx,cy in zip(px,py):
    m=(np.abs(xd-cx)<150)&(np.abs(yd-cy)<150)
    if m.sum()<8000: continue
    xs=xd[m];ys=yd[m];zs=zd[m];ts=td[m]
    ordr=np.argsort(ts);xs,ys,zs,ts=xs[ordr],ys[ordr],zs[ordr],ts[ordr]
    gaps=np.where(np.diff(ts)>2)[0]; seg=[ss for ss in np.split(np.arange(len(ts)),gaps+1) if len(ss)>40]
    if len(seg)<3: continue
    sx=np.mean([xs[ss].std() for ss in seg]);sy=np.mean([ys[ss].std() for ss in seg]);sz=np.mean([zs[ss].std() for ss in seg])
    bt=np.array([ts[ss].mean() for ss in seg]);bx=np.array([np.median(xs[ss]) for ss in seg])
    by=np.array([np.median(ys[ss]) for ss in seg]);bz=np.array([np.median(zs[ss]) for ss in seg])
    rad=np.hypot(np.hypot(bx-bx[0],by-by[0]),bz-bz[0])
    rows.append(dict(cx=float(cx),cy=float(cy),n=int(m.sum()*3),nb=len(seg),sx=sx,sy=sy,sz=sz,
        drift_ptp=float(rad.max()),drift_rms=float(np.sqrt(bx.std()**2+by.std()**2+bz.std()**2)),dur=float((bt.max()-bt.min())/60)))
    beads.append((float(cx),float(cy),xs,ys,zs,ts,seg,bt,bx,by,bz))
rows2=sorted(rows,key=lambda r:r['sx']+r['sy'])
for r in rows2[:14]:
    print("(%6.0f,%6.0f) n~%6d bursts=%2d dur=%3.0fm | prec x/y/z=%4.1f/%4.1f/%4.1f | drift ptp=%5.1f rms=%4.1f nm"%(
        r['cx'],r['cy'],r['n'],r['nb'],r['dur'],r['sx'],r['sy'],r['sz'],r['drift_ptp'],r['drift_rms']))
pickle.dump((beads,rows),open('beads2.pkl','wb'))
clean=[r for r in rows if r['sx']<50 and r['sy']<50 and r['sz']<50]
print('\nCLEAN beads (x/y/z prec<50nm):',len(clean),'of',len(rows))
if clean:
    print('median prec x/y/z=%.1f/%.1f/%.1f nm'%(np.median([r['sx'] for r in clean]),np.median([r['sy'] for r in clean]),np.median([r['sz'] for r in clean])))
    print('median 3h drift ptp=%.1f nm, rms=%.1f nm; max ptp=%.1f nm'%(
        np.median([r['drift_ptp'] for r in clean]),np.median([r['drift_rms'] for r in clean]),max(r['drift_ptp'] for r in clean)))
