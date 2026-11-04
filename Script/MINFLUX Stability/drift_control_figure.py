# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np, pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
beads,rows=pickle.load(open('beads2.pkl','rb'))
sm=np.load('static_msd.npz'); tau=sm['tau']; M=sm['M']
rows2=sorted(range(len(rows)),key=lambda i:rows[i]['sx']+rows[i]['sy'])
rep=beads[rows2[0]]
cx,cy,xs,ys,zs,ts,seg,bt,bx,by,bz=rep
fig=plt.figure(figsize=(12,8))
# A: representative bead localizations, xy scatter (recentred)
ax=fig.add_subplot(2,3,1)
ax.scatter(xs-np.median(xs),ys-np.median(ys),s=1,alpha=.05,c='#1f4e79',edgecolors='none')
ax.set_xlim(-40,40);ax.set_ylim(-40,40);ax.set_aspect('equal')
ax.set_xlabel('x (nm)');ax.set_ylabel('y (nm)')
ax.set_title('A  Static bead localizations\n(%d loc, 1 kHz)'%len(xs),fontsize=10,loc='left')
th=np.linspace(0,2*np.pi,100)
for r in (11,22): ax.plot(r*np.cos(th),r*np.sin(th),'r-',lw=.6,alpha=.6)
# B: x,y,z of representative bead vs time (burst medians)
ax=fig.add_subplot(2,3,2)
for d,lab,c in [(bx-bx[0],'x','#c00'),(by-by[0],'y','#080'),(bz-bz[0],'z','#00c')]:
    ax.plot(bt/60,d,'-o',ms=3,lw=1,color=c,label=lab)
ax.set_xlabel('time (min)');ax.set_ylabel('drift (nm)');ax.set_ylim(-10,10)
ax.legend(fontsize=8,ncol=3);ax.set_title('B  Residual drift, one bead',fontsize=10,loc='left')
ax.axhspan(-2,2,color='gray',alpha=.12)
# C: drift traces for many beads
ax=fig.add_subplot(2,3,3)
for b in beads[:20]:
    _,_,_,_,_,_,_,bt2,bx2,by2,bz2=b
    r=np.hypot(np.hypot(bx2-bx2[0],by2-by2[0]),bz2-bz2[0])
    ax.plot(bt2/60,r,'-',lw=.7,alpha=.6)
ax.set_xlabel('time (min)');ax.set_ylabel('3D displacement (nm)')
ax.set_ylim(0,12);ax.set_title('C  Drift, 20 beads (~3 h)',fontsize=10,loc='left')
# D: precision histogram
ax=fig.add_subplot(2,3,4)
sx=[r['sx'] for r in rows];sy=[r['sy'] for r in rows];sz=[r['sz'] for r in rows]
ax.hist(sx,bins=np.arange(0,30,2),alpha=.6,label='x (med %.1f)'%np.median(sx))
ax.hist(sy,bins=np.arange(0,30,2),alpha=.6,label='y (med %.1f)'%np.median(sy))
ax.hist(sz,bins=np.arange(0,30,2),alpha=.6,label='z (med %.1f)'%np.median(sz))
ax.set_xlabel('localization precision σ (nm)');ax.set_ylabel('# beads')
ax.legend(fontsize=8);ax.set_title('D  Precision, %d beads'%len(rows),fontsize=10,loc='left')
# E: drift ptp histogram
ax=fig.add_subplot(2,3,5)
dp=[r['drift_ptp'] for r in rows]
ax.hist(dp,bins=np.arange(0,12,1),color='#555',alpha=.8)
ax.axvline(np.median(dp),color='r',lw=1)
ax.set_xlabel('peak-to-peak 3D drift over ~3 h (nm)');ax.set_ylabel('# beads')
ax.set_title('E  Drift, median %.1f nm'%np.median(dp),fontsize=10,loc='left')
# F: static MSD flat
ax=fig.add_subplot(2,3,6)
ax.plot(tau[1:]*1000,M[1:],'-o',ms=3,color='#1f4e79')
ax.axhline(4*np.median(sx)**2,color='r',ls='--',lw=1,label='4σ² noise floor')
ax.set_xlabel('lag τ (ms)');ax.set_ylabel('MSD (nm²)')
ax.set_ylim(0,700);ax.legend(fontsize=8)
ax.set_title('F  Static-bead MSD (flat)',fontsize=10,loc='left')
plt.tight_layout()
plt.savefig('MINFLUX_drift_control.png',dpi=150)
print('saved figure')
# also dump summary numbers
print('N beads',len(rows))
print('precision x/y/z median',np.median(sx),np.median(sy),np.median(sz))
print('drift ptp median/max',np.median(dp),max(dp))
print('MSD plateau',M[10],'sqrt',np.sqrt(M[10]))
