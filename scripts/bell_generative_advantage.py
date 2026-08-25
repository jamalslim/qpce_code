#!/usr/bin/env python3
"""Bell-certified conditional generation: the dataset-free advantage of QPCE.

A: classical bound, executable -- ALL deterministic local-response strategies
   (the LHV polytope vertices; complete for CHSH) are enumerated: max S = 2.
B: quantum generator -- per-germ CHSH capability S(eps) for every coupler pair,
   full model vs entangler-stripped model (phi=0: provably S<=2, verified).
C: finite-shot generative Bell experiment at representative germs.
Outputs: outputs/bell_generative.{json,png,pdf}
"""
import json, sys, time
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from qpce.quantum_features import InterferometricCircuit
t0=time.time()

# ---------- A: the classical generative bound, brute-forced ----------
best=-9; allS=[]
for aA in [(-1,-1),(-1,1),(1,-1),(1,1)]:          # A's response to s_A in {0,1}
    for aB in [(-1,-1),(-1,1),(1,-1),(1,1)]:      # B's response to s_B in {0,1}
        S=aA[0]*aB[0]+aA[0]*aB[1]+aA[1]*aB[0]-aA[1]*aB[1]
        allS.append(S); best=max(best,S)
print(f"[A] classical local-response strategies: {len(allS)} vertices, max S = {best}"
      "  (shared randomness = convex hull => bound is exact for ANY classical generator)")

# ---------- B: per-germ CHSH capability of the trained generator ----------
z=np.load('outputs/checkpoint/params.npz',allow_pickle=True)
def model(phi_on=True):
    m=InterferometricCircuit(8,6,[tuple(e) for e in z['edges']],walls='trainable',
                             dither=False,shared_germ=1)
    for k in ('beta','phi','theta','w'): setattr(m,k,z[k].astype(float))
    m.a=z['a'].astype(float)
    if not phi_on: m.phi=np.zeros_like(m.phi)
    return m
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1,-1]).astype(complex)
P3=[sx,sy,sz]
def pair_S(psi_flat,i,j,n=8):
    ax=[2]*n; ps=psi_flat.reshape(ax)
    order=[i,j]+[q for q in range(n) if q not in (i,j)]
    ps=np.transpose(ps,order).reshape(4,-1)
    r=ps@ps.conj().T
    T=np.array([[np.real(np.trace(r@np.kron(a,b))) for b in P3] for a in P3])
    w=np.linalg.eigvalsh(T.T@T)
    return 2*np.sqrt(max(w[-1]+w[-2],0))
RNG=np.random.default_rng(6); G=1000
E=RNG.uniform(0,1,(G,17))
res={'classical_max_S':best,'pairs':{}}
for tag,on in (('full',True),('stripped',False)):
    m=model(on); psi=m.statevector(E).reshape(G,-1)
    Sarr=np.zeros((7,G))
    for pidx,(i,j) in enumerate(m.edges):
        for g in range(G): Sarr[pidx,g]=pair_S(psi[g],i,j)
    res['pairs'][tag]={f"{i}-{j}":{'max':float(Sarr[p].max()),'mean':float(Sarr[p].mean()),
                                   'frac_violating':float((Sarr[p]>2+1e-9).mean())}
                       for p,(i,j) in enumerate(m.edges)}
    res[f'S_{tag}']=Sarr.tolist() if False else None
    np.save(f'/tmp/S_{tag}.npy',Sarr)
    print(f"[B] {tag:9s}: global max S = {Sarr.max():.3f}, overall violating fraction "
          f"{(Sarr>2).mean():.1%}   [{round(time.time()-t0)}s]")

Sf=np.load('/tmp/S_full.npy'); Ss=np.load('/tmp/S_stripped.npy')
assert Ss.max()<=2.0+1e-9, "strip control violated the theorem!"
print(f"[B] entangler-stripped control: max S = {Ss.max():.6f} <= 2 for ALL {G} germs x 7 pairs (theorem verified)")

# ---------- C: finite-shot experiments at 3 representative germs (pair 0-1) ----------
m=model(True); psi=m.statevector(E).reshape(G,-1)
def rdm(psi_flat,i,j):
    ps=psi_flat.reshape([2]*8)
    order=[i,j]+[q for q in range(8) if q not in (i,j)]
    return (p:=np.transpose(ps,order).reshape(4,-1))@p.conj().T
def horodecki(T):
    U,S,Vt=np.linalg.svd(T); c=S[0]/np.sqrt(S[0]**2+S[1]**2); s=S[1]/np.sqrt(S[0]**2+S[1]**2)
    return (c*U[:,0]+s*U[:,1], c*U[:,0]-s*U[:,1], Vt[0], Vt[1])
order=np.argsort(Sf[0])[::-1]
shots=5000; rng=np.random.default_rng(17); Cres=[]
for g in order[[0, G//4, G//2]]:
    r=rdm(psi[g],0,1)
    T=np.array([[np.real(np.trace(r@np.kron(a,b))) for b in P3] for a in P3])
    rA=np.array([np.real(np.trace(r@np.kron(a,np.eye(2)))) for a in P3])
    rB=np.array([np.real(np.trace(r@np.kron(np.eye(2),b))) for b in P3])
    a1,a2,b1,b2=horodecki(T); Sh=0; var=0
    for (a,b,sgn) in ((a1,b1,1),(a1,b2,1),(a2,b1,1),(a2,b2,-1)):
        p=np.array([(1+oa*(a@rA)+ob*(b@rB)+oa*ob*(a@T@b))/4 for oa in(1,-1) for ob in(1,-1)])
        p=np.clip(p,0,None); p/=p.sum()
        cnt=rng.multinomial(shots,p); o=np.array([1,-1,-1,1])
        Eab=(cnt@o)/shots; Sh+=sgn*Eab; var+=(1-Eab**2)/shots
    Cres.append({'germ':int(g),'S_hat':float(Sh),'se':float(np.sqrt(var)),
                 'sigma':float((Sh-2)/np.sqrt(var))})
    print(f"[C] germ {g:4d}: finite-shot S = {Sh:.3f} +- {np.sqrt(var):.3f}  ({(Sh-2)/np.sqrt(var):+.1f} sigma vs classical bound)")
res['finite_shot']=Cres

# ---------- figure ----------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(13.5,3.8))
bins=np.linspace(1.2,2.9,52)
ax[0].hist(Sf[0],bins=bins,alpha=0.75,color='crimson',label='full model (pair 0-1)')
ax[0].hist(Ss[0],bins=bins,alpha=0.75,color='gray',label='entanglers stripped')
ax[0].axvline(2,color='k',ls='--',lw=1.2,label='classical bound (ALL generators)')
ax[0].axvline(2*np.sqrt(2),color='b',ls=':',lw=1.2,label='Tsirelson')
ax[0].set_xlabel('CHSH capability $S(\\varepsilon)$'); ax[0].set_ylabel('germs')
ax[0].set_title('per-condition Bell capability'); ax[0].legend(fontsize=7.5)
pairs=[f"{i}-{j}" for (i,j) in model(True).edges]
mx=[res['pairs']['full'][p]['max'] for p in pairs]
fr=[res['pairs']['full'][p]['frac_violating'] for p in pairs]
x=np.arange(7)
ax[1].bar(x-0.2,mx,0.4,color='crimson',label='max $S$')
ax[1].bar(x+0.2,[2+f for f in fr],0.4,color='steelblue',label='2 + frac violating')
ax[1].axhline(2,color='k',ls='--',lw=1.2); ax[1].axhline(2*np.sqrt(2),color='b',ls=':',lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels(pairs,fontsize=8)
ax[1].set_xlabel('coupler pair'); ax[1].set_title('capability across the path'); ax[1].legend(fontsize=8)
gg=[c['S_hat'] for c in Cres]; ee=[c['se'] for c in Cres]
ax[2].errorbar(range(len(gg)),gg,yerr=ee,fmt='o',color='crimson',capsize=4,
               label='finite-shot generative Bell test')
ax[2].axhline(2,color='k',ls='--',lw=1.2,label='classical bound')
ax[2].axhline(2*np.sqrt(2),color='b',ls=':',lw=1)
ax[2].set_xticks(range(len(gg))); ax[2].set_xticklabels([f"germ {c['germ']}" for c in Cres],fontsize=8)
ax[2].set_ylim(1.6,2.95); ax[2].set_title(f'{shots} shots/setting'); ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig('outputs/bell_generative.pdf'); fig.savefig('outputs/bell_generative.png',dpi=150)
json.dump(res,open('outputs/bell_generative.json','w'),indent=1)
print(f"saved outputs/bell_generative.*   [{round(time.time()-t0)}s]")
