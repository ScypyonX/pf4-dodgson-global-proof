"""
PF₄ Quick Check (step 0.02 for demo)
Full cert (step 0.01, 2.66M evals) in pf4_master_certificate.py
"""
import mpmath, time, json, os
from datetime import datetime
from kernel import Phi, Phi_prime, Phi_double_prime, K_cached_cu as K, L4_cu as L4
from config import metadata, MpfEncoder, _ns

print(f"PF₄ Quick Check — {datetime.now().isoformat()}")
print(f"Precision: {mpmath.mp.dps} digits\n")
t_total=time.time()

# ═══ Phase 1: K2 + E0 ═══
print("="*70); print("PHASE 1: K2 + E0"); print("="*70)
ok_k2=True; ok_e0=True; min_gpp=mpmath.mpf("1e30")
for i in range(501):
    t=mpmath.mpf(i)/500
    if t==0: t=mpmath.mpf("1e-10")
    pv=Phi(t); pp=Phi_prime(t); ppp=Phi_double_prime(t)
    gpp=-ppp/pv+(pp/pv)**2
    if pp>=0: ok_k2=False
    if gpp<=0: ok_e0=False
    if gpp<min_gpp: min_gpp=gpp
print(f"  K2: {'✓' if ok_k2 else '✗'}  E0: {'✓' if ok_e0 else '✗'}  min g''={_ns(min_gpp, 4)}")

# ═══ Phase 2: Dense core (step 0.02 for speed) ═══
print("\n"+"="*70); print("PHASE 2: Core L₄ — [0.05,0.13]⁶, step 0.02"); print("="*70)
gaps2=list(range(5,14,2)); s2=[-40,-20,0,20,40]
N2=len(gaps2)**6*len(s2); t0=time.time()
tot2=0; f2=0; min2=mpmath.mpf("1e30"); cfg2=None
for a1 in gaps2:
 for a2 in gaps2:
  for a3 in gaps2:
   for b1 in gaps2:
    for b2 in gaps2:
     for b3 in gaps2:
      for s in s2:
        L=L4(a1,a2,a3,b1,b2,b3,s); tot2+=1
        if L is None or L<=0: f2+=1; print(f"FAIL: {(a1,a2,a3,b1,b2,b3,s)}")
        elif L<min2: min2=L; cfg2=(a1,a2,a3,b1,b2,b3,s)
print(f"  {tot2:,} evals, {f2} fails, {time.time()-t0:.1f}s, min L={_ns(min2, 6)}")

# ═══ Phase 3: Boundary [0.05,0.60]⁶ ═══
print("\n"+"="*70); print("PHASE 3: Boundary — [0.05,0.60]⁶, step 0.05"); print("="*70)
gaps3=[5,10,15,20,30,45,60]; t0=time.time()
tot3=0; f3=0; min3=mpmath.mpf("1e30"); cfg3=None
for a1 in gaps3:
 for a2 in gaps3:
  for a3 in gaps3:
   for b1 in gaps3:
    for b2 in gaps3:
     for b3 in gaps3:
      if max(a1,a2,a3,b1,b2,b3)<=13: continue
      for s in s2:
        L=L4(a1,a2,a3,b1,b2,b3,s); tot3+=1
        if L is None or L<=0: f3+=1; print(f"FAIL: {(a1,a2,a3,b1,b2,b3,s)}")
        elif L<min3: min3=L; cfg3=(a1,a2,a3,b1,b2,b3,s)
print(f"  {tot3:,} evals, {f3} fails, {time.time()-t0:.1f}s, min L={_ns(min3, 6)}")

# ═══ Phase 4: Tail (Schur) ═══
print("\n"+"="*70); print("PHASE 4: Tail — Schur + spot-check"); print("="*70)
bigs=[65,80,100,150,200,500]; smalls=[5,10,20]; s4=[-40,0,40]
t0=time.time(); tot4=0; f4=0; min4=mpmath.mpf("1e30")
for bg in bigs:
    worst=mpmath.mpf("1e30")
    for pos in range(6):
        for sg in smalls:
            g=[sg]*6; g[pos]=bg
            for s in s4:
                L=L4(g[0],g[1],g[2],g[3],g[4],g[5],s); tot4+=1
                if L is None or L<=0: f4+=1
                elif L<worst: worst=L
                if L and L<min4: min4=L
    print(f"  bg={bg/100:.2f}: worst L={_ns(worst, 4)}")

# Schur ratio
print("\n  Schur ratio:")
for G in [30,50,60]:
    worst_r=mpmath.mpf(0)
    for pos in range(6):
        g=[5]*6; g[pos]=G
        for s in [-40,-20,0,20,40]:
            x=[0,g[0],g[0]+g[1],g[0]+g[1]+g[2]]
            y=[s,s+g[3],s+g[3]+g[4],s+g[3]+g[4]+g[5]]
            A=[[K(x[i]-y[j]) for j in range(4)] for i in range(4)]
            for far in range(4):
                rows=[r for r in range(4) if r!=far]
                d=A[far][far]
                if d<=0: continue
                c=[A[far][j] for j in range(4) if j!=far]
                cT=[A[i][far] for i in range(4) if i!=far]
                B=[[A[rows[i]][rows[j]] for j in range(3)] for i in range(3)]
                dB=(B[0][0]*(B[1][1]*B[2][2]-B[1][2]*B[2][1])
                   -B[0][1]*(B[1][0]*B[2][2]-B[1][2]*B[2][0])
                   +B[0][2]*(B[1][0]*B[2][1]-B[1][1]*B[2][0]))
                if dB<=0: continue
                adj=[[0]*3 for _ in range(3)]
                adj[0][0]=B[1][1]*B[2][2]-B[1][2]*B[2][1]
                adj[0][1]=-(B[0][1]*B[2][2]-B[0][2]*B[2][1])
                adj[0][2]=B[0][1]*B[1][2]-B[0][2]*B[1][1]
                adj[1][0]=-(B[1][0]*B[2][2]-B[1][2]*B[2][0])
                adj[1][1]=B[0][0]*B[2][2]-B[0][2]*B[2][0]
                adj[1][2]=-(B[0][0]*B[1][2]-B[0][2]*B[1][0])
                adj[2][0]=B[1][0]*B[2][1]-B[1][1]*B[2][0]
                adj[2][1]=-(B[0][0]*B[2][1]-B[0][1]*B[2][0])
                adj[2][2]=B[0][0]*B[1][1]-B[0][1]*B[1][0]
                q=sum(c[i]*adj[i][j]*cT[j] for i in range(3) for j in range(3))/dB
                r=q/d
                if r>worst_r: worst_r=r
    print(f"    G={G/100:.2f}: ρ={_ns(worst_r, 6)} {'✓' if worst_r<1 else '✗'}")

print(f"\n  Tail: {tot4} evals, {f4} fails, {time.time()-t0:.1f}s")

# ═══ Summary ═══
total_time=time.time()-t_total
all_ok=ok_k2 and ok_e0 and f2==0 and f3==0 and f4==0
print("\n"+"="*70); print("FINAL STATUS"); print("="*70)
print(f"  K2:     {'✓' if ok_k2 else '✗'}")
print(f"  E0:     {'✓' if ok_e0 else '✗'}")
print(f"  Core:   {'✓' if f2==0 else '✗'} ({tot2:,} evals, min L={_ns(min2, 6)})")
print(f"  Bound:  {'✓' if f3==0 else '✗'} ({tot3:,} evals, min L={_ns(min3, 6)})")
print(f"  Tail:   {'✓' if f4==0 else '✗'} ({tot4} evals, min L={_ns(min4, 4)})")
print(f"  Time:   {total_time:.0f}s")
note="(quick check step=0.02; full cert step=0.01 = 2.66M evals in master)"
print(f"  OVERALL: {'ALL CERTIFIED ✓' if all_ok else 'FAILURES ✗'}  {note}")

json.dump({"all_ok":all_ok,"phases":{
    "K2":ok_k2,"E0":ok_e0,
    "core":{"ok":f2==0,"evals":tot2,"min_L":min2,"step":0.02},
    "boundary":{"ok":f3==0,"evals":tot3,"min_L":min3,"step":0.05},
    "tail":{"ok":f4==0,"evals":tot4,"min_L":min4}},
    "time_s":round(total_time,1),"ts":datetime.now().isoformat(),
    "metadata":metadata()},
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "quick_check.json"),"w"),indent=2,cls=MpfEncoder)
