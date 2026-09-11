"""Re-score the domain benchmarks with the arithmetic coder on BOTH sides.

Upgrading only Cadence's back end would be cheating: the AC helps whichever
residual stream it codes. Every predictor here -- Cadence and all six classical
ones -- is coded with the identical adaptive AC, so the comparison stays a
comparison of predictors.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, glob, json, time
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import rangecoder as rc
from sz_style import PREDS

TD=CACHE
N=2048; START=512; RHO=[0.01,0.05,0.2]
def ac(q): return (len(rc.encode(np.asarray(q,dtype=np.int64)))+24)*8.0/len(q)

CFG=[("grid","data/grid",f"{TD}/gridq_N2048.npz","results/grid_bench.json"),
     ("transit","data/transit",f"{TD}/transitq_N2048.npz","results/transit_bench.json")]
out={}
for dom,dd,qf,rf in CFG:
  d=np.load(qf); old=json.load(open(rf))
  sets=sorted({k.split("|")[0] for k in old} - {"SEC"})
  print(f"\n=== {dom}: {len(sets)} series, AC back end on both sides ===")
  print(f"{'rho':>6s}{'classic':>9s}{'Cadence':>9s}{'gain':>8s}{'wins':>9s}"
        f"{'(was, xz back end)':>22s}")
  for r in RHO:
    G=[];CL=[];CA=[]
    for n in sets:
      xf=np.load(f"{dd}/{n}.npy").astype(float); x=xf[START:START+N]
      tau=max(r*float(np.std(xf)),0.5)
      cl=min(ac(f(x,tau)[0]) for f in PREDS.values())
      ca=ac(d[f"{n}|{r}"])
      CL.append(cl); CA.append(ca); G.append(100*(1-ca/cl))
    G=np.array(G)
    oldg=np.median([v["gain"] for k,v in old.items()
                    if k.endswith(f"|{r}") and not k.startswith("SEC|")])
    out[f"{dom}|{r}"]=dict(classic=float(np.mean(CL)),cadence=float(np.mean(CA)),
                           median=float(np.median(G)),wins=int((G>0).sum()),n=len(G),
                           old_median=float(oldg))
    print(f"{r:6.2f}{np.mean(CL):9.3f}{np.mean(CA):9.3f}{np.median(G):+7.1f}%"
          f"{str(int((G>0).sum()))+'/'+str(len(G)):>9s}{oldg:+21.1f}%")
allg=[]; 
for dom,_,_,_ in CFG:
  g=[out[f"{dom}|{r}"] for r in RHO]
  tot=sum(x["wins"] for x in g); n=sum(x["n"] for x in g)
  m=float(np.median([x["median"] for x in g]))
  print(f"\n{dom:8s} AC back end: median {m:+.1f}%  wins {tot}/{n}")
  allg.append((m,tot,n))
print(f"\nCOMBINED demand domains: wins {sum(a[1] for a in allg)}/{sum(a[2] for a in allg)}")
json.dump(out,open("results/rescore_ac.json","w"),indent=1)
