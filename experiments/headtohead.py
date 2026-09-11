"""Exact head-to-head on MATCHED ranges: TimesFM vs the full SZ3-class family."""
import numpy as np, json, sys
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from sz_style import PREDS, _bits
lo=json.load(open("results/lossy_parsed.json"))
SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
RHO=[0.001,0.01,0.05,0.2]; N=2048; START=1024          # exactly what lossy.py scored
out={}
print(f"{'dataset':17s}{'rho':>7s}{'TimesFM':>9s}{'lorenzo1':>9s}{'lpc32':>8s}{'interpLin':>10s}"
      f"{'interpCub':>10s}{'BEST-CLASSIC':>14s}{'gain%':>8s}")
print("-"*94)
gains={r:[] for r in RHO}
for n in SETS:
  xf=np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64)
  x=xf[START:START+N]; sd=float(np.std(xf))     # tau from full-series std, as in lossy.py
  for rho in RHO:
    tau=max(rho*sd,0.5); row={}
    for k,f in PREDS.items():
      q,xh=f(x,tau); assert np.abs(xh-x).max()<=tau+1e-6
      row[k]=_bits(q)
    bk=min(row,key=row.get); bv=row[bk]
    tfm=lo.get(f"{n}|{rho}|1",{}).get("tfm")
    g=100*(1-tfm/bv) if tfm else float("nan")
    gains[rho].append(g); out[f"{n}|{rho}"]=dict(tfm=tfm,classic=row,best=bk,bestv=bv,gain=g)
    print(f"{n:17s}{rho:7.3f}{tfm:9.3f}{row['lorenzo1']:9.3f}{row['lpc32']:8.3f}"
          f"{row['interp_linear']:10.3f}{row['interp_cubic']:10.3f}{bk:>14s}{g:+8.1f}")
  print()
print("-"*94)
for r in RHO:
  g=np.array(gains[r]); print(f"  rho={r:<7} median {np.median(g):+6.1f}%  mean {np.mean(g):+6.1f}%  "
                              f"wins {int((g>0).sum())}/{len(g)}")
json.dump(out,open("results/headtohead.json","w"),indent=1)
