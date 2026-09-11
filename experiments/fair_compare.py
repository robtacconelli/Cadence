"""Apples-to-apples: every predictor's residual goes through the SAME
adaptive mixture-of-scales coder. Differences = prediction quality only."""
import numpy as np, glob, os, sys, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from fair import adaptive_resid_bits, pmf_adaptive
ESC=1e-4

def lpc_pred(x, order, pos):
  x=np.asarray(x,dtype=np.float64)
  X=np.lib.stride_tricks.sliding_window_view(x,order)[:-1]
  c,*_=np.linalg.lstsq(X,x[order:],rcond=None); c=np.rint(c*16384)/16384
  return np.rint(X@c)[pos-order]

def seasonal_pred(x, lags, pos):
  x=np.asarray(x,dtype=np.float64); L=max(lags)
  idx=np.arange(L,len(x))
  F=np.column_stack([np.ones(len(idx))]+[x[idx-k] for k in lags])
  c,*_=np.linalg.lstsq(F,x[idx],rcond=None)
  Fp=np.column_stack([np.ones(len(pos))]+[x[pos-k] for k in lags])
  return np.rint(Fp@c)

SEAS=list(range(1,49))+[71,72,73,95,96,97,119,120,121,167,168,169,335,336,337,503,504,505,671,672,673]
rows={}
for f in sorted(glob.glob("cache/*_c1024.npz")):
  name=os.path.basename(f).replace("_c1024.npz","")
  d=np.load(f); Q,V,pos,x=d["Q"],d["V"],d["pos"],d["x"]
  span=float(x.max()-x.min()+1)
  med=Q[:,4]
  cands={
    "tfm":        V-med,
    "lpc32":      V-lpc_pred(x,32,pos),
    "lpc256":     V-lpc_pred(x,256,pos) if len(x)>3000 else None,
    "seasonal":   V-seasonal_pred(x,SEAS,pos) if len(x)>1500 else None,
  }
  # stacked: model median + last 16 samples, LS-fit (coeffs in header)
  F=np.column_stack([np.ones(len(V)),med]+[x[pos-k] for k in range(1,17)])
  c,*_=np.linalg.lstsq(F,V,rcond=None); cands["stack"]=V-np.rint(F@c)
  F0=np.column_stack([np.ones(len(V))]+[x[pos-k] for k in range(1,17)])
  c0,*_=np.linalg.lstsq(F0,V,rcond=None); cands["stack_nomodel"]=V-np.rint(F0@c0)
  out={}
  for k,r in cands.items():
    if r is None: continue
    out[k]=adaptive_resid_bits(r,span)
    out["mae_"+k]=float(np.abs(r).mean())
  cl=min(v for k,v in out.items() if k in("lpc32","lpc256","seasonal","stack_nomodel"))
  bestm=min(out["tfm"],out.get("stack",1e9))
  out["_classical_fair"]=cl; out["_model_fair"]=bestm
  out["_gain_pct"]=100*(1-bestm/cl)
  rows[name]=out
  print(f"{name:22s} tfm={out['tfm']:7.3f} stack={out.get('stack',float('nan')):7.3f} | "
        f"lpc32={out['lpc32']:7.3f} lpc256={out.get('lpc256',float('nan')):7.3f} "
        f"seas={out.get('seasonal',float('nan')):7.3f} nomodel={out['stack_nomodel']:7.3f} "
        f"|| gain={out['_gain_pct']:+6.2f}%  MAE tfm/lpc32={out['mae_tfm']/max(out['mae_lpc32'],1e-9):5.2f}")
json.dump(rows,open("results/fair_compare.json","w"),indent=1)
