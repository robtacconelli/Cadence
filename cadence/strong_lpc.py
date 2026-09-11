"""Honest classical bar: seasonal-aware linear prediction.
Plain LPC(32) cannot see a 24h or 168h cycle. If we are going to claim a
foundation model beats 'classical', classical must be allowed to see them."""
import numpy as np, sys
sys.path.insert(0,"src")
from tfm_entropy2 import running_mean
from tfm_entropy import p_laplace
ESC=1e-4

def lag_lpc_bits(x, lags, adapt=256, ret_pred=False):
  x=np.asarray(x,dtype=np.float64); span=float(x.max()-x.min()+1)
  L=max(lags); idx=np.arange(L,len(x))
  F=np.column_stack([np.ones(len(idx))]+[x[idx-k] for k in lags])
  y=x[idx]
  c,*_=np.linalg.lstsq(F,y,rcond=None)
  pred=np.rint(F@c); r=y-pred
  s=running_mean(np.abs(r),adapt)
  p=(1-ESC)*p_laplace(pred,s,y)+ESC/span
  bits=(-np.log2(np.maximum(p,1e-300))).sum()+32.0*(len(lags)+1)
  if ret_pred: return bits/len(idx), pred, idx
  return bits/len(idx)

if __name__=="__main__":
  import glob,os,json
  SETS={
    "lpc32":       list(range(1,33)),
    "lpc168":      list(range(1,169)),
    "lpc336":      list(range(1,337)),
    "seasonal24":  list(range(1,25))+[47,48,49,71,72,73,167,168,169,335,336,337],
    "seasonal_big":list(range(1,49))+[71,72,73,95,96,97,119,120,121,143,144,145,
                                      166,167,168,169,170,334,335,336,337,338,504,505,506,672,673,674],
  }
  out={}
  for f in sorted(glob.glob("data/real/*.npy"))+sorted(glob.glob("data/synth/*.npy")):
    n=os.path.basename(f)[:-4]; a=np.load(f)
    if len(a)<3000: continue
    r={k:lag_lpc_bits(a,v) for k,v in SETS.items()}
    out[n]=r
    best=min(r,key=r.get)
    print(f"{n:20s} "+" ".join(f"{k}={v:7.3f}" for k,v in r.items())+f"   BEST={best}")
  json.dump(out,open("results/strong_lpc.json","w"),indent=1)
