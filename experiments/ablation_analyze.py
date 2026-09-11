"""Analyse both ablations. Real bytes throughout."""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, glob, lzma, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import zstandard as zstd
from codec import SEED_ENC, _pack, _sq

TD=CACHE
N=int(os.environ.get("N",2048))
def nbytes(q):
  b,lo,code=_pack(np.asarray(q,dtype=np.int64)); return len(_sq(b))+24
def bpv(q,n=None): return nbytes(q)*8.0/(n or len(q))

# ---------- B: heteroscedastic index coding, decoder-derivable buckets ----------
def het_bytes(q, spread, D, nb=8):
  """Split the index stream by predicted-spread bucket and compress each
  separately. The decoder knows the quantiles BEFORE decoding k, so bucket
  assignment needs no side information."""
  sig=np.maximum(spread/max(D,1e-9),1e-6)
  edges=np.quantile(sig,np.linspace(0,1,nb+1)[1:-1])
  b=np.searchsorted(edges,sig)
  tot=0
  for i in range(nb):
    m=b==i
    if m.sum()==0: continue
    tot+=nbytes(q[m])
  return tot+8*nb                      # bucket edges in the header

def flat_bytes(q): return nbytes(q)

# --- ideal codelengths: isolates the MODELLING value from container effects ---
from fair import adaptive_resid_bits
def het_ideal_bits(q, spread, D, adapt=256):
  """Single stream, one adaptive coder, model conditioned on predicted spread.
  This is what a real arithmetic coder would do -- no fragmentation."""
  q=np.asarray(q,dtype=np.float64); sig=np.maximum(spread/max(D,1e-9),1e-3)
  aq=np.abs(q)
  # causal calibration of the gain: alpha_t = runmean|q| / runmean sigma
  ra=np.empty(len(q)); rs=np.empty(len(q))
  a=max(aq[:adapt].mean(),1e-3); b=max(sig[:adapt].mean(),1e-3)
  for i in range(len(q)):
    ra[i]=a; rs[i]=b
    a+=(aq[i]-a)/adapt; b+=(sig[i]-b)/adapt
  sc=np.maximum(ra/np.maximum(rs,1e-9)*sig,1e-3)
  hi=(aq+0.5)/sc; lo=np.maximum((aq-0.5)/sc,0.0)
  p=0.5*(np.exp(-lo)-np.exp(-hi))
  p=np.where(aq<0.5,1.0-np.exp(-hi),p)
  span=float(aq.max()*2+3)
  p=(1-1e-4)*np.clip(p,0,1)+1e-4/span
  return float((-np.log2(np.maximum(p,1e-300))).mean())
def flat_ideal_bits(q):
  q=np.asarray(q,dtype=np.float64)
  return adaptive_resid_bits(q, max(abs(q).max()*2+3,3))

if __name__=="__main__":
  files=sorted(glob.glob(f"{TD}/abl_ctx*_N{N}.npz"),
               key=lambda f:int(f.split("ctx")[1].split("_")[0]))
  if not files: print("no ablation files yet"); sys.exit()
  res={}
  print("=== ABLATION A: context length (body bits/value, real bytes) ===")
  print(f"{'ctx':>6s}"+"".join(f"{'rho='+str(r):>12s}" for r in [0.01,0.05,0.2])+f"{'mean':>9s}")
  bodies={}
  for f in files:
    ctx=int(f.split("ctx")[1].split("_")[0]); d=np.load(f,allow_pickle=True)
    names=[str(x) for x in d["names"]]; row=[]
    for r in [0.01,0.05,0.2]:
      v=[bpv(d[f"q|{n}|{r}"]) for n in names]
      assert all(float(d[f"err|{n}|{r}"])<=float(d[f"tau|{n}|{r}"])+1e-6 for n in names)
      row.append(np.mean(v))
    bodies[ctx]=row
    print(f"{ctx:6d}"+"".join(f"{x:12.3f}" for x in row)+f"{np.mean(row):9.3f}")
  base=bodies[max(bodies)]
  print(f"\n{'ctx':>6s}{'body vs ctx1024':>18s}{'seed samples':>14s}")
  for ctx in sorted(bodies):
    print(f"{ctx:6d}{100*(1-np.mean(bodies[ctx])/np.mean(base)):+17.1f}%{ctx:14d}")

  # end-to-end: body(ctx) over n-ctx samples + classical seed over ctx samples
  print("\n=== END-TO-END by series length (rho=0.05) ===")
  d0=np.load(files[0],allow_pickle=True); names=[str(x) for x in d0["names"]]
  seedcost={}
  for ctx in sorted(bodies):
    tot=0
    for n in names:
      x=np.load(f"data/grid/{n}.npy").astype(float)
      tau=max(0.05*float(np.std(x)),0.5)
      best=min(nbytes(e(x[:ctx],tau)[0]) for e in SEED_ENC)
      tot+=best*8.0/ctx
    seedcost[ctx]=tot/len(names)
  CLS=2.420
  print(f"{'ctx':>6s}{'seed bpv':>10s}{'body bpv':>10s}"+"".join(f"{'n='+str(n):>11s}" for n in [4300,8760,17520,43800]))
  best_by_n={}
  for ctx in sorted(bodies):
    b=bodies[ctx][1]; row=[]
    for n in [4300,8760,17520,43800]:
      tot=(b*(n-ctx)+seedcost[ctx]*ctx)/n; row.append(100*(1-tot/CLS))
      best_by_n.setdefault(n,[]).append((row[-1],ctx))
    print(f"{ctx:6d}{seedcost[ctx]:10.3f}{b:10.3f}"+"".join(f"{v:+10.1f}%" for v in row))
  print("\noptimal ctx per length:", {n:max(v)[1] for n,v in best_by_n.items()},
        " best gain:", {n:round(max(v)[0],1) for n,v in best_by_n.items()})
  res["A_body"]=bodies; res["A_seed"]=seedcost

  print("\n=== ABLATION B: heteroscedastic index coding (uses the 9 quantiles) ===")
  LN5=np.log(5.0)
  print("  practical = split stream into 8 spread buckets (fragments xz)")
  print("  ideal     = ONE adaptive coder conditioned on predicted spread (no fragmentation)")
  print(f"{'ctx':>6s}{'rho':>7s}{'flat B':>10s}{'het B':>10s}{'gain':>8s}"
        f"{'flat ideal':>11s}{'het ideal':>10s}{'gain':>10s}")
  hb={}
  for f in files:
    ctx=int(f.split("ctx")[1].split("_")[0]); d=np.load(f,allow_pickle=True)
    names=[str(x) for x in d["names"]]
    for r in [0.01,0.05,0.2]:
      fl=[];he=[];fi=[];hi=[]
      for n in names:
        q=d[f"q|{n}|{r}"]; u=d[f"u|{n}|{r}"].astype(np.float64); tau=float(d[f"tau|{n}|{r}"])
        sp=(u[:,8]-u[:,0])/(2*LN5)
        fl.append(flat_bytes(q)*8.0/len(q)); he.append(het_bytes(q,sp,2*tau)*8.0/len(q))
        fi.append(flat_ideal_bits(q)); hi.append(het_ideal_bits(q,sp,2*tau))
      g=100*(1-np.mean(he)/np.mean(fl)); hb[(ctx,r)]=(np.mean(fl),np.mean(he),g)
      print(f"{ctx:6d}{r:7.3f}{np.mean(fl):10.3f}{np.mean(he):10.3f}{g:+8.1f}%"
            f"{np.mean(fi):11.3f}{np.mean(hi):10.3f}{100*(1-np.mean(hi)/np.mean(fi)):+9.1f}%")
      hb[(ctx,r)]=(np.mean(fl),np.mean(he),g,np.mean(fi),np.mean(hi),
                   100*(1-np.mean(hi)/np.mean(fi)))
  res["B_het"]={f"{k[0]}|{k[1]}":v for k,v in hb.items()}
  g=[v[2] for v in hb.values()]; gi=[v[5] for v in hb.values()]
  print(f"\npractical (bucketed) : median {np.median(g):+.1f}%  wins {sum(1 for x in g if x>0)}/{len(g)}")
  print(f"ideal (conditioned)  : median {np.median(gi):+.1f}%  wins {sum(1 for x in gi if x>0)}/{len(gi)}")
  json.dump(res,open("results/ablation.json","w"),indent=1,default=float)
