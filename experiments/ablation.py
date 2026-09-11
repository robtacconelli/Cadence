"""Two ablations from one GPU pass.

A. CONTEXT LENGTH. The 512-sample bootstrap costs 4.008 bpv against a 2.126 bpv
   body, and takes years of hourly data to amortize (E42). If a shorter context
   holds most of the accuracy, that matters far more at realistic lengths than
   any modelling improvement. Sweep ctx in {64,128,256,512,1024}.

B. HETEROSCEDASTIC INDEX CODING. Every lossy run so far threw the 9 quantiles
   away and coded quantization indices with a flat adaptive coder. Save the
   quantiles here so the value of conditioning on predicted spread can be
   measured offline.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, glob, time, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))

N=int(os.environ.get("N",2048)); RHO=[0.01,0.05,0.2]
CTXS=[int(c) for c in os.environ.get("CTXS","1024,512,256,128,64").split(",")]
NSER=int(os.environ.get("NSER",16))
TD=CACHE
H=64
cand=sorted(os.path.basename(f)[:-4] for f in glob.glob("data/grid/*.npy"))
cand=[c for c in cand if c!="SEC"]
pri=[n for n in ["MISO","ERCO","PJM","CISO","BPAT","SOCO","DUK","AZPS"] if n in cand]
NAMES=(pri+[c for c in cand if c not in pri])[:NSER]
XF={n:np.load(f"data/grid/{n}.npy").astype(np.float64) for n in NAMES}

if len(sys.argv)>1 and sys.argv[1]=="gpu":
  from timesfm3 import TimesFM3Forecaster, ModelConfig
  pairs=[(n,r) for r in RHO for n in NAMES]; B=len(pairs)
  print(f"{NSER} series x {len(RHO)} rho = {B} loops; ctx sweep {CTXS}",flush=True)
  for ctx in CTXS:
    out=f"{TD}/abl_ctx{ctx}_N{N}.npz"
    if os.path.exists(out): print("skip",ctx); continue
    START=max(CTXS)                      # same scored window for every ctx -> comparable
    fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                      per_core_batch_size=B, device="cuda"))
    taus={p:max(p[1]*float(np.std(XF[p[0]])),0.5) for p in pairs}
    xh={p:XF[p[0]].copy() for p in pairs}
    Q={p:np.empty(N) for p in pairs}; QU={p:np.empty((N,9),dtype=np.float32) for p in pairs}
    t0=time.time()
    for i in range(N):
      t=START+i
      o=list(fc.predict_batch(contexts=[xh[p][t-ctx:t].astype(np.float32) for p in pairs],
             horizon=H,return_quantiles=True,sort_quantiles=True))
      for j,p in enumerate(pairs):
        qq=np.asarray(o[j].quantiles,dtype=np.float64).reshape(-1,9)[0]
        pr=float(qq[4]); D=2*taus[p]
        k=np.rint((XF[p[0]][t]-pr)/D); xh[p][t]=pr+k*D
        Q[p][i]=k; QU[p][i]=qq
      if i%400==0: print(f"  ctx={ctx} {i}/{N} {time.time()-t0:.0f}s "
                         f"eta {(time.time()-t0)/(i+1)*(N-i-1)/60:.1f}min",flush=True)
    np.savez_compressed(out, names=np.array(NAMES), rhos=np.array(RHO), start=START,
      **{f"q|{n}|{r}":Q[(n,r)] for n,r in pairs},
      **{f"u|{n}|{r}":QU[(n,r)] for n,r in pairs},
      **{f"tau|{n}|{r}":taus[(n,r)] for n,r in pairs},
      **{f"err|{n}|{r}":np.abs(xh[(n,r)][START:START+N]-XF[n][START:START+N]).max() for n,r in pairs})
    print(f"ctx={ctx} done {time.time()-t0:.0f}s -> {out}",flush=True)
    del fc
    import torch; torch.cuda.empty_cache()
  sys.exit()
print("run with 'gpu' to collect")
