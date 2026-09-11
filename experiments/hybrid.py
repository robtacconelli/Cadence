"""Resolve the hybrid question properly (E26 was confounded by stream splitting).

Three closed loops per (series, rho), each producing ONE index stream so no
fragmentation is possible:
  A  tfm      -- TimesFM alone
  C  blend    -- pred = w*tfm + (1-w)*cls, w from causal inverse-error weighting
  D  leader   -- per block, use whichever predictor won the PREVIOUS block

Both C and D need ZERO side information: the decoder holds the same
reconstructed history, so it can recompute the classical prediction and both
predictors' past errors itself. That is strictly better than the 1-bit-per-block
switch we originally proposed.
The classical arm (B) is causal Lorenzo-2 so it can run inside a streaming loop;
SZ3's multilevel interpolation cannot, which is why it is not the partner here.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, glob, time, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))

CTX=512; H=64; N=int(os.environ.get("N",2048)); START=CTX; BLK=256
RHO=[0.01,0.05,0.2]; NSER=int(os.environ.get("NSER",16))
TD=CACHE
cand=sorted(os.path.basename(f)[:-4] for f in glob.glob("data/grid/*.npy"))
cand=[c for c in cand if c!="SEC"]
pri=[n for n in ["MISO","ERCO","PJM","CISO","BPAT","SOCO","DUK","AZPS"] if n in cand]
NAMES=(pri+[c for c in cand if c not in pri])[:NSER]
XF={n:np.load(f"data/grid/{n}.npy").astype(np.float64) for n in NAMES}

def cls_pred(h):
  """Causal Lorenzo-2 on reconstructed history (decoder can reproduce)."""
  return 2*h[-1]-h[-2]

if len(sys.argv)>1 and sys.argv[1]=="gpu":
  from timesfm3 import TimesFM3Forecaster, ModelConfig
  MODES=["tfm","blend","leader"]
  keys=[(n,r,m) for m in MODES for r in RHO for n in NAMES]
  B=len(keys); print(f"{NSER} series x {len(RHO)} rho x {len(MODES)} modes = {B} loops",flush=True)
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=B, device="cuda"))
  taus={k:max(k[1]*float(np.std(XF[k[0]])),0.5) for k in keys}
  xh={k:XF[k[0]].copy() for k in keys}
  Q={k:np.empty(N) for k in keys}
  # running abs errors per predictor, and per-block accumulators for 'leader'
  et={k:1.0 for k in keys}; ec={k:1.0 for k in keys}
  bt={k:0.0 for k in keys}; bc={k:0.0 for k in keys}; lead={k:"tfm" for k in keys}
  used={k:[] for k in keys}
  t0=time.time()
  for i in range(N):
    t=START+i
    P=list(fc.predict_batch(contexts=[xh[k][t-CTX:t].astype(np.float32) for k in keys],
           horizon=H,return_quantiles=False,sort_quantiles=True))
    for j,k in enumerate(keys):
      n,r,m=k; p_t=float(np.asarray(P[j].forecast)[0])
      p_c=cls_pred(xh[k][t-2:t])
      if m=="tfm":   p=p_t
      elif m=="blend":
        w=ec[k]/max(et[k]+ec[k],1e-9); p=w*p_t+(1-w)*p_c
      else:
        p=p_t if lead[k]=="tfm" else p_c
        used[k].append(1 if lead[k]=="tfm" else 0)
      D=2*taus[k]; q=np.rint((XF[n][t]-p)/D)
      xh[k][t]=p+q*D; Q[k][i]=q
      # causal error tracking (both predictors, on the SAME reconstructed history)
      at=abs(XF[n][t]-p_t); ac=abs(XF[n][t]-p_c)
      et[k]+= (at-et[k])/256; ec[k]+= (ac-ec[k])/256
      bt[k]+=at; bc[k]+=ac
      if (i+1)%BLK==0:
        lead[k]="tfm" if bt[k]<=bc[k] else "cls"; bt[k]=bc[k]=0.0
    if i%400==0: print(f"  {i}/{N} {time.time()-t0:.0f}s "
                       f"eta {(time.time()-t0)/(i+1)*(N-i-1)/60:.1f}min",flush=True)
  np.savez_compressed(f"{TD}/hybrid_N{N}.npz",
    names=np.array(NAMES),
    **{f"q|{n}|{r}|{m}":Q[(n,r,m)] for n,r,m in keys},
    **{f"tau|{n}|{r}":taus[(n,r,"tfm")] for r in RHO for n in NAMES},
    **{f"err|{n}|{r}|{m}":np.abs(xh[(n,r,m)][START:START+N]-XF[n][START:START+N]).max()
       for n,r,m in keys},
    **{f"use|{n}|{r}":np.array(used[(n,r,"leader")]) for r in RHO for n in NAMES})
  print(f"done {time.time()-t0:.0f}s",flush=True); sys.exit()

# ---------------- scoring ----------------
import lzma, zstandard as zstd
from sz_style import lorenzo, interp_multilevel
def nb(q):
  q=np.asarray(q,dtype=np.int64); u=q-q.min(); s=int(u.max())
  for w,dt in ((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64)):
    if s<2**(8*w): buf=u.astype(dt).tobytes(); break
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return (min(len(lzma.compress(buf,preset=9|lzma.PRESET_EXTREME)),len(zc.compress(buf)))+24)*8.0/len(q)
d=np.load(f"{TD}/hybrid_N{N}.npz",allow_pickle=True)
names=[str(x) for x in d["names"]]; res={}
print("HYBRID (single index stream per variant; C and D use ZERO side information)")
print(f"{'rho':>6s}{'TimesFM':>10s}{'Lorenzo2':>10s}{'interpCub':>11s}{'blend':>9s}"
      f"{'leader':>9s}{'best gain':>11s}{'tfm blocks':>12s}")
for r in RHO:
  a=[];b=[];ic=[];c=[];e=[];u=[]
  for n in names:
    x=XF[n][START:START+N]; tau=float(d[f"tau|{n}|{r}"])
    for m,acc in (("tfm",a),("blend",c),("leader",e)):
      assert float(d[f"err|{n}|{r}|{m}"])<=tau+1e-6
      acc.append(nb(d[f"q|{n}|{r}|{m}"]))
    b.append(nb(lorenzo(x,tau,2)[0])); ic.append(nb(interp_multilevel(x,tau,"cubic")[0]))
    u.append(d[f"use|{n}|{r}"].mean())
  base=min(np.mean(b),np.mean(ic),np.mean(a))
  best=min(np.mean(c),np.mean(e))
  res[str(r)]=dict(tfm=np.mean(a),lor2=np.mean(b),interp=np.mean(ic),
                   blend=np.mean(c),leader=np.mean(e),gain=100*(1-best/base),tfm_frac=np.mean(u))
  print(f"{r:6.2f}{np.mean(a):10.3f}{np.mean(b):10.3f}{np.mean(ic):11.3f}{np.mean(c):9.3f}"
        f"{np.mean(e):9.3f}{100*(1-best/base):+10.1f}%{100*np.mean(u):11.0f}%")
json.dump(res,open("results/hybrid.json","w"),indent=1,default=float)
