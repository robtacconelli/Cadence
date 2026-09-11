"""Electricity grid load: the uncontaminated test of the 'aggregate human-demand'
hypothesis. EIA-930, 50 US balancing authorities, hourly, Jan-Jun 2026 -- well
after any plausible TimesFM training cutoff.

Throughput: every (series, tolerance) pair is an INDEPENDENT closed loop, so all
150 of them batch in lockstep -- one forward pass advances all 150 by one step.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, json, time, glob, lzma, subprocess
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import zstandard as zstd
from sz_style import PREDS

CTX=512; H=64; N=int(os.environ.get("N",2048)); START=CTX
RHO=[0.01,0.05,0.2]
TD=CACHE
SZ3="ext/sz3-install/bin/sz3"; env=dict(os.environ,LD_LIBRARY_PATH=os.path.abspath("ext/sz3-install/lib"))
SETS=sorted(os.path.basename(f)[:-4] for f in glob.glob("data/grid/*.npy"))
XF={n:np.load(f"data/grid/{n}.npy").astype(np.float64) for n in SETS}
SETS=[n for n in SETS if len(XF[n])>=START+N+H]

def rb(q):
  q=np.asarray(q,dtype=np.int64); u=q-q.min(); s=int(u.max())
  for w,dt in ((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64)):
    if s<2**(8*w): buf=u.astype(dt).tobytes(); break
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return (min(len(lzma.compress(buf,preset=9|lzma.PRESET_EXTREME)),len(zc.compress(buf)))+16)*8.0/len(q)

def sz3_bpv(x,tau):
  f=f"{TD}/gb.bin"; x.astype(np.float64).tofile(f)
  r=subprocess.run([SZ3,"-d","-i",f,"-z",f+".sz","-1",str(len(x)),"-M","ABS","-A",str(tau)],
                   capture_output=True,env=env)
  return os.path.getsize(f+".sz")*8.0/len(x) if r.returncode==0 else float("nan")

if len(sys.argv)>1 and sys.argv[1]=="gpu":
  from timesfm3 import TimesFM3Forecaster, ModelConfig
  pairs=[(n,r) for r in RHO for n in SETS]
  B=len(pairs); print(f"{len(SETS)} series x {len(RHO)} tolerances = {B} lockstep closed loops")
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=B,device="cuda"))
  taus={(n,r):max(r*float(np.std(XF[n])),0.5) for n,r in pairs}
  xh={p:XF[p[0]].copy() for p in pairs}; Q={p:[] for p in pairs}
  t0=time.time()
  for i in range(N):
    t=START+i
    outs=list(fc.predict_batch(contexts=[xh[p][t-CTX:t].astype(np.float32) for p in pairs],
              horizon=H,return_quantiles=False,sort_quantiles=True))
    for j,p in enumerate(pairs):
      pr=float(np.asarray(outs[j].forecast)[0]); D=2*taus[p]
      k=np.rint((XF[p[0]][t]-pr)/D); xh[p][t]=pr+k*D; Q[p].append(k)
    if i%200==0:
      el=time.time()-t0
      print(f"  {i}/{N} {el:.0f}s  {(i+1)*B/max(el,1e-9):.0f} val/s  eta {el/(i+1)*(N-i-1)/60:.1f}min",flush=True)
  np.savez(f"{TD}/gridq_N{N}.npz",
           **{f"{n}|{r}":np.array(Q[(n,r)]) for n,r in pairs},
           **{f"err|{n}|{r}":np.abs(xh[(n,r)][START:START+N]-XF[n][START:START+N]).max() for n,r in pairs})
  print(f"done {time.time()-t0:.0f}s  ({N*B/(time.time()-t0):.0f} val/s aggregate)")
  sys.exit()

d=np.load(f"{TD}/gridq_N{N}.npz"); res={}
print(f"ELECTRICITY GRID LOAD (EIA-930, 2026 Jan-Jun, {len(SETS)} balancing authorities)")
print("real bytes, vs best-of-6 classical error-bounded predictors\n")
for rho in RHO:
  G=[]
  for n in SETS:
    x=XF[n][START:START+N]; tau=max(rho*float(np.std(XF[n])),0.5)
    cl=min(rb(f(x,tau)[0]) for f in PREDS.values())
    t=rb(np.asarray(d[f"{n}|{rho}"])); assert float(d[f"err|{n}|{rho}"])<=tau+1e-6
    g=100*(1-t/cl); G.append(g); res[f"{n}|{rho}"]=dict(classic=cl,tfm=t,gain=g,tau=tau)
  G=np.array(G)
  print(f"rho={rho:<5}  median {np.median(G):+6.1f}%   mean {np.mean(G):+6.1f}%   "
        f"wins {int((G>0).sum())}/{len(G)}   p25 {np.percentile(G,25):+.1f}  p75 {np.percentile(G,75):+.1f}")
allg=np.array([v["gain"] for v in res.values()])
print(f"\nALL: median {np.median(allg):+.1f}%  mean {np.mean(allg):+.1f}%  wins {int((allg>0).sum())}/{len(allg)}")
top=sorted(((v["gain"],k) for k,v in res.items()),reverse=True)
print("\nbest 6:"); [print(f"  {k:16s}{g:+6.1f}%") for g,k in top[:6]]
print("worst 6:"); [print(f"  {k:16s}{g:+6.1f}%") for g,k in top[-6:]]
json.dump(res,open("results/grid_bench.json","w"),indent=1,default=float)
