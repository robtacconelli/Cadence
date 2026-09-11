"""Definitive benchmark. Fixes both flaws found above:
  1. N=8192 so SZ3's ~500-byte container overhead is amortized (it dominated at 2048)
  2. reports REAL BYTES for every method, not just ideal codelength -- my
     quantization indices go through actual xz/zstd, the way SZ3 uses Huffman+zstd
Stage 1 (GPU) writes quantization indices; stage 2 (CPU) scores them.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, json, time, lzma, subprocess
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import zstandard as zstd, zfpy
from fair import adaptive_resid_bits
from sz_style import PREDS

N=int(os.environ.get("N",8192)); START=1024; CTX=1024; H=64
SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
RHO=[float(r) for r in os.environ.get("RHO","0.01,0.05,0.2").split(",")]
TD=CACHE
SZ3="ext/sz3-install/bin/sz3"
env=dict(os.environ,LD_LIBRARY_PATH=os.path.abspath("ext/sz3-install/lib"))
XF={n:np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64) for n in SETS}

def real_bytes(q):
  """Actual bytes for quantization indices: natural width + xz/zstd, best of."""
  q=np.asarray(q,dtype=np.int64); lo=q.min(); u=(q-lo)
  span=int(u.max())
  for w,dt in ((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64)):
    if span < 2**(8*w): buf=u.astype(dt).tobytes(); break
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return min(len(lzma.compress(buf,preset=9|lzma.PRESET_EXTREME)), len(zc.compress(buf)))+16

def sz3_run(x,tau):
  f=f"{TD}/fb.bin"; x.astype(np.float64).tofile(f)
  r=subprocess.run([SZ3,"-d","-i",f,"-z",f+".sz","-o",f+".out","-1",str(len(x)),
                    "-M","ABS","-A",str(tau)],capture_output=True,env=env)
  if r.returncode!=0: return None,None
  y=np.fromfile(f+".out",dtype=np.float64)
  return os.path.getsize(f+".sz")*8.0/len(x), float(np.abs(y-x).max()/tau)

if sys.argv[1]=="gpu":
  from timesfm3 import TimesFM3Forecaster, ModelConfig
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=len(SETS),device="cuda"))
  for rho in RHO:
    taus={n:max(rho*float(np.std(XF[n])),0.5) for n in SETS}
    xh={n:XF[n].copy() for n in SETS}; Q={n:[] for n in SETS}
    t0=time.time()
    for i in range(N):
      t=START+i
      outs=list(fc.predict_batch(contexts=[xh[n][t-CTX:t].astype(np.float32) for n in SETS],
                horizon=H,return_quantiles=False,sort_quantiles=True))
      for j,n in enumerate(SETS):
        p=float(np.asarray(outs[j].forecast)[0]); D=2*taus[n]
        k=np.rint((XF[n][t]-p)/D); xh[n][t]=p+k*D; Q[n].append(k)
      if i%1000==0: print(f"  rho={rho} {i}/{N} {time.time()-t0:.0f}s",flush=True)
    np.savez(f"{TD}/tfmq_rho{rho}_N{N}.npz",**{n:np.array(Q[n]) for n in SETS},
             **{f"err_{n}":np.abs(xh[n][START:START+N]-XF[n][START:START+N]).max() for n in SETS})
    print(f"rho={rho} done {time.time()-t0:.0f}s ({N*len(SETS)/(time.time()-t0):.0f} val/s)",flush=True)
  sys.exit()

# ---- scoring ----
res={}
print(f"REAL BYTES (bits/value), N={N}, overhead-amortized.  'ideal' = my idealized coder for reference")
hdr=(f"{'dataset':17s}{'rho':>6s}{'SZ3':>8s}{'ZFP':>8s}{'lorenzo1':>9s}{'lpc32':>8s}"
     f"{'intCubic':>9s}{'CLASSIC':>8s}{'TimesFM':>8s}{'gain%':>7s}{'vsSZ3%':>8s}")
print(hdr); print("-"*len(hdr))
allg=[]; allsz=[]
for rho in RHO:
  for n in SETS:
    x=XF[n][START:START+N]; tau=max(rho*float(np.std(XF[n])),0.5)
    row={}
    for k,f in PREDS.items():
      q,xh=f(x,tau); assert np.abs(xh-x).max()<=tau+1e-6
      row[k]=real_bytes(q)*8.0/N
    sz,szerr=sz3_run(x,tau)
    zc=zfpy.compress_numpy(np.ascontiguousarray(x),tolerance=tau)
    zf=len(zc)*8.0/N
    d=np.load(f"{TD}/tfmq_rho{rho}_N{N}.npz")
    tfm=real_bytes(d[n])*8.0/N
    assert float(d[f"err_{n}"])<=tau+1e-6
    cl=min(row.values()); ck=min(row,key=row.get)
    g=100*(1-tfm/cl); gs=100*(1-tfm/sz) if sz else float("nan")
    allg.append(g); allsz.append(gs)
    res[f"{n}|{rho}"]=dict(sz3=sz,zfp=zf,classic=row,best=ck,tfm=tfm,gain=g,gain_sz3=gs)
    print(f"{n:17s}{rho:6.3f}{sz:8.3f}{zf:8.3f}{row['lorenzo1']:9.3f}{row['lpc32']:8.3f}"
          f"{row['interp_cubic']:9.3f}{cl:8.3f}{tfm:8.3f}{g:+7.1f}{gs:+8.1f}")
  print()
print("-"*len(hdr))
print(f"TimesFM vs best-of-6 classical (same coder): median {np.median(allg):+.1f}%  wins {sum(1 for v in allg if v>0)}/{len(allg)}")
print(f"TimesFM vs real SZ3 binary:                  median {np.median(allsz):+.1f}%  wins {sum(1 for v in allsz if v>0)}/{len(allsz)}")
json.dump(res,open("results/final_bench.json","w"),indent=1,default=float)
