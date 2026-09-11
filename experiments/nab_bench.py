"""Error-bounded lossy on REAL operational/observability data (NAB).

This is the domain where the wins concentrated, and where SZ3/ZFP are not even
the incumbents -- TSDBs use lossless Gorilla XOR or naive downsampling.
Primary comparison uses IDEAL codelengths so every predictor shares one coder
and no container overhead distorts the result (the flaw that broke the earlier
SZ3 numbers at small N); real bytes reported alongside.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, json, time, lzma
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import zstandard as zstd
from sz_style import PREDS, _bits
from fair import adaptive_resid_bits

CTX=512; H=64; N=2048; START=CTX
SETS=["ec2_cpu_utilization_5f5533","ec2_network_in_5abac7","ec2_disk_write_bytes_1ef3de",
      "rds_cpu_utilization_cc0c53","grok_asg_anomaly","nyc_taxi",
      "ambient_temperature_system_failure","machine_temperature_system_failure"]
RHO=[float(r) for r in os.environ.get("RHO","0.01,0.05,0.2").split(",")]
TD=CACHE
XF={n:np.load(f"data/nab/{n}.npy").astype(np.float64) for n in SETS}
for n in SETS: assert len(XF[n])>=START+N+H, (n,len(XF[n]))

def rb(q):
  q=np.asarray(q,dtype=np.int64); u=q-q.min(); s=int(u.max())
  for w,dt in ((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64)):
    if s<2**(8*w): buf=u.astype(dt).tobytes(); break
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return min(len(lzma.compress(buf,preset=9|lzma.PRESET_EXTREME)),len(zc.compress(buf)))

def downsample(x,R):
  pos=np.arange(0,len(x),R); k=x[pos]
  d=np.diff(k,prepend=k[0]).astype(np.int64)
  rec=np.interp(np.arange(len(x)),pos,k)
  return rb(d)*8.0/len(x), float(np.abs(rec-x).max()), float(np.abs(rec-x).mean())

if len(sys.argv)>1 and sys.argv[1]=="gpu":
  from timesfm3 import TimesFM3Forecaster, ModelConfig
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=len(SETS),device="cuda"))
  for rho in RHO:
    taus={n:max(rho*float(np.std(XF[n])),0.5) for n in SETS}
    xh={n:XF[n].copy() for n in SETS}; Q={n:[] for n in SETS}; t0=time.time()
    for i in range(N):
      t=START+i
      outs=list(fc.predict_batch(contexts=[xh[n][t-CTX:t].astype(np.float32) for n in SETS],
                horizon=H,return_quantiles=False,sort_quantiles=True))
      for j,n in enumerate(SETS):
        p=float(np.asarray(outs[j].forecast)[0]); D=2*taus[n]
        k=np.rint((XF[n][t]-p)/D); xh[n][t]=p+k*D; Q[n].append(k)
      if i%500==0: print(f"  rho={rho} {i}/{N} {time.time()-t0:.0f}s",flush=True)
    np.savez(f"{TD}/nabq_rho{rho}.npz",**{n:np.array(Q[n]) for n in SETS},
             **{f"err_{n}":np.abs(xh[n][START:START+N]-XF[n][START:START+N]).max() for n in SETS})
    print(f"rho={rho} done {time.time()-t0:.0f}s",flush=True)
  sys.exit()

res={}; allg=[]; dsr=[]
print("REAL OBSERVABILITY DATA (NAB) -- bits/value, identical adaptive coder for all")
hdr=(f"{'series':36s}{'rho':>6s}{'lorenzo1':>9s}{'lpc32':>7s}{'intCub':>8s}{'CLASSIC':>8s}"
     f"{'TimesFM':>8s}{'gain%':>7s}")
print(hdr); print("-"*len(hdr))
for rho in RHO:
  f=f"{TD}/nabq_rho{rho}.npz"
  if not os.path.exists(f): print(f"  [pending {f}]"); continue
  d=np.load(f)
  for n in SETS:
    x=XF[n][START:START+N]; tau=max(rho*float(np.std(XF[n])),0.5)
    row={}
    for k,fn in PREDS.items():
      q,xhh=fn(x,tau); assert np.abs(xhh-x).max()<=tau+1e-6
      row[k]=_bits(q)
    tfm=_bits(np.asarray(d[n])); assert float(d[f"err_{n}"])<=tau+1e-6
    cl=min(row.values()); g=100*(1-tfm/cl); allg.append(g)
    res[f"{n}|{rho}"]=dict(classic=row,best=min(row,key=row.get),tfm=tfm,gain=g,tau=tau)
    print(f"{n[:35]:36s}{rho:6.3f}{row['lorenzo1']:9.3f}{row['lpc32']:7.3f}"
          f"{row['interp_cubic']:8.3f}{cl:8.3f}{tfm:8.3f}{g:+7.1f}")
  print()
if allg:
  print("-"*len(hdr))
  print(f"TimesFM vs best-of-6 classical: median {np.median(allg):+.1f}%  mean {np.mean(allg):+.1f}%  "
        f"wins {sum(1 for v in allg if v>0)}/{len(allg)}")
  for rho in RHO:
    g=[v["gain"] for k,v in res.items() if k.endswith(f"|{rho}")]
    if g: print(f"   rho={rho:<6} median {np.median(g):+6.1f}%  wins {sum(1 for i in g if i>0)}/{len(g)}")
  # vs downsampling at matched rate
  print(f"\nvs DOWNSAMPLING at matched bitrate (what TSDBs actually do):")
  print(f"{'series':36s}{'rho':>6s}{'TFM bpv':>9s}{'tau':>10s}{'R':>4s}{'DS bpv':>8s}{'DS maxE':>11s}{'xTau':>7s}")
  for rho in RHO:
    fp=f"{TD}/nabq_rho{rho}.npz"
    if not os.path.exists(fp): continue
    d=np.load(fp)
    for n in SETS:
      x=XF[n][START:START+N]; tau=max(rho*float(np.std(XF[n])),0.5)
      tfm=_bits(np.asarray(d[n])); best=None
      for R in [2,4,8,16,32,64,128,256]:
        b,mx,mn=downsample(x,R)
        if b<=tfm and (best is None or b>best[0]): best=(b,R,mx,mn)
      if not best: continue
      b,R,mx,mn=best; dsr.append(mx/tau)
      print(f"{n[:35]:36s}{rho:6.3f}{tfm:9.3f}{tau:10.1f}{R:4d}{b:8.3f}{mx:11.1f}{mx/tau:6.1f}x")
  if dsr: print(f"\nAt equal size, downsampling's WORST-CASE error is {np.median(dsr):.1f}x (median) "
                f"the bound TimesFM guarantees.")
json.dump(res,open("results/nab_bench.json","w"),indent=1,default=float)
