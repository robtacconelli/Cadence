"""SDRBench evaluation -- the standard corpus for error-bounded lossy compression.

Framing, stated up front: SDRBench is scientific simulation output, and our
domain characterization predicts Cadence LOSES here. Smooth multidimensional
fields are exactly where a local/interpolating predictor is already near-optimal.
Running it is therefore a falsification test of the domain claim, not an attempt
to win. We compare 1-D against 1-D (SZ3 -1) for fairness, and additionally
report SZ3 in its native dimensionality, which our 1-D codec cannot match by
construction.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, glob, time, json, subprocess, lzma
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import zstandard as zstd
from sz_style import PREDS

CTX=512; H=64; N=int(os.environ.get("N",2048)); START=CTX
RHO=[0.01,0.05,0.2]
TD=CACHE
SZ3="ext/sz3-install/bin/sz3"
env=dict(os.environ,LD_LIBRARY_PATH=os.path.abspath("ext/sz3-install/lib"))

def load_fields():
  F={}
  for f in sorted(glob.glob("data/sdrb/exaalt/*/*.f32")):
    a=np.fromfile(f,dtype=np.float32).astype(np.float64)
    F["exaalt_"+os.path.basename(f)[:-4]]=a[:START+N+H]
  for f in sorted(glob.glob("data/sdrb/isabel_*/*/*.f32"))[:3]:
    a=np.fromfile(f,dtype=np.float32).astype(np.float64)
    nm="isabel_"+os.path.basename(f)[:-4]
    # a 1-D scanline through the middle of the 100x500x500 volume
    if a.size>=500*500*100:
      v=a.reshape(100,500,500); F[nm]=np.concatenate([v[50,i,:] for i in range(6)])[:START+N+H]
    else: F[nm]=a[:START+N+H]
  return {k:v for k,v in F.items() if len(v)>=START+N+H and np.std(v)>0}

def nb(q):
  q=np.asarray(np.rint(q),dtype=np.int64); u=q-q.min(); s=int(u.max())
  for w,dt in ((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64)):
    if s<2**(8*w): buf=u.astype(dt).tobytes(); break
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return (min(len(lzma.compress(buf,preset=9|lzma.PRESET_EXTREME)),len(zc.compress(buf)))+24)*8.0/len(q)

def sz3(x,tau,dims):
  f=f"{TD}/sdr.bin"; x.astype(np.float32).tofile(f)
  cmd=[SZ3,"-f","-i",f,"-z",f+".sz","-o",f+".out"]+dims+["-M","ABS","-A",str(tau)]
  r=subprocess.run(cmd,capture_output=True,env=env)
  if r.returncode!=0: return np.nan,np.nan
  y=np.fromfile(f+".out",dtype=np.float32).astype(np.float64)
  return os.path.getsize(f+".sz")*8.0/x.size, float(np.abs(y-x).max()/tau)

if len(sys.argv)>1 and sys.argv[1]=="gpu":
  from timesfm3 import TimesFM3Forecaster, ModelConfig
  F=load_fields(); keys=[(k,r) for r in RHO for k in F]
  B=len(keys); print(f"{len(F)} fields x {len(RHO)} rho = {B} loops: {sorted(F)}",flush=True)
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=B, device="cuda"))
  taus={k:max(k[1]*float(np.std(F[k[0]])),1e-12) for k in keys}
  xh={k:F[k[0]].copy() for k in keys}; Q={k:np.empty(N) for k in keys}
  t0=time.time()
  for i in range(N):
    t=START+i
    P=list(fc.predict_batch(contexts=[xh[k][t-CTX:t].astype(np.float32) for k in keys],
           horizon=H,return_quantiles=False,sort_quantiles=True))
    for j,k in enumerate(keys):
      p=float(np.asarray(P[j].forecast)[0]); D=2*taus[k]
      q=np.rint((F[k[0]][t]-p)/D); xh[k][t]=p+q*D; Q[k][i]=q
    if i%400==0: print(f"  {i}/{N} {time.time()-t0:.0f}s",flush=True)
  np.savez_compressed(f"{TD}/sdrb_N{N}.npz",
    fields=np.array(sorted(F)),
    **{f"q|{k}|{r}":Q[(k,r)] for k,r in keys},
    **{f"tau|{k}|{r}":taus[(k,r)] for k,r in keys},
    **{f"err|{k}|{r}":np.abs(xh[(k,r)][START:START+N]-F[k][START:START+N]).max() for k,r in keys})
  print(f"done {time.time()-t0:.0f}s",flush=True); sys.exit()

F=load_fields(); d=np.load(f"{TD}/sdrb_N{N}.npz",allow_pickle=True)
res={}; G=[]
print("SDRBENCH (scientific simulation output) -- bits/value, real bytes")
print(f"{'field':22s}{'rho':>6s}{'classic':>9s}{'Cadence':>9s}{'gain':>8s}{'SZ3 1-D':>9s}")
for r in RHO:
  for k in sorted(F):
    x=F[k][START:START+N]; tau=float(d[f"tau|{k}|{r}"])
    assert float(d[f"err|{k}|{r}"])<=tau*1.001+1e-12
    cl=min(nb(f(x,tau)[0]) for f in PREDS.values())
    c=nb(d[f"q|{k}|{r}"]); s1,_=sz3(x,tau,["-1",str(N)])
    g=100*(1-c/cl); G.append(g)
    res[f"{k}|{r}"]=dict(classic=cl,cadence=c,gain=g,sz3_1d=s1)
    print(f"{k[:21]:22s}{r:6.2f}{cl:9.3f}{c:9.3f}{g:+7.1f}%{s1:9.3f}")
  print()
G=np.array(G)
print(f"OVERALL vs best-of-six classical: median {np.median(G):+.1f}%  "
      f"mean {np.mean(G):+.1f}%  wins {int((G>0).sum())}/{len(G)}")
json.dump(res,open("results/sdrbench.json","w"),indent=1,default=float)
