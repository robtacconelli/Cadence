"""Run the GPU once, cache (quantiles, values, positions) so scoring can be
iterated on CPU without re-running the model."""
import numpy as np, glob, os, sys, time
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig

CTX=int(os.environ.get("CTX",1024)); NPOS=int(os.environ.get("NPOS",3000))
BATCH=int(os.environ.get("BATCH",48)); H=64
os.makedirs("cache",exist_ok=True)
fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                  per_core_batch_size=BATCH,device="cuda"))

def collect(x,cov=None,ctx=CTX,npos=NPOS):
  x=np.asarray(x,dtype=np.float64)
  starts=list(range(ctx,min(ctx+npos,len(x)-H)))
  Q=np.zeros((len(starts),9)); t0=time.time()
  for s in range(0,len(starts),BATCH):
    ch=starts[s:s+BATCH]
    kw={}
    if cov is not None:
      kw["past_future_covariates"]=[cov[t-ctx:t+H].astype(np.float32)[None,:] for t in ch]
    outs=list(fc.predict_batch(contexts=[x[t-ctx:t].astype(np.float32) for t in ch],
                horizon=H,return_quantiles=True,sort_quantiles=True,**kw))
    for j,t in enumerate(ch):
      Q[s+j]=np.asarray(outs[j].quantiles,dtype=np.float64).reshape(-1,9)[0]
  return Q,np.array([x[t] for t in starts]),np.array(starts),time.time()-t0

jobs=[]
for f in sorted(glob.glob("data/synth/*.npy"))+sorted(glob.glob("data/real/*.npy")):
  n=os.path.basename(f)[:-4]; a=np.load(f)
  if len(a)>=CTX+500: jobs.append((n,a,None,"-"))
en=np.load("data/real/wikihr_en.npy"); de=np.load("data/real/wikihr_de.npy")
jobs+= [("wikihr_en|de",en,de,"de"),("wikihr_de|en",de,en,"en")]
for n,a,cov,cn in jobs:
  out=f"cache/{n.replace('|','_COND_')}_c{CTX}.npz"
  if os.path.exists(out): print("skip",n); continue
  Q,V,pos,dt=collect(a,cov)
  np.savez_compressed(out,Q=Q,V=V,pos=pos,x=np.asarray(a,dtype=np.float64),cov=(cov if cov is not None else np.zeros(0)))
  print(f"{n:16s} n={len(V):5d} {dt:6.1f}s {len(V)/dt:6.0f} val/s -> {out}")
