"""Real data + the cross-channel test.

Codec framing: encode channel A in full, then encode channel B conditioned on
A's ENTIRE trajectory (past and future are both known to the decoder by then).
That is what past_future_covariates gives us, and no classical compressor can
do anything like it.
"""
import numpy as np, json, os, sys, time
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig
from tfm_entropy2 import score, running_mean
from strong_lpc import lag_lpc_bits

CTX=int(os.environ.get("CTX",1024)); NPOS=int(os.environ.get("NPOS",2500))
BATCH=int(os.environ.get("BATCH",48)); STRIDE=1
fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                  per_core_batch_size=BATCH,device="cuda"))

def collect(x, cov=None, ctx=CTX, npos=NPOS, horizon=64):
  """cov: full-length aligned covariate series (known to decoder), or None."""
  x=np.asarray(x,dtype=np.float64)
  starts=list(range(ctx,min(ctx+npos,len(x)-horizon)))
  Q=np.zeros((len(starts),9)); V=np.zeros(len(starts)); t0=time.time()
  for s in range(0,len(starts),BATCH):
    ch=starts[s:s+BATCH]
    ctxs=[x[t-ctx:t].astype(np.float32) for t in ch]
    kw={}
    if cov is not None:
      kw["past_future_covariates"]=[cov[t-ctx:t+horizon].astype(np.float32)[None,:] for t in ch]
    outs=list(fc.predict_batch(contexts=ctxs,horizon=horizon,return_quantiles=True,
                               sort_quantiles=True,**kw))
    for j,t in enumerate(ch):
      Q[s+j]=np.asarray(outs[j].quantiles,dtype=np.float64).reshape(-1,9)[0]
      V[s+j]=x[t]
  return Q,V,np.array(starts),time.time()-t0

res={}
en=np.load("data/real/wikihr_en.npy"); de=np.load("data/real/wikihr_de.npy")
strong=json.load(open("results/strong_lpc.json"))
for name,x,cov,covname in [("wikihr_en",en,None,"-"),("wikihr_de",de,None,"-"),
                           ("wikihr_en|de",en,de,"de"),("wikihr_de|en",de,en,"en")]:
  Q,V,pos,gs=collect(x,cov)
  r=score(x,Q,V,pos)
  key=name.split("|")[0]
  r["_classical_best"]=min(strong[key].values())
  r["_vals_per_s"]=len(V)/gs; r["_cov"]=covname
  res[name]=r
  print(f"{name:14s} cov={covname:3s} qcdf={r['qcdf']:7.3f} lapQcal={r['lapQcal']:7.3f} "
        f"lpc32={r['lpc32']:7.3f} stack={r['stack']:7.3f} stack_nomodel={r['stack_nomodel']:7.3f} "
        f"MIX={r['MIX']:7.3f} | classical={r['_classical_best']:7.3f} "
        f"| MAEmodel={r['_mae_model']:.0f} MAElpc={r['_mae_lpc']:.0f} MAEstack={r['_mae_stack']:.0f}")
json.dump(res,open("results/real_crosschannel.json","w"),indent=1)
