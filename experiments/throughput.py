"""How fast can this actually go? The closed loop is sequential per series, but
independent series batch freely -- so throughput scales with batch, not with
per-step latency. Also test reduced precision and shorter context."""
import numpy as np, torch, time, sys, os, json
import os as _os
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig

x=np.load("data/real/wikihr_en.npy").astype(np.float64)
def bench(bs, ctx, dtype=None, iters=6, H=64):
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=bs, device="cuda"))
  if dtype is not None:
    fc.model=fc.model.to(dtype)
  ctxs=[x[i:i+ctx].astype(np.float32) for i in range(bs)]
  for _ in range(2):                                  # warmup
    list(fc.predict_batch(contexts=ctxs,horizon=H,return_quantiles=False))
  torch.cuda.synchronize(); t0=time.time()
  for _ in range(iters):
    list(fc.predict_batch(contexts=ctxs,horizon=H,return_quantiles=False))
  torch.cuda.synchronize(); dt=(time.time()-t0)/iters
  mem=torch.cuda.max_memory_allocated()/1e9
  del fc; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
  return dt, bs/dt, mem

print(f"{'batch':>6s}{'ctx':>6s}{'dtype':>10s}{'s/step':>9s}{'values/s':>10s}{'VRAM GB':>9s}{'vs base':>9s}")
print("-"*60)
base=None; rows=[]
for bs,ctx,dt_ in [(6,1024,None),(32,1024,None),(64,1024,None),(128,1024,None),(256,1024,None),
                   (256,512,None),(512,512,None),
                   (256,1024,torch.bfloat16),(512,1024,torch.bfloat16),(1024,512,torch.bfloat16)]:
  try:
    s,v,m=bench(bs,ctx,dt_)
    if base is None: base=v
    nm={None:"fp32",torch.bfloat16:"bf16"}[dt_]
    rows.append(dict(batch=bs,ctx=ctx,dtype=nm,s_step=s,vps=v,vram=m,speedup=v/base))
    print(f"{bs:6d}{ctx:6d}{nm:>10s}{s:9.3f}{v:10.0f}{m:9.2f}{v/base:8.1f}x")
  except RuntimeError as e:
    print(f"{bs:6d}{ctx:6d}{'OOM' if 'memory' in str(e) else 'ERR':>10s}  {str(e)[:40]}")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
json.dump(rows,open("results/throughput.json","w"),indent=1)
