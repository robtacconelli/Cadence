"""Can batch-size invariance be forced? Test the knobs that plausibly cause it:
SDPA kernel selection (batch-dependent), TF32, and torch deterministic mode."""
import numpy as np, torch, sys, json, os
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig
CTX=512;H=64
NAMES=["MISO","ERCO","PJM","CISO","BPAT","SOCO","DUK","AZPS"]
X={n:np.load(f"data/grid/{n}.npy").astype(np.float64) for n in NAMES}
T=2000
def mk(n): return X[n][T-CTX:T].astype(np.float32)
def preds(fc,ctxs):
  o=list(fc.predict_batch(contexts=ctxs,horizon=H,return_quantiles=False,sort_quantiles=True))
  return np.array([float(np.asarray(r.forecast)[0]) for r in o])

def trial(label, use_sdpa, tf32, determ, backend=None):
  torch.backends.cuda.matmul.allow_tf32=tf32
  torch.backends.cudnn.allow_tf32=tf32
  if determ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
  cfg=ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",per_core_batch_size=32,
                  device="cuda", use_sdpa=use_sdpa)
  fc=TimesFM3Forecaster(cfg)
  ctx=None
  if backend is not None:
    ctx=torch.nn.attention.sdpa_kernel(backend)
  def go():
    a=preds(fc,[mk(n) for n in NAMES])
    s=np.array([preds(fc,[mk(n)])[0] for n in NAMES])
    p=preds(fc,[mk(n) for n in NAMES]+[mk("MISO")])[:len(NAMES)]
    return a,s,p
  if ctx is not None:
    with ctx: a,s,p=go()
  else: a,s,p=go()
  ok1=np.array_equal(a,s); ok2=np.array_equal(a,p)
  print(f"{label:34s} b1==b8:{str(ok1):>5s} ({np.abs(a-s).max():.2e})   "
        f"b8==b9:{str(ok2):>5s} ({np.abs(a-p).max():.2e})")
  del fc; torch.cuda.empty_cache()
  return ok1 and ok2

r={}
r["default"]=trial("default (sdpa on, tf32 default)",True,torch.backends.cuda.matmul.allow_tf32,False)
r["no_tf32"]=trial("tf32 OFF",True,False,False)
r["no_sdpa"]=trial("use_sdpa=False (plain matmul)",False,False,False)
r["no_sdpa_det"]=trial("use_sdpa=False + deterministic",False,False,True)
try:
  from torch.nn.attention import SDPBackend
  r["math_backend"]=trial("SDPA forced MATH backend",True,False,True,[SDPBackend.MATH])
except Exception as e: print("math backend n/a:",repr(e)[:70])
print("\nbatch-size invariant configs:",[k for k,v in r.items() if v] or "NONE")
json.dump(r,open("results/determinism2.json","w"),indent=1)
