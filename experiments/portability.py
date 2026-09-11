"""Cross-device bitstream portability.

Section 5.6 showed batch-size invariance is unattainable on one GPU. The paper's
Limitations flag cross-machine reproducibility as untested. We cannot test two
machines here, but we can test something strictly harder on the same box:
CPU vs GPU execution of the identical model and inputs. If the bitstream does
not survive a device change it certainly will not survive a hardware change.

Also tests fp64, which is the obvious proposed remedy.
"""
import numpy as np, torch, sys, json, time
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig

CTX=512; H=64; G=8
NAMES=["MISO","ERCO","PJM","CISO","BPAT","SOCO","DUK","AZPS"]
X={n:np.load(f"data/grid/{n}.npy").astype(np.float64) for n in NAMES}
T=2000
def wins(): return [X[n][T-CTX:T].astype(np.float32) for n in NAMES]

def preds(fc, w):
  o=list(fc.predict_batch(contexts=w,horizon=H,return_quantiles=False,sort_quantiles=True))
  return np.array([float(np.asarray(r.forecast)[0]) for r in o])

res={}
def load(dev, dtype=None):
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=G, device=dev))
  if dtype is not None: fc.model=fc.model.to(dtype)
  return fc

print("Cross-device / cross-precision agreement, batch=8, ctx=512, grid load (MW)\n")
t0=time.time(); fg=load("cuda"); pg=preds(fg,wins()); tg=time.time()-t0
del fg; torch.cuda.empty_cache()
t0=time.time(); fc_=load("cpu"); pc=preds(fc_,wins()); tc=time.time()-t0
d=np.abs(pg-pc)
print(f"{'GPU fp32 vs CPU fp32':32s} identical={np.array_equal(pg,pc)}  "
      f"max|diff|={d.max():.4e}  ({(d>0).sum()}/{len(d)} series differ)")
res["gpu_vs_cpu_fp32"]={"identical":bool(np.array_equal(pg,pc)),"max":float(d.max())}
del fc_

try:
  torch.set_default_dtype(torch.float64)
  fd=load("cpu"); pd_=preds(fd,wins()); d2=np.abs(pd_-pc)
  print(f"{'CPU fp64 vs CPU fp32':32s} identical={np.array_equal(pd_,pc)}  max|diff|={d2.max():.4e}")
  res["cpu64_vs_cpu32"]={"identical":bool(np.array_equal(pd_,pc)),"max":float(d2.max())}
  del fd
except Exception as e:
  print(f"{'CPU fp64':32s} n/a: {repr(e)[:70]}")
  res["cpu64_vs_cpu32"]=None
finally:
  torch.set_default_dtype(torch.float32)

# what does that drift mean for a codec at rho=0.05?
tau={n:max(0.05*float(np.std(X[n])),0.5) for n in NAMES}
D=np.array([2*tau[n] for n in NAMES])
print(f"\nquantization steps D=2tau (MW): {np.round(D,1)}")
print(f"GPU-vs-CPU drift as a fraction of D: "
      f"{np.round(d/D,6)}")
print(f"  worst = {float((d/D).max()):.3e}  -> per-sample desync probability ~{float((d/D).max()):.1e}")
res["desync_prob_per_sample"]=float((d/D).max())
# expected samples before a desync
p=float((d/D).max())
if p>0: print(f"  expected samples before first desync ~ {1/p:,.0f}")
res["expected_samples_to_desync"]=(1/p if p>0 else None)
json.dump(res,open("results/portability.json","w"),indent=1)
