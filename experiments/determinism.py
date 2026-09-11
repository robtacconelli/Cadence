"""Does the model produce BIT-IDENTICAL predictions across calls?

This is existential for a codec. Three tests, in increasing order of realism:
  A. same batch, repeated       -> run-to-run determinism
  B. batch 1 vs batch B         -> does batch COMPOSITION change the result?
  C. position within the batch  -> does a series' neighbours change its result?

(B) is the realistic failure: an encoder batches many series for throughput, a
decoder may replay one series alone. If outputs differ by even 1 ULP, the
arithmetic decoder desyncs and the file is destroyed from that point on.
"""
import numpy as np, torch, sys, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig

CTX=512; H=64
NAMES=["MISO","ERCO","PJM","CISO","BPAT","SOCO","DUK","AZPS"]
X={n:np.load(f"data/grid/{n}.npy").astype(np.float64) for n in NAMES}
def mk(n,t): return X[n][t-CTX:t].astype(np.float32)

def preds(fc, ctxs):
  o=list(fc.predict_batch(contexts=ctxs,horizon=H,return_quantiles=False,sort_quantiles=True))
  return np.array([float(np.asarray(r.forecast)[0]) for r in o])

res={}
for bs_cfg in [8]:
  fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=32,device="cuda"))
  T=2000
  full=[mk(n,T) for n in NAMES]

  a1=preds(fc,full); a2=preds(fc,full); a3=preds(fc,full)
  same=np.array_equal(a1,a2) and np.array_equal(a1,a3)
  print(f"A. same batch repeated x3      : bit-identical = {same}"
        f"   maxdiff={max(np.abs(a1-a2).max(),np.abs(a1-a3).max()):.3e}")
  res["A_repeat_identical"]=bool(same)

  solo=np.array([preds(fc,[mk(n,T)])[0] for n in NAMES])
  d=np.abs(solo-a1)
  print(f"B. batch 1 vs batch 8          : bit-identical = {np.array_equal(solo,a1)}"
        f"   maxdiff={d.max():.3e}  ({(d>0).sum()}/{len(d)} series differ)")
  res["B_batch1_vs_batch8_identical"]=bool(np.array_equal(solo,a1))
  res["B_maxdiff"]=float(d.max())

  rev=preds(fc,[mk(n,T) for n in NAMES[::-1]])[::-1]
  d2=np.abs(rev-a1)
  print(f"C. reordered batch             : bit-identical = {np.array_equal(rev,a1)}"
        f"   maxdiff={d2.max():.3e}  ({(d2>0).sum()}/{len(d2)} differ)")
  res["C_reorder_identical"]=bool(np.array_equal(rev,a1))
  res["C_maxdiff"]=float(d2.max())

  pad=preds(fc,full+[mk("MISO",T+1)])[:len(NAMES)]
  d3=np.abs(pad-a1)
  print(f"D. batch 8 vs batch 9 (padded) : bit-identical = {np.array_equal(pad,a1)}"
        f"   maxdiff={d3.max():.3e}  ({(d3>0).sum()}/{len(d3)} differ)")
  res["D_batch8_vs_9_identical"]=bool(np.array_equal(pad,a1))
  res["D_maxdiff"]=float(d3.max())

# what does a diff of that size mean against a quantization step?
tau=0.05*float(np.std(X["MISO"]))
print(f"\nquantization step at rho=0.05 for MISO: D = 2*tau = {2*tau:.1f} MW")
print(f"largest observed prediction difference : {max(res['B_maxdiff'],res['C_maxdiff'],res['D_maxdiff']):.3e} MW")
print(f"-> probability a given sample sits within that of a bin edge ~ "
      f"{max(res['B_maxdiff'],res['C_maxdiff'],res['D_maxdiff'])/(2*tau):.2e} per sample")
json.dump(res,open("results/determinism.json","w"),indent=1)
