import numpy as np, json, glob, os, sys, time
sys.path.insert(0, "src")
from timesfm3 import TimesFM3Forecaster, ModelConfig
from tfm_entropy import score_series
from baselines import all_baselines

ctx    = int(os.environ.get("CTX", 1024))
stride = int(os.environ.get("STRIDE", 64))
batch  = int(os.environ.get("BATCH", 32))
tag    = os.environ.get("TAG", f"c{ctx}_s{stride}")

fc = TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                    per_core_batch_size=batch, device="cuda"))
base = json.load(open("results/baselines.json"))
rows = {}
print(f"\n=== ctx={ctx} stride={stride} ===")
hdr = f"{'dataset':18s} {'raw':>5s} {'BEST-CLASSICAL':>22s} {'qcdf':>7s} {'bestLap':>8s} {'MIX':>7s} {'win%':>6s} {'val/s':>7s}"
print(hdr); print("-" * len(hdr))
for f in sorted(glob.glob("data/synth/*.npy")):
  name = os.path.basename(f)[:-4]
  x = np.load(f)
  r = score_series(fc, x, ctx=ctx, stride=stride, batch=batch)
  b = base[name]
  cls = {k: v for k, v in b.items() if not k.startswith("_")}
  bk = min(cls, key=cls.get); bv = cls[bk]
  laps = {k: v for k, v in r.items() if k.startswith(("lap", "t3"))}
  blk = min(laps, key=laps.get)
  win = 100 * (1 - r["MIX"] / bv)
  rows[name] = dict(model=r, classical=b, best_classical=(bk, bv), win_pct=win)
  print(f"{name:18s} {b['_raw_bpv']:5.1f} {bk+' '+format(bv,'.3f'):>22s} "
        f"{r['qcdf']:7.3f} {r[blk]:8.3f} {r['MIX']:7.3f} {win:+6.1f} {r['_vals_per_s']:7.0f}")
json.dump(rows, open(f"results/viability_{tag}.json", "w"), indent=1)
print(f"\nsaved results/viability_{tag}.json")
