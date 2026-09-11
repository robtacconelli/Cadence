import numpy as np, json, glob, os, sys
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig
from tfm_entropy2 import collect, score

CTX=int(os.environ.get("CTX",1024)); STRIDE=int(os.environ.get("STRIDE",1))
BATCH=int(os.environ.get("BATCH",64)); NPOS=int(os.environ.get("NPOS",3000))
TAG=os.environ.get("TAG",f"v2_c{CTX}_s{STRIDE}")
fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                  per_core_batch_size=BATCH,device="cuda"))
base=json.load(open("results/baselines.json")); rows={}
cols=["qcdf","lapA","lapQcal","t3Qcal","lpc32","stack","stack_nomodel","MIX","MIX_classical_only"]
print(f"\n=== ctx={CTX} stride={STRIDE} npos={NPOS} (h0 = true stride-1 quality) ===")
h=f"{'dataset':17s}"+"".join(f"{c[:9]:>10s}" for c in cols)+f"{'gain%':>7s}{'val/s':>7s}"
print(h); print("-"*len(h))
for f in sorted(glob.glob("data/synth/*.npy")):
    name=os.path.basename(f)[:-4]; x=np.load(f)
    Q,V,pos,gs=collect(fc,x,CTX,STRIDE,BATCH,NPOS)
    r=score(x,Q,V,pos)
    gain=100*(1-r["MIX"]/r["MIX_classical_only"])
    r["_gain_vs_classical_mix"]=gain; r["_vals_per_s"]=len(V)/gs
    rows[name]=r
    print(f"{name:17s}"+"".join(f"{r[c]:10.3f}" for c in cols)+f"{gain:+7.2f}{len(V)/gs:7.0f}")
json.dump(rows,open(f"results/{TAG}.json","w"),indent=1)
print(f"\nsaved results/{TAG}.json")
