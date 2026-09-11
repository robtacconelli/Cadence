"""One consistent back end across every corpus: the adaptive arithmetic coder,
applied identically to Cadence and to all six classical predictors."""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, glob, json, time
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import rangecoder as rc
from sz_style import PREDS
TD=CACHE
def ac(q): return (len(rc.encode(np.asarray(q,dtype=np.int64)))+24)*8.0/len(q)
RHO=[0.01,0.05,0.2]
CORP=[
 ("Grid load","data/grid",f"{TD}/gridq_N2048.npz",512,2048,None),
 ("Transit","data/transit",f"{TD}/transitq_N2048.npz",512,2048,None),
 ("NAB oper.","data/nab",None,512,2048,"nab"),
 ("Synthetic","data/synth",None,1024,8192,"syn"),
 ("SDRBench",None,f"{TD}/sdrb_N2048.npz",512,2048,"sdr"),
]
SYN=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic"]
NAB=["ec2_cpu_utilization_5f5533","ec2_network_in_5abac7","ec2_disk_write_bytes_1ef3de",
     "rds_cpu_utilization_cc0c53","grok_asg_anomaly","nyc_taxi",
     "ambient_temperature_system_failure","machine_temperature_system_failure"]
out={}; t0=time.time()
for name,dd,qf,CTX,N,kind in CORP:
  G=[]
  if kind=="sdr":
    d=np.load(qf,allow_pickle=True); fields=[str(x) for x in d["fields"]]
    import sdrbench as sb; F=sb.load_fields()
    for r in RHO:
      for k in fields:
        x=F[k][CTX:CTX+N]; tau=float(d[f"tau|{k}|{r}"])
        cl=min(ac(f(x,tau)[0]) for f in PREDS.values())
        G.append(100*(1-ac(d[f"q|{k}|{r}"])/cl))
  elif kind in ("nab","syn"):
    sets=NAB if kind=="nab" else SYN
    for r in RHO:
      f_=f"{TD}/nabq_rho{r}.npz" if kind=="nab" else f"{TD}/tfmq_rho{r}_N8192.npz"
      d=np.load(f_)
      for n in sets:
        xf=np.load(f"{dd}/{n}.npy").astype(float); x=xf[CTX:CTX+N]
        tau=max(r*float(np.std(xf)),0.5)
        cl=min(ac(f(x,tau)[0]) for f in PREDS.values())
        G.append(100*(1-ac(d[n])/cl))
  else:
    d=np.load(qf); sets=sorted({os.path.basename(p)[:-4] for p in glob.glob(f"{dd}/*.npy")}-{"SEC"})
    sets=[s for s in sets if f"{s}|0.01" in d]
    for r in RHO:
      for n in sets:
        xf=np.load(f"{dd}/{n}.npy").astype(float); x=xf[CTX:CTX+N]
        tau=max(r*float(np.std(xf)),0.5)
        cl=min(ac(f(x,tau)[0]) for f in PREDS.values())
        G.append(100*(1-ac(d[f"{n}|{r}"])/cl))
  G=np.array(G); out[name]=dict(gains=G.tolist(),median=float(np.median(G)),
                                mean=float(np.mean(G)),wins=int((G>0).sum()),n=len(G))
  print(f"{name:12s} median {np.median(G):+7.1f}%  mean {np.mean(G):+7.1f}%  "
        f"wins {int((G>0).sum())}/{len(G)}   [{time.time()-t0:.0f}s]",flush=True)
dem=np.array(out["Grid load"]["gains"]+out["Transit"]["gains"])
print(f"\nDEMAND COMBINED: median {np.median(dem):+.1f}%  mean {np.mean(dem):+.1f}%  "
      f"wins {int((dem>0).sum())}/{len(dem)}")
out["_demand_combined"]=dict(median=float(np.median(dem)),mean=float(np.mean(dem)),
                             wins=int((dem>0).sum()),n=len(dem))
json.dump(out,open("results/rescore_all.json","w"),indent=1)
