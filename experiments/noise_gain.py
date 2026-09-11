"""Mechanism test: how much does each predictor AMPLIFY noise in its own input?

In closed-loop lossy coding the predictor is fed its own reconstruction, so
every input sample carries quantization error of up to tau. Define the gain

    G = std(pred(x + eps) - pred(x)) / std(eps)

G = 1 for Lorenzo-1 (pred = x[t-1]); G = ||c||_2 for an LPC with coefficients c,
which can be >> 1. If TimesFM has G < 1 it is CONTRACTIVE, and the closed loop
is where that pays -- which would explain why a 26x-less-accurate predictor wins.

Falsifiable prediction: total closed-loop error ~ sqrt(clean_err^2 + (G*tau_rms)^2),
so the predictor with lower clean error wins at small tau and the predictor with
lower G wins at large tau. The crossover tau is computable -- and must match the
rho sweep already measured in results/lossy_parsed.json.
"""
import numpy as np, sys, os, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig

CTX=1024; H=64; NPOS=192; BATCH=48
SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
RHO_TEST=[0.001,0.01,0.05]
rng=np.random.default_rng(7)
fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                  per_core_batch_size=BATCH,device="cuda"))
X={n:np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64) for n in SETS}

def tfm_pred(x, starts, noise=None):
  out=np.empty(len(starts))
  for s in range(0,len(starts),BATCH):
    ch=starts[s:s+BATCH]
    ctxs=[]
    for i,t in enumerate(ch):
      c=x[t-CTX:t].copy()
      if noise is not None: c=c+noise[s+i]
      ctxs.append(c.astype(np.float32))
    o=list(fc.predict_batch(contexts=ctxs,horizon=H,return_quantiles=False,sort_quantiles=True))
    for j in range(len(ch)): out[s+j]=float(np.asarray(o[j].forecast)[0])
  return out

def lpc_coef(x,order):
  Xf=np.lib.stride_tricks.sliding_window_view(x,order)[:-1]
  c,*_=np.linalg.lstsq(Xf,x[order:],rcond=None)
  return np.rint(c*16384)/16384

res={}
print(f"{'dataset':17s}{'rho':>7s}{'tau':>10s}{'G_tfm':>8s}{'G_lpc32':>9s}{'G_lor1':>8s}"
      f"{'clean_tfm':>11s}{'clean_lpc':>11s}{'pred_win':>10s}")
print("-"*98)
for n in SETS:
  x=X[n]; sd=float(np.std(x))
  starts=np.linspace(CTX, len(x)-H-1, NPOS).astype(int)
  c32=lpc_coef(x,32); G_lpc=float(np.linalg.norm(c32))       # analytic, iid input noise
  p_tfm=tfm_pred(x,starts)
  Xw=np.stack([x[t-32:t] for t in starts])
  p_lpc=Xw@c32
  clean_tfm=float(np.std(x[starts]-p_tfm)); clean_lpc=float(np.std(x[starts]-p_lpc))
  for rho in RHO_TEST:
    tau=max(rho*sd,0.5); eps_rms=tau/np.sqrt(3)              # uniform(-tau,tau)
    noise=rng.uniform(-tau,tau,size=(len(starts),CTX))
    p_tfm_n=tfm_pred(x,starts,noise)
    G_tfm=float(np.std(p_tfm_n-p_tfm)/eps_rms)
    G_lor=1.0
    # falsifiable prediction: total closed-loop error
    tot_tfm=np.hypot(clean_tfm,G_tfm*eps_rms); tot_lpc=np.hypot(clean_lpc,G_lpc*eps_rms)
    win="TFM" if tot_tfm<tot_lpc else "LPC"
    res[f"{n}|{rho}"]=dict(G_tfm=G_tfm,G_lpc=G_lpc,clean_tfm=clean_tfm,clean_lpc=clean_lpc,
                           tot_tfm=float(tot_tfm),tot_lpc=float(tot_lpc),pred_win=win,tau=tau)
    print(f"{n:17s}{rho:7.3f}{tau:10.1f}{G_tfm:8.3f}{G_lpc:9.2f}{G_lor:8.1f}"
          f"{clean_tfm:11.1f}{clean_lpc:11.1f}{win:>10s}")
json.dump(res,open("results/noise_gain.json","w"),indent=1)
