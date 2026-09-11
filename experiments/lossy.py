"""Error-bounded lossy, CLOSED LOOP, at large stride.

Why this regime and not the lossless one: bits ~ log2(residual_scale/tau), so a
better forecaster only ever buys log2(gain) bits -- the cap the lossless test hit.
But once tau grows past the forecast error, the quantized residual becomes
EXACTLY ZERO and the value costs ~0 bits. That is not a logarithmic gain, it is
a phase transition, and it is also the only way the 107 val/s problem is fixed:
one forward pass covers `stride` values instead of one.

Guarantee: |x_hat - x| <= tau for every sample. Context is fed RECONSTRUCTED
values, so the decoder can reproduce it exactly.
"""
import numpy as np, sys, os, time, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig
from fair import adaptive_resid_bits

CTX=1024; H=64
SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
RHO=[0.001,0.01,0.05,0.2]          # tau = rho * std(x)
STRIDES=[1,8,32,64]

fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                  per_core_batch_size=len(SETS),device="cuda"))
X={n:np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64) for n in SETS}
NP_=int(os.environ.get("NPOS",4096))

def lpc_lossy(x, tau, order=32, npos=NP_, start=CTX):
  """Classical closed-loop bar: LPC on RECONSTRUCTED history, h=1 (decoder
  always has the previous reconstruction, so classical never needs a horizon)."""
  D=2*tau; xh=x.copy(); q=np.empty(npos)
  Xf=np.lib.stride_tricks.sliding_window_view(x,order)[:-1]
  c,*_=np.linalg.lstsq(Xf,x[order:],rcond=None); c=np.rint(c*16384)/16384
  for i in range(npos):
    t=start+i
    pred=float(xh[t-order:t][::-1] @ c[::-1]) if False else float(xh[t-order:t] @ c)
    k=np.rint((x[t]-pred)/D); q[i]=k; xh[t]=pred+k*D
  return q, xh

def tfm_lossy(tau_by, stride, npos=NP_, start=CTX):
  """TimesFM closed loop: one forward pass yields `stride` predictions."""
  xh={n:X[n].copy() for n in SETS}
  q={n:[] for n in SETS}
  nb=npos//stride; t0=time.time()
  for b in range(nb):
    t=start+b*stride
    ctxs=[xh[n][t-CTX:t].astype(np.float32) for n in SETS]
    outs=list(fc.predict_batch(contexts=ctxs,horizon=max(stride,H),
                               return_quantiles=False,sort_quantiles=True))
    for j,n in enumerate(SETS):
      pred=np.asarray(outs[j].forecast,dtype=np.float64)[:stride]
      D=2*tau_by[n]
      k=np.rint((X[n][t:t+stride]-pred)/D)
      xh[n][t:t+stride]=pred+k*D
      q[n].append(k)
  return {n:np.concatenate(q[n]) for n in SETS}, xh, time.time()-t0

tau={n:float(np.std(X[n])) for n in SETS}
res={}
print(f"{'dataset':17s}{'rho':>7s}{'tau':>10s}{'stride':>7s}{'TFM_bpv':>9s}{'zero%':>7s}"
      f"{'LPC_bpv':>9s}{'LPCzero%':>9s}{'gain%':>8s}{'val/s':>8s}")
print("-"*100)
for rho in RHO:
  tb={n:max(rho*tau[n],0.5) for n in SETS}
  lp={}
  for n in SETS:
    qq,_=lpc_lossy(X[n],tb[n])
    lp[n]=(adaptive_resid_bits(qq,max(abs(qq).max()*2+1,3)), float((qq==0).mean()))
  for st in STRIDES:
    qs,xh,dt=tfm_lossy(tb,st)
    for n in SETS:
      qq=qs[n]
      err=np.abs(xh[n][CTX:CTX+len(qq)]-X[n][CTX:CTX+len(qq)]).max()
      assert err<=tb[n]+1e-6, (n,err,tb[n])
      bpv=adaptive_resid_bits(qq,max(abs(qq).max()*2+1,3))
      z=float((qq==0).mean())
      g=100*(1-bpv/lp[n][0])
      res[f"{n}|{rho}|{st}"]=dict(tfm=bpv,zero=z,lpc=lp[n][0],lpczero=lp[n][1],gain=g,
                                  vps=len(qq)*len(SETS)/dt)
      print(f"{n:17s}{rho:7.3f}{tb[n]:10.1f}{st:7d}{bpv:9.3f}{100*z:7.1f}"
            f"{lp[n][0]:9.3f}{100*lp[n][1]:9.1f}{g:+8.1f}{len(qq)*len(SETS)/dt:8.0f}")
  print()
json.dump(res,open("results/lossy.json","w"),indent=1)
