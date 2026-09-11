"""TimesFM as an INTERPOLATOR (neural super-resolution codec).

Motivation from the head-to-head: SZ3's multilevel interpolation beats TimesFM
on smooth signals precisely because interpolation is anchored -- it sees both
sides, so it is contractive (G<1) and cannot drift. Extrapolation drifts, which
is also exactly why TimesFM's gains die at stride > 8.

So: give TimesFM anchors. Two-level codec.
  L0  code every K-th sample (coarse grid) with a cheap predictor
  L1  upsample the coarse RECONSTRUCTION to full length and hand it to TimesFM
      as a past_future_covariate spanning context AND horizon, then predict the
      fine samples. The model now knows roughly where the signal is going, so
      one forward pass can cover many samples without drifting.

Decoder order: L0 fully, build the covariate, then L1 blockwise. Fully valid.
"""
import numpy as np, sys, os, json, time
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from timesfm3 import TimesFM3Forecaster, ModelConfig
from sz_style import lorenzo, interp_multilevel, _bits
from fair import adaptive_resid_bits

CTX=1024; H=64
SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                  per_core_batch_size=len(SETS),device="cuda"))
XF={n:np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64) for n in SETS}

def coarse_layer(x, tau, K):
  """Code x[::K] with Lorenzo-1 on the coarse grid; return bits and full-length
  linear upsampling of the coarse reconstruction (decoder can build the same)."""
  c=x[::K]
  q,ch=lorenzo(c,tau,1)
  n=len(x); pos=np.arange(0,n,K)
  full=np.interp(np.arange(n),pos,ch)
  return _bits(q)*len(q), full, ch

def run(rho=0.05, K=16, stride=64, N=2048, START=CTX):
  taus={n:max(rho*float(np.std(XF[n])),0.5) for n in SETS}
  # coarse layer needs history before START too, so build it over the whole prefix
  cov={}; cbits={}
  for n in SETS:
    seg=XF[n][:START+N+H]
    b,full,_=coarse_layer(seg,taus[n],K)
    cov[n]=full
    # only the coarse samples inside the scored window count against those N values
    ncoarse=len(range(START,START+N,K))
    cbits[n]=b*ncoarse/max(len(range(0,len(seg),K)),1)
  xh={n:XF[n].copy() for n in SETS}
  fine={n:[] for n in SETS}
  t0=time.time(); nb=N//stride
  for b in range(nb):
    t=START+b*stride
    ctxs=[xh[n][t-CTX:t].astype(np.float32) for n in SETS]
    pfc=[cov[n][t-CTX:t+max(stride,H)].astype(np.float32)[None,:] for n in SETS]
    outs=list(fc.predict_batch(contexts=ctxs,horizon=max(stride,H),
              past_future_covariates=pfc,return_quantiles=False,sort_quantiles=True))
    for j,n in enumerate(SETS):
      pred=np.asarray(outs[j].forecast,dtype=np.float64)[:stride]
      idx=np.arange(t,t+stride)
      mask=(idx%K)!=0                       # coarse samples already sent
      D=2*taus[n]
      k=np.rint((XF[n][idx]-pred)/D)
      xh[n][idx]=pred+k*D
      # coarse positions are restored exactly from L0, not re-coded
      cpos=idx[~mask]
      if len(cpos): xh[n][cpos]=cov[n][cpos]
      fine[n].append(k[mask])
  dt=time.time()-t0
  out={}
  for n in SETS:
    q=np.concatenate(fine[n])
    err=np.abs(xh[n][START:START+N]-XF[n][START:START+N]).max()
    fb=adaptive_resid_bits(q,max(abs(q).max()*2+1,3))*len(q)
    out[n]=dict(bpv=(fb+cbits[n])/N, fine_bpv=fb/N, coarse_bpv=cbits[n]/N,
                maxerr=float(err), tau=taus[n], vps=N*len(SETS)/dt)
  return out

if __name__=="__main__":
  h2h=json.load(open("results/headtohead.json"))
  RES={}
  print("TimesFM-as-INTERPOLATOR (coarse grid K as past_future_covariate)")
  print(f"{'dataset':17s}{'rho':>6s}{'K':>4s}{'stride':>7s}{'TFMinterp':>10s}"
        f"{'TFMextrap':>10s}{'BESTclassic':>12s}{'gain%':>8s}{'val/s':>8s}")
  print("-"*90)
  for rho in [0.01,0.05]:
    for K,st in [(16,64),(8,64),(16,32)]:
      r=run(rho=rho,K=K,stride=st)
      for n in SETS:
        hk=f"{n}|{rho}"; bc=h2h[hk]["bestv"]; te=h2h[hk]["tfm"]
        g=100*(1-r[n]["bpv"]/bc); RES[f"{n}|{rho}|{K}|{st}"]=dict(**r[n],gain=g,bestclassic=bc)
        print(f"{n:17s}{rho:6.3f}{K:4d}{st:7d}{r[n]['bpv']:10.3f}{te:10.3f}{bc:12.3f}"
              f"{g:+8.1f}{r[n]['vps']:8.0f}")
      print()
  json.dump(RES,open("results/tfm_interp.json","w"),indent=1)
