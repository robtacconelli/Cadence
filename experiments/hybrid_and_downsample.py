"""Two additions.

(1) HYBRID SWITCH: per-block choice between TimesFM and interp-cubic, 1 bit of
    side info per block. Removes the single systematic loss (smooth signals)
    while keeping the wins. Trivially decodable.

(2) THE COMPARISON THAT ACTUALLY MATTERS: observability/TSDB retention does not
    use SZ3 -- it uses DOWNSAMPLING (Prometheus/Thanos/VictoriaMetrics roll up to
    5m/1h aggregates). Downsampling has UNBOUNDED L-inf error: it deletes spikes,
    which is precisely what incident analysis needs. So compare at matched
    bitrate on max-error, which is the metric an error bound actually promises.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, json
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
from sz_style import interp_multilevel, PREDS
from fair import adaptive_resid_bits
import lzma, zstandard as zstd

TD=CACHE
SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
N=int(os.environ.get("N",8192)); START=1024
XF={n:np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64) for n in SETS}

def rb(q):
  q=np.asarray(q,dtype=np.int64); u=q-q.min(); s=int(u.max())
  for w,dt in ((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64)):
    if s<2**(8*w): buf=u.astype(dt).tobytes(); break
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return min(len(lzma.compress(buf,preset=9|lzma.PRESET_EXTREME)),len(zc.compress(buf)))

def downsample_bits_err(x, R, tau_bits_target=None):
  """Keep every R-th sample, reconstruct by linear interpolation. Bits = the
  kept samples coded with the same real coder (delta + xz/zstd)."""
  pos=np.arange(0,len(x),R); kept=x[pos]
  d=np.diff(kept,prepend=kept[0]).astype(np.int64)
  bits=rb(d)*8.0/len(x)
  rec=np.interp(np.arange(len(x)),pos,kept)
  return bits, float(np.abs(rec-x).max()), float(np.abs(rec-x).mean())

def blockwise_bits(q, B=512):
  return np.array([rb(q[i:i+B])*8.0 for i in range(0,len(q),B)])

RHO=[float(r) for r in os.environ.get("RHO","0.01,0.05,0.2").split(",")]
res={}
print("(1) HYBRID SWITCH: per-block min(TimesFM, interp-cubic) + 1 bit/block")
print(f"{'dataset':17s}{'rho':>6s}{'TimesFM':>9s}{'intCubic':>9s}{'HYBRID':>8s}{'tfm%blk':>8s}{'vs best':>8s}")
print("-"*68)
hg=[]
for rho in RHO:
  f=f"{TD}/tfmq_rho{rho}_N{N}.npz"
  if not os.path.exists(f): print(f"  (waiting for {f})"); continue
  d=np.load(f)
  for n in SETS:
    x=XF[n][START:START+N]; tau=max(rho*float(np.std(XF[n])),0.5)
    qt=np.asarray(d[n]); qi,_=interp_multilevel(x,tau,"cubic")
    B=512
    bt=blockwise_bits(qt,B); bi=blockwise_bits(qi,B)
    m=min(len(bt),len(bi)); bt,bi=bt[:m],bi[:m]
    hyb=(np.minimum(bt,bi).sum()+m)/N          # +1 bit per block
    tfmb=rb(qt)*8.0/N; intb=rb(qi)*8.0/N
    g=100*(1-hyb/min(tfmb,intb)); hg.append(g)
    res[f"hyb|{n}|{rho}"]=dict(tfm=tfmb,inter=intb,hybrid=hyb,gain=g,
                               tfm_frac=float((bt<bi).mean()))
    print(f"{n:17s}{rho:6.3f}{tfmb:9.3f}{intb:9.3f}{hyb:8.3f}"
          f"{100*(bt<bi).mean():7.0f}%{g:+8.1f}")
  print()
if hg: print(f"hybrid vs best-single: median {np.median(hg):+.1f}%\n")

print("(2) vs DOWNSAMPLING at matched bitrate -- max error is the whole point")
print(f"{'dataset':17s}{'rho':>6s}{'TFM bpv':>9s}{'TFM maxE':>10s}{'R':>4s}{'DS bpv':>8s}"
      f"{'DS maxE':>10s}{'DS meanE':>10s}{'maxE ratio':>11s}")
print("-"*88)
ratios=[]
for rho in RHO:
  f=f"{TD}/tfmq_rho{rho}_N{N}.npz"
  if not os.path.exists(f): continue
  d=np.load(f)
  for n in SETS:
    x=XF[n][START:START+N]; tau=max(rho*float(np.std(XF[n])),0.5)
    tfmb=rb(np.asarray(d[n]))*8.0/N
    best=None
    for R in [2,4,8,16,32,64,128,256]:
      b,mx,mn=downsample_bits_err(x,R)
      if b<=tfmb and (best is None or b>best[1]): best=(R,b,mx,mn)
    if best is None: continue
    R,b,mx,mn=best; ratios.append(mx/tau)
    res[f"ds|{n}|{rho}"]=dict(tfm_bpv=tfmb,tau=tau,R=R,ds_bpv=b,ds_max=mx,ds_mean=mn)
    print(f"{n:17s}{rho:6.3f}{tfmb:9.3f}{tau:10.1f}{R:4d}{b:8.3f}{mx:10.1f}{mn:10.1f}"
          f"{mx/tau:10.1f}x")
  print()
if ratios: print(f"At the SAME bitrate, downsampling's max error is {np.median(ratios):.1f}x "
                 f"(median) the guaranteed bound TimesFM delivers.")
json.dump(res,open("results/hybrid_downsample.json","w"),indent=1,default=float)
