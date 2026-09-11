"""Does the arithmetic coder beat xz/zstd on real quantization indices, and does
conditioning on predicted spread help once fragmentation is impossible?

E44 answered the second question with idealized code lengths (+0.3%) and with a
bucket-splitting implementation (-66%, pure fragmentation). This settles it with
a real coder: one stream, contexts selected per symbol, nothing split.
"""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, sys, os, glob, json, lzma, time
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import zstandard as zstd
import rangecoder as rc

TD=CACHE
LN5=np.log(5.0); NS=8

def genpurpose(q):
  q=np.asarray(q,dtype=np.int64); u=q-q.min(); s=int(u.max())
  for w,dt in ((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64)):
    if s<2**(8*w): buf=u.astype(dt).tobytes(); break
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return min(len(lzma.compress(buf,preset=9|lzma.PRESET_EXTREME)),len(zc.compress(buf)))+24

def buckets(spread,D,nb=NS):
  sig=np.maximum(spread/max(D,1e-9),1e-9)
  ed=np.quantile(sig,np.linspace(0,1,nb+1)[1:-1])
  return np.searchsorted(ed,sig).astype(int), ed

rows=[]; verified=0
files=sorted(glob.glob(f"{TD}/abl_ctx*_N2048.npz"),
             key=lambda f:int(f.split("ctx")[1].split("_")[0]))
print("Real quantization indices: general-purpose back end vs adaptive AC")
print(f"{'ctx':>5s}{'rho':>7s}{'xz/zstd':>9s}{'AC flat':>9s}{'gain':>8s}"
      f"{'AC +spread':>12s}{'gain':>8s}{'RT':>4s}")
for f in files:
  ctx=int(f.split("ctx")[1].split("_")[0]); d=np.load(f,allow_pickle=True)
  names=[str(x) for x in d["names"]]
  for r in [0.01,0.05,0.2]:
    g=[];a=[];h=[]
    for n in names:
      q=np.asarray(d[f"q|{n}|{r}"],dtype=np.int64)
      u=d[f"u|{n}|{r}"].astype(np.float64); tau=float(d[f"tau|{n}|{r}"])
      sp=(u[:,8]-u[:,0])/(2*LN5); bk,_=buckets(sp,2*tau)
      g.append(genpurpose(q)*8.0/len(q))
      b1=rc.encode(q); a.append((len(b1)+24)*8.0/len(q))
      b2=rc.encode(q,bk,NS); h.append((len(b2)+24+8*NS)*8.0/len(q))
      if verified<6:                        # decode-verify a sample of streams
        assert np.array_equal(rc.decode(b1,len(q)),q)
        assert np.array_equal(rc.decode(b2,len(q),bk,NS),q)
        verified+=1
    G,A,H=np.mean(g),np.mean(a),np.mean(h)
    rows.append(dict(ctx=ctx,rho=r,gp=G,ac=A,ac_spread=H,
                     gain_ac=100*(1-A/G),gain_spread=100*(1-H/A)))
    print(f"{ctx:5d}{r:7.3f}{G:9.3f}{A:9.3f}{100*(1-A/G):+7.1f}%{H:12.3f}"
          f"{100*(1-H/A):+7.1f}%{'ok' if verified else '':>4s}")
ga=np.array([x["gain_ac"] for x in rows]); gs=np.array([x["gain_spread"] for x in rows])
print(f"\nAC vs xz/zstd        : median {np.median(ga):+.1f}%  wins {int((ga>0).sum())}/{len(ga)}")
print(f"spread context vs flat: median {np.median(gs):+.1f}%  wins {int((gs>0).sum())}/{len(gs)}")
print(f"decode-verified streams: {verified}")
json.dump(rows,open("results/ac_bench.json","w"),indent=1,default=float)
