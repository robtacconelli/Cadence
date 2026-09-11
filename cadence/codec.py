"""Error-bounded lossy codec with TimesFM, with a real encode->bytes->decode path.

Design constraints established by measurement:
  * batch-size invariance is impossible (results/determinism2.json), so GROUP
    SIZE G is part of the format; encoder and decoder both batch exactly G.
  * the 512-sample context bootstrap is a cost unique to the neural codec
    (Lorenzo-1 needs 1 seed sample). It is therefore coded LOSSILY at the same
    tolerance tau using a classical predictor that needs no side information,
    picking the best of 5 per series and storing a 1-byte id.
  * quantization indices are packed to the smallest int width after offset
    subtraction before entropy coding (raw int64 wasted 7/8 bytes).
"""
import numpy as np, sys, os, lzma, hashlib, time, glob
sys.path.insert(0,"src")
import zstandard as zstd
from timesfm3 import TimesFM3Forecaster, ModelConfig
import rangecoder as rc

CTX=512; H=64; MAGIC=b"TFMC2"

# ---------- seed predictors: pure, decodable with no side information ----------
_LC={1:[1.0],2:[2.0,-1.0],3:[3.0,-3.0,1.0]}

def _lorenzo_enc(x,tau,order):
  D=2*tau; n=len(x); xh=np.zeros(n); q=np.empty(n); C=_LC[order]
  for i in range(n):
    p=0.0
    for j,c in enumerate(C):
      if i-1-j>=0: p+=c*xh[i-1-j]
    k=np.rint((x[i]-p)/D); xh[i]=p+k*D; q[i]=k
  return q,xh

def _lorenzo_dec(q,tau,order):
  D=2*tau; n=len(q); xh=np.zeros(n); C=_LC[order]
  for i in range(n):
    p=0.0
    for j,c in enumerate(C):
      if i-1-j>=0: p+=c*xh[i-1-j]
    xh[i]=p+q[i]*D
  return xh

def _interp_enc(x,tau,kind):
  D=2*tau; n=len(x); xh=np.zeros(n); q=[]
  L=max(int(np.floor(np.log2(max(n-1,2)))),1); s=2**L; prev=0.0
  for i in range(0,n,s):
    k=np.rint((x[i]-prev)/D); xh[i]=prev+k*D; q.append(k); prev=xh[i]
  while s>1:
    h=s//2
    for i in range(h,n,s):
      l,r=i-h,i+h
      if r>=n: p=xh[l]
      elif kind=="linear" or i-3*h<0 or i+3*h>=n: p=0.5*(xh[l]+xh[r])
      else: p=(-xh[i-3*h]+9*xh[l]+9*xh[r]-xh[i+3*h])/16.0
      k=np.rint((x[i]-p)/D); xh[i]=p+k*D; q.append(k)
    s=h
  return np.array(q),xh

def _interp_dec(q,tau,n,kind):
  D=2*tau; xh=np.zeros(n); qi=0
  L=max(int(np.floor(np.log2(max(n-1,2)))),1); s=2**L; prev=0.0
  for i in range(0,n,s):
    xh[i]=prev+q[qi]*D; qi+=1; prev=xh[i]
  while s>1:
    h=s//2
    for i in range(h,n,s):
      l,r=i-h,i+h
      if r>=n: p=xh[l]
      elif kind=="linear" or i-3*h<0 or i+3*h>=n: p=0.5*(xh[l]+xh[r])
      else: p=(-xh[i-3*h]+9*xh[l]+9*xh[r]-xh[i+3*h])/16.0
      xh[i]=p+q[qi]*D; qi+=1
    s=h
  return xh

SEED_ENC=[lambda x,t:_lorenzo_enc(x,t,1), lambda x,t:_lorenzo_enc(x,t,2),
          lambda x,t:_lorenzo_enc(x,t,3), lambda x,t:_interp_enc(x,t,"linear"),
          lambda x,t:_interp_enc(x,t,"cubic")]
SEED_DEC=[lambda q,t,n:_lorenzo_dec(q,t,1), lambda q,t,n:_lorenzo_dec(q,t,2),
          lambda q,t,n:_lorenzo_dec(q,t,3), lambda q,t,n:_interp_dec(q,t,n,"linear"),
          lambda q,t,n:_interp_dec(q,t,n,"cubic")]
SEED_NAME=["lorenzo1","lorenzo2","lorenzo3","interp_lin","interp_cub"]

# ---------- packing ----------
_DT=[np.uint8,np.uint16,np.uint32,np.uint64]
def _pack(a):
  a=np.asarray(a,dtype=np.int64); lo=int(a.min()); u=a-lo; span=int(u.max())
  for code,(w,dt) in enumerate(((1,np.uint8),(2,np.uint16),(4,np.uint32),(8,np.uint64))):
    if span < 2**(8*w): return u.astype(dt).tobytes(), lo, code
  raise ValueError
def _unpack(buf,lo,code,n):
  return np.frombuffer(buf,dtype=_DT[code],count=n).astype(np.int64)+lo
def _sq(b):
  zc=zstd.ZstdCompressor(level=22,write_content_size=False)
  return min(lzma.compress(b,preset=9|lzma.PRESET_EXTREME), zc.compress(b), key=len)
def _unsq(c,maxout):
  try: return lzma.decompress(c)
  except lzma.LZMAError: return zstd.ZstdDecompressor().decompress(c,max_output_size=maxout)


class Codec:
  def __init__(self, G=8, ctx=CTX, device="cuda"):
    self.G=G; self.ctx=ctx
    self.fc=TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                                           per_core_batch_size=G, device=device))
  def _pred(self, w):
    assert len(w)==self.G
    o=list(self.fc.predict_batch(contexts=w,horizon=H,return_quantiles=False,sort_quantiles=True))
    return np.array([float(np.asarray(r.forecast)[0]) for r in o])

  def encode(self, series, taus):
    G=self.G; assert len(series)==G
    seed=self.ctx; n=len(series[0])
    parts=[MAGIC, np.array([G,n,seed,self.ctx],dtype=np.int64).tobytes(),
           np.array(taus,dtype=np.float64).tobytes()]
    xh=[]; seedinfo=[]
    for g in range(G):
      x=np.asarray(series[g],dtype=np.float64)
      best=None
      for sid,enc in enumerate(SEED_ENC):
        q,r=enc(x[:seed],taus[g])
        c=rc.encode(np.asarray(q,dtype=np.int64))
        if best is None or len(c)<len(best[0]): best=(c,0,0,sid,r,len(q))
      c,lo,code,sid,rec,ql=best
      a=x.copy(); a[:seed]=rec                      # model context = RECONSTRUCTED seed
      xh.append(a); seedinfo.append((sid,len(c),SEED_NAME[sid]))
      parts.append(np.array([len(c),lo,code,sid,ql],dtype=np.int64).tobytes()); parts.append(c)
    Q=[[] for _ in range(G)]
    for t in range(seed,n):
      p=self._pred([xh[g][t-self.ctx:t].astype(np.float32) for g in range(G)])
      for g in range(G):
        D=2*taus[g]; k=np.rint((series[g][t]-p[g])/D)
        xh[g][t]=p[g]+k*D; Q[g].append(k)
    body=[]
    for g in range(G):
      c=rc.encode(np.asarray(Q[g],dtype=np.int64)); body.append(len(c)); lo=code=0
      parts.append(np.array([len(c),lo,code],dtype=np.int64).tobytes()); parts.append(c)
    self.last_seed=seedinfo; self.last_body=body
    return b"".join(parts), xh

  def decode(self, blob):
    off=len(MAGIC); assert blob[:off]==MAGIC
    G,n,seed,ctx=[int(v) for v in np.frombuffer(blob[off:off+32],dtype=np.int64)]; off+=32
    taus=np.frombuffer(blob[off:off+8*G],dtype=np.float64).copy(); off+=8*G
    assert G==self.G, f"group size {G} != codec group {self.G}"
    xh=[]
    for g in range(G):
      L,lo,code,sid,ql=[int(v) for v in np.frombuffer(blob[off:off+40],dtype=np.int64)]; off+=40
      c=blob[off:off+L]; off+=L
      q=rc.decode(c,ql).astype(np.float64)
      a=np.zeros(n); a[:seed]=SEED_DEC[sid](q,taus[g],seed); xh.append(a)
    Q=[]
    for g in range(G):
      L,lo,code=[int(v) for v in np.frombuffer(blob[off:off+24],dtype=np.int64)]; off+=24
      c=blob[off:off+L]; off+=L
      Q.append(rc.decode(c,n-seed).astype(np.float64))
    for t in range(seed,n):
      p=self._pred([xh[g][t-ctx:t].astype(np.float32) for g in range(G)])
      for g in range(G): xh[g][t]=p[g]+Q[g][t-seed]*2*taus[g]
    return xh


if __name__=="__main__":
  G=int(os.environ.get("G",16)); N=int(os.environ.get("N",4300)); RHO=float(os.environ.get("RHO",0.05))
  cand=sorted(os.path.basename(f)[:-4] for f in glob.glob("data/grid/*.npy"))
  cand=[c for c in cand if c!="SEC" and len(np.load(f"data/grid/{c}.npy"))>=N]
  pri=[n for n in ["MISO","ERCO","PJM","CISO","BPAT","SOCO","DUK","AZPS"] if n in cand]
  NAMES=(pri+[c for c in cand if c not in pri])[:G]; assert len(NAMES)==G
  X=[np.load(f"data/grid/{n}.npy")[:N] for n in NAMES]
  taus=[max(RHO*float(np.std(np.load(f"data/grid/{n}.npy").astype(float))),0.5) for n in NAMES]
  c=Codec(G=G)
  t0=time.time(); blob,xe=c.encode(X,taus); es=time.time()-t0
  t0=time.time(); xd=c.decode(blob); ds=time.time()-t0
  sb=sum(s[1] for s in c.last_seed)+40*G; bb=sum(c.last_body)+24*G
  print(f"group={G} n={N} rho={RHO}   encode {es:.0f}s   decode {ds:.0f}s   "
        f"({G*(N-c.ctx)/es:.0f} val/s enc)")
  print(f"container {len(blob)} B = {len(blob)*8/(G*N):.3f} bits/value   ratio {32*G*N/(len(blob)*8):.1f}x")
  print(f"  seed  {sb:6d} B = {sb*8/(G*N):.3f} bpv ({100*sb/len(blob):.0f}% of file)  "
        f"predictors used: {sorted(set(s[2] for s in c.last_seed))}")
  print(f"  body  {bb:6d} B = {bb*8/(G*(N-c.ctx)):.3f} bpv over coded samples\n")
  ok=True
  for g,n in enumerate(NAMES):
    ex=np.array_equal(xe[g],xd[g]); er=np.abs(xd[g]-X[g].astype(np.float64)).max()
    w=er<=taus[g]+1e-9; ok&=ex and w
    if g<6 or not (ex and w):
      print(f"  {n:6s} dec==enc: {str(ex):>5s}  max|x_hat-x| {er:9.2f} (bound {taus[g]:9.2f}) within: {w}")
  print(f"\nROUND TRIP {'PASS' if ok else 'FAIL'}  ({G}/{G} series verified)")
  print(f"sha256 enc {hashlib.sha256(np.concatenate(xe).tobytes()).hexdigest()[:16]}  "
        f"dec {hashlib.sha256(np.concatenate(xd).tobytes()).hexdigest()[:16]}")
