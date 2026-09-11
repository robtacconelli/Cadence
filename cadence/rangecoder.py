"""Adaptive binary arithmetic coder (LZMA-style) + context-modelled binarization
of quantization indices.

Why: every general-purpose back end (xz/zstd) forced us to keep the index stream
contiguous, because splitting it fragmented the compressor. That confound
invalidated two experiments (E26 hybrid switching, E44 heteroscedastic coding).
A conditional arithmetic coder removes the problem at its root: one stream, many
models, selected per symbol by a decoder-derivable context. Nothing is split.

Binarization of a signed index k (CABAC-like):
    zero flag        -> context-coded, ctx = f(recent magnitudes, spread bucket)
    sign             -> bypass (residual signs are ~symmetric)
    magnitude-1      -> truncated unary, context-coded prefix (6 bins)
    tail             -> Exp-Golomb bypass
"""
import numpy as np

KTOP=1<<24; PBITS=11; PMAX=1<<PBITS; PINIT=PMAX//2; MOVE=5

class Enc:
  def __init__(self):
    self.low=0; self.range=0xFFFFFFFF; self.cache=0; self.csize=1; self.out=bytearray()
  def _shift(self):
    if self.low < 0xFF000000 or self.low > 0xFFFFFFFF:
      c=self.cache
      while True:
        self.out.append((c + (self.low>>32)) & 0xFF)
        c=0xFF; self.csize-=1
        if self.csize==0: break
      self.cache=(self.low>>24)&0xFF
    self.csize+=1; self.low=(self.low<<8)&0xFFFFFFFF
  def bit(self,p,b):
    bound=(self.range>>PBITS)*p[0]
    if b==0: self.range=bound; p[0]+=(PMAX-p[0])>>MOVE
    else:    self.low+=bound; self.range-=bound; p[0]-=p[0]>>MOVE
    while self.range<KTOP: self._shift(); self.range=(self.range<<8)&0xFFFFFFFF
  def raw(self,b):
    self.range>>=1
    if b: self.low+=self.range
    while self.range<KTOP: self._shift(); self.range=(self.range<<8)&0xFFFFFFFF
  def finish(self):
    for _ in range(5): self._shift()
    return bytes(self.out)

class Dec:
  def __init__(self,data):
    self.data=data; self.pos=1; self.range=0xFFFFFFFF; self.code=0
    for _ in range(4):
      self.code=(self.code<<8)|(self.data[self.pos] if self.pos<len(self.data) else 0); self.pos+=1
  def _nb(self):
    b=self.data[self.pos] if self.pos<len(self.data) else 0; self.pos+=1; return b
  def bit(self,p):
    bound=(self.range>>PBITS)*p[0]
    if self.code<bound: self.range=bound; p[0]+=(PMAX-p[0])>>MOVE; b=0
    else: self.code-=bound; self.range-=bound; p[0]-=p[0]>>MOVE; b=1
    while self.range<KTOP:
      self.range=(self.range<<8)&0xFFFFFFFF; self.code=((self.code<<8)|self._nb())&0xFFFFFFFF
    return b
  def raw(self):
    self.range>>=1
    if self.code>=self.range: self.code-=self.range; b=1
    else: b=0
    while self.range<KTOP:
      self.range=(self.range<<8)&0xFFFFFFFF; self.code=((self.code<<8)|self._nb())&0xFFFFFFFF
    return b

NMAG=6; NCTX_MAG=3
class Model:
  """Contexts are decoder-derivable: recent magnitudes, and optionally a
  spread bucket supplied by the predictor. Nothing is transmitted."""
  def __init__(self,nspread=1):
    self.ns=nspread
    self.zero=[[PINIT] for _ in range(nspread*4)]
    self.mag=[[PINIT] for _ in range(nspread*NCTX_MAG*NMAG)]
    self.eg=[[PINIT] for _ in range(nspread*12)]
    self.hist=0.0
  def _c(self):                      # recent-magnitude bucket, 0..3
    h=self.hist
    return 0 if h<0.5 else 1 if h<1.5 else 2 if h<4 else 3
  def _cm(self):
    h=self.hist
    return 0 if h<1.0 else 1 if h<4.0 else 2
  def upd(self,k): self.hist += (abs(k)-self.hist)/32.0

def _eg_bits(v):
  """Exp-Golomb order 0 of v>=0 -> list of bits."""
  v=int(v)+1; n=v.bit_length()-1
  return [0]*n + [1] + [(v>>(n-1-i))&1 for i in range(n)] if n else [1]

def encode(idx, spread_bucket=None, nspread=1):
  e=Enc(); m=Model(nspread)
  sb = spread_bucket if spread_bucket is not None else np.zeros(len(idx),dtype=int)
  for t,k in enumerate(idx):
    k=int(k); s=int(sb[t]); c=m._c()
    e.bit(m.zero[s*4+c], 0 if k==0 else 1)
    if k!=0:
      e.raw(0 if k>0 else 1)
      a=abs(k)-1; cm=m._cm()
      j=0
      while j<NMAG-1 and a>j:
        e.bit(m.mag[(s*NCTX_MAG+cm)*NMAG+j],1); j+=1
      if a<NMAG-1: e.bit(m.mag[(s*NCTX_MAG+cm)*NMAG+a],0)
      else:
        for b in _eg_bits(a-(NMAG-1)): e.raw(b)
    m.upd(k)
  return e.finish()

def decode(blob, n, spread_bucket=None, nspread=1):
  d=Dec(blob); m=Model(nspread); out=np.empty(n,dtype=np.int64)
  sb = spread_bucket if spread_bucket is not None else np.zeros(n,dtype=int)
  for t in range(n):
    s=int(sb[t]); c=m._c()
    if d.bit(m.zero[s*4+c])==0: k=0
    else:
      neg=d.raw(); cm=m._cm(); j=0
      while j<NMAG-1 and d.bit(m.mag[(s*NCTX_MAG+cm)*NMAG+j])==1: j+=1
      if j<NMAG-1: a=j
      else:
        nz=0
        while d.raw()==0: nz+=1
        v=1
        for _ in range(nz): v=(v<<1)|d.raw()
        a=(v-1)+(NMAG-1)
      k=(a+1)*(-1 if neg else 1)
    out[t]=k; m.upd(k)
  return out

if __name__=="__main__":
  rng=np.random.default_rng(0)
  print("round-trip validation")
  for name,gen in [("laplace", lambda: np.rint(rng.laplace(0,3,4000))),
                   ("sparse",  lambda: np.rint(rng.laplace(0,0.4,4000))),
                   ("heavy",   lambda: np.rint(rng.standard_t(2,4000)*8)),
                   ("zeros",   lambda: np.zeros(4000)),
                   ("uniform", lambda: rng.integers(-200,200,4000))]:
    a=gen().astype(np.int64); b=encode(a); r=decode(b,len(a))
    ok=np.array_equal(a,r)
    print(f"  {name:9s} n={len(a)} bytes={len(b):6d} bpv={len(b)*8/len(a):6.3f}  round-trip={ok}")
    assert ok, name
  print("  ALL PASS")
