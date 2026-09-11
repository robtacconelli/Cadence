#!/usr/bin/env python3
"""End-to-end codec demo: encode -> bytes -> decode -> verify bit-exactness.

Reproduces Table 8. Verifies that the decoder's reconstruction is byte-identical
to the encoder's (SHA-256) and that |x_hat - x| <= tau on every sample.

  python experiments/codec_demo.py            # G=16, n=4300, rho=0.05 (paper config)
  G=8 N=1200 python experiments/codec_demo.py # faster smoke test
"""
import os as _os, sys, glob, time, hashlib
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
import numpy as np
from codec import Codec

DATA=_os.environ.get("CADENCE_DATA",_os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),"data"))
G=int(_os.environ.get("G",16)); N=int(_os.environ.get("N",4300)); RHO=float(_os.environ.get("RHO",0.05))
cand=sorted(_os.path.basename(f)[:-4] for f in glob.glob(f"{DATA}/grid/*.npy"))
cand=[c for c in cand if c!="SEC" and len(np.load(f"{DATA}/grid/{c}.npy"))>=N]
if len(cand)<G: sys.exit(f"need {G} grid series of length >= {N}; run scripts/fetch_data.py grid")
pri=[n for n in ["MISO","ERCO","PJM","CISO","BPAT","SOCO","DUK","AZPS"] if n in cand]
NAMES=(pri+[c for c in cand if c not in pri])[:G]
X=[np.load(f"{DATA}/grid/{n}.npy")[:N] for n in NAMES]
taus=[max(RHO*float(np.std(np.load(f"{DATA}/grid/{n}.npy").astype(float))),0.5) for n in NAMES]

c=Codec(G=G)
t0=time.time(); blob,xe=c.encode(X,taus); es=time.time()-t0
t0=time.time(); xd=c.decode(blob); ds=time.time()-t0
sb=sum(s[1] for s in c.last_seed)+40*G; bb=sum(c.last_body)+24*G
print(f"group={G} n={N} rho={RHO}   encode {es:.0f}s  decode {ds:.0f}s  ({G*(N-c.ctx)/es:.0f} val/s)")
print(f"container {len(blob)} B = {len(blob)*8/(G*N):.3f} bits/value   ratio {32*G*N/(len(blob)*8):.1f}x")
print(f"  seed {sb*8/(G*N):.3f} bpv ({100*sb/len(blob):.0f}%)   body {bb*8/(G*(N-c.ctx)):.3f} bpv")
ok=True
for g,n in enumerate(NAMES):
    ex=np.array_equal(xe[g],xd[g]); er=np.abs(xd[g]-X[g].astype(np.float64)).max()
    w=er<=taus[g]+1e-9; ok&=ex and w
    if g<4 or not (ex and w):
        print(f"  {n:6s} dec==enc {str(ex):>5s}  max|x_hat-x| {er:9.2f} (bound {taus[g]:9.2f}) ok={w}")
he=hashlib.sha256(np.concatenate(xe).tobytes()).hexdigest()[:16]
hd=hashlib.sha256(np.concatenate(xd).tobytes()).hexdigest()[:16]
print(f"\nROUND TRIP {'PASS' if ok else 'FAIL'}   sha256 enc {he}  dec {hd}")
sys.exit(0 if ok else 1)
