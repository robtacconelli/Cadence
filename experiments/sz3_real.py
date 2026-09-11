"""Absolute grounding: run the REAL SZ3 binary and zfpy end-to-end.
Reports actual compressed bytes, so my ideal-codelength numbers can be checked
for a systematic offset (my coder is idealized; SZ3 uses Huffman+zstd)."""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, subprocess, os, tempfile, json, sys
import zfpy
SZ3=os.path.abspath("ext/sz3-install/bin/sz3")
SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
RHO=[0.001,0.01,0.05,0.2]; N=2048; START=1024
TD=CACHE
os.makedirs(TD,exist_ok=True)
env=dict(os.environ, LD_LIBRARY_PATH=os.path.abspath("ext/sz3-install/lib"))

def sz3_bytes(x, tau):
  f=f"{TD}/in.f32"; x.astype(np.float32).tofile(f)
  # -M ABS -A <tau>: absolute error bound
  r=subprocess.run([SZ3,"-f","-i",f,"-z",f+".sz","-1",str(len(x)),"-M","ABS","-A",str(tau)],
                   capture_output=True,env=env,text=True)
  if r.returncode!=0: return None, r.stderr[:200]
  nb=os.path.getsize(f+".sz")
  # verify the bound on the actual decompression
  subprocess.run([SZ3,"-f","-z",f+".sz","-o",f+".out","-1",str(len(x)),"-M","ABS","-A",str(tau)],
                 capture_output=True,env=env)
  y=np.fromfile(f+".out",dtype=np.float32)
  return nb*8.0/len(x), float(np.abs(y-x.astype(np.float32)).max())

def zfp_bytes(x, tau):
  c=zfpy.compress_numpy(np.ascontiguousarray(x.astype(np.float64)),tolerance=tau)
  y=zfpy.decompress_numpy(c)
  return len(c)*8.0/len(x), float(np.abs(y-x).max())

h2h=json.load(open("results/headtohead.json"))
print(f"{'dataset':17s}{'rho':>7s}{'tau':>10s}{'SZ3':>9s}{'err/tau':>9s}{'ZFP':>9s}{'err/tau':>9s}"
      f"{'myBEST':>9s}{'TimesFM':>9s}{'vs SZ3':>8s}")
print("-"*100)
res={}
for n in SETS:
  xf=np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64)
  x=xf[START:START+N]; sd=float(np.std(xf))
  for rho in RHO:
    tau=max(rho*sd,0.5)
    s=sz3_bytes(x,tau); z=zfp_bytes(x,tau)
    hk=f"{n}|{rho}"; mb=h2h[hk]["bestv"]; tf=h2h[hk]["tfm"]
    sv=s[0] if s[0] else float("nan")
    g=100*(1-tf/sv) if sv==sv else float("nan")
    res[hk]=dict(sz3=sv,sz3err=s[1],zfp=z[0],zfperr=z[1],mybest=mb,tfm=tf,gain_vs_sz3=g)
    print(f"{n:17s}{rho:7.3f}{tau:10.1f}{sv:9.3f}{(s[1]/tau if s[1] else 0):9.2f}"
          f"{z[0]:9.3f}{z[1]/tau:9.2f}{mb:9.3f}{tf:9.3f}{g:+8.1f}")
  print()
json.dump(res,open("results/sz3_real.json","w"),indent=1)
import numpy as _n
g=[v["gain_vs_sz3"] for v in res.values() if v["gain_vs_sz3"]==v["gain_vs_sz3"]]
print(f"TimesFM vs REAL SZ3: median {_n.median(g):+.1f}%  mean {_n.mean(g):+.1f}%  wins {sum(1 for i in g if i>0)}/{len(g)}")
