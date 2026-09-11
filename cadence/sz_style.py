"""Predictor-isolated comparison: same quantizer, same ideal entropy coder,
only the PREDICTOR varies. This is the scientific question.

Important refinement to my own mechanism story: SZ3's default mode is NOT causal
extrapolation, it is multilevel INTERPOLATION -- predict a midpoint from
already-decoded neighbours on both sides. Analytic noise gains:

    Lorenzo-1  pred = x[i-1]                          G = 1.00
    Lorenzo-2  pred = 2x[i-1] - x[i-2]                G = 2.24   (amplifies)
    Lorenzo-3  pred = 3x[i-1] -3x[i-2] + x[i-3]       G = 4.36   (amplifies)
    interp-lin pred = (x[i-h] + x[i+h]) / 2           G = 0.71   (contractive)
    interp-cub 4-point midpoint cubic                 G = 0.80   (contractive)
    LPC-32     pred = c . x[i-32:i]                   G = ||c||2 (usually >> 1)

So SZ3 ALREADY exploits contractiveness -- it is not a gap nobody noticed. The
sharpened claim TimesFM has to defend is: contractive AND far more accurate than
a local interpolator. That is what this file measures.
"""
import numpy as np, sys
sys.path.insert(0,"src")
from fair import adaptive_resid_bits

def _bits(q):
  q=np.asarray(q,dtype=np.float64)
  return adaptive_resid_bits(q, max(abs(q).max()*2+1, 3))

def lorenzo(x, tau, order=1):
  D=2*tau; n=len(x); xh=np.zeros(n); q=np.empty(n)
  C={1:[1.0],2:[2.0,-1.0],3:[3.0,-3.0,1.0]}[order]
  for i in range(n):
    p=0.0
    for j,c in enumerate(C):
      if i-1-j>=0: p+=c*xh[i-1-j]
    k=np.rint((x[i]-p)/D); xh[i]=p+k*D; q[i]=k
  return q,xh

def lpc_closed(x, tau, order=32):
  D=2*tau; n=len(x)
  Xf=np.lib.stride_tricks.sliding_window_view(x,order)[:-1]
  c,*_=np.linalg.lstsq(Xf,x[order:],rcond=None); c=np.rint(c*16384)/16384
  xh=np.zeros(n); q=np.empty(n)
  for i in range(n):
    p=float(xh[max(0,i-order):i]@c[order-min(i,order):]) if i>0 else 0.0
    k=np.rint((x[i]-p)/D); xh[i]=p+k*D; q[i]=k
  return q,xh

def interp_multilevel(x, tau, kind="cubic"):
  """SZ3-style: code a coarse anchor grid, then refine midpoints level by level."""
  D=2*tau; n=len(x); xh=np.zeros(n); q=[]
  L=max(int(np.floor(np.log2(max(n-1,2)))),1); s=2**L
  prev=0.0
  for i in range(0,n,s):                       # anchors: Lorenzo-1 on coarse grid
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

def gain_analytic(kind):
  return {"lorenzo1":1.0,"lorenzo2":np.sqrt(5),"lorenzo3":np.sqrt(19),
          "interp_linear":np.sqrt(0.5),"interp_cubic":np.sqrt(2*(1/16)**2+2*(9/16)**2)}[kind]

PREDS={"lorenzo1":lambda x,t:lorenzo(x,t,1),"lorenzo2":lambda x,t:lorenzo(x,t,2),
       "lorenzo3":lambda x,t:lorenzo(x,t,3),"lpc32":lambda x,t:lpc_closed(x,t,32),
       "interp_linear":lambda x,t:interp_multilevel(x,t,"linear"),
       "interp_cubic":lambda x,t:interp_multilevel(x,t,"cubic")}

if __name__=="__main__":
  import json
  SETS=["ecg_like","sparse_spiky","seasonal_metric","lorenz_chaotic","wikihr_en","wikihr_de"]
  RHO=[0.001,0.01,0.05,0.2]; N=4096; out={}
  print("Analytic noise gains:", {k:round(gain_analytic(k),3) for k in
        ["lorenzo1","lorenzo2","lorenzo3","interp_linear","interp_cubic"]})
  print(f"\n{'dataset':17s}{'rho':>7s}"+"".join(f"{k[:11]:>13s}" for k in PREDS)+f"{'BEST':>15s}")
  print("-"*118)
  for n in SETS:
    x=np.load(f"data/{'real' if n.startswith('wiki') else 'synth'}/{n}.npy").astype(np.float64)
    x=x[1024:1024+N]; sd=float(np.std(x))
    for rho in RHO:
      tau=max(rho*sd,0.5); row={}
      for k,f in PREDS.items():
        q,xh=f(x,tau)
        assert np.abs(xh-x).max()<=tau+1e-6,(n,k,np.abs(xh-x).max(),tau)
        row[k]=_bits(q)
      bk=min(row,key=row.get); out[f"{n}|{rho}"]=row
      print(f"{n:17s}{rho:7.3f}"+"".join(f"{row[k]:13.3f}" for k in PREDS)+f"{bk:>15s}")
    print()
  json.dump(out,open("results/sz_style.json","w"),indent=1)
