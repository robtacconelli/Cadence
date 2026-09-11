"""Fairness fix.

v2 coded every predictor's residual with an adaptive Laplace. But `iid_noise`
showed a +2.15% 'gain' -- impossible for real noise. The cause: the TimesFM
expert used a flexible piecewise-linear density while the classical expert was
locked to a Laplace, which fits uniform noise badly. That measures density
family, not prediction skill.

Fix: code EVERY predictor's residual with the identical, equally flexible
adaptive coder -- a causal mixture over a grid of Laplace scales plus a uniform
component. Differences then reflect prediction quality alone.
"""
import numpy as np
ESC=1e-4
SCALES=np.array([0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,2**20],dtype=np.float64)

def _lap_pmf(r, b):
  """P(residual r) under discretized Laplace(0,b). r:(M,) b:(S,) -> (S,M)"""
  a=np.abs(r)[None,:]; b=b[:,None]
  hi=(a+0.5)/b; lo=np.maximum((a-0.5)/b,0.0)
  p=0.5*(np.exp(-lo)-np.exp(-hi))
  return np.where(a<0.5, 1.0-np.exp(-hi), p)

def adaptive_resid_bits(resid, span, gamma=0.999, ret_bits=False):
  """Causal mixture-of-scales coder. Identical machinery for every predictor."""
  resid=np.rint(np.asarray(resid,dtype=np.float64))
  M=len(resid)
  P=_lap_pmf(resid, SCALES)                       # (S,M)
  P=np.vstack([P, np.full((1,M), 1.0/span)])      # + uniform component
  P=(1-ESC)*np.clip(P,0,1)+ESC/span
  S=P.shape[0]; w=np.full(S,1.0/S); bits=np.empty(M)
  for i in range(M):
    bits[i]=-np.log2(max(float(w@P[:,i]),1e-300))
    w=w*P[:,i]; w/=w.sum(); w=gamma*w+(1-gamma)/S
  return (bits if ret_bits else float(bits.mean()))

def pmf_adaptive(resid, span, gamma=0.999):
  """Same coder, but returns the per-position probability (for mixing)."""
  return np.exp2(-adaptive_resid_bits(resid,span,gamma,ret_bits=True))
