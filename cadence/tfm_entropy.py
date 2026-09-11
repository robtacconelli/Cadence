"""Measure achievable codelength (bits/value) using TimesFM as the entropy model.

No arithmetic coder is needed: an arithmetic coder achieves -sum log2 P(x_t)
to within a few bytes total, so the exact cross-entropy IS the codec's output
size. Everything here is strictly causal, so every number is realizable.

Experts scored per position:
  qcdf        piecewise-linear CDF through the 9 quantiles + exponential tails
  lap[a]      median + discretized Laplace, scale = a * (q90-q10)/(2 ln 5)
  t3[a]       median + discretized Student-t(3), heavier tails
  lpc         classical order-p LPC + adaptive Laplace (the FLAC-class bar)
  MIX         Bayesian mixture of all of the above (causal weight updates)
"""
import numpy as np, time, os, json
from scipy.stats import t as student_t

ESC = 1e-4
LN5 = np.log(5.0)
T3_Q90 = float(student_t.ppf(0.9, 3))


def cdf_from_quantiles(q, x):
  """F(x) for a batch. q:(N,9) ascending, x:(N,) -> (N,) cumulative prob.
  Piecewise-linear between quantiles; exponential tails carrying 0.1 each,
  with scales chosen so the density is continuous at q10 and q90."""
  p = np.linspace(0.1, 0.9, 9)
  N = q.shape[0]
  eps = 1e-6
  q = np.maximum.accumulate(q, axis=1)
  spread = np.maximum(q[:, 8] - q[:, 0], eps)
  b_lo = np.maximum(q[:, 1] - q[:, 0], spread * 1e-3)
  b_hi = np.maximum(q[:, 8] - q[:, 7], spread * 1e-3)
  idx = (q <= x[:, None]).sum(1)                      # 0..9
  F = np.empty(N)
  lo = idx == 0
  F[lo] = 0.1 * np.exp((x[lo] - q[lo, 0]) / b_lo[lo])
  hi = idx == 9
  F[hi] = 1.0 - 0.1 * np.exp(-(x[hi] - q[hi, 8]) / b_hi[hi])
  mid = ~(lo | hi)
  if mid.any():
    i = idx[mid] - 1
    r = np.arange(N)[mid]
    qa, qb = q[r, i], q[r, i + 1]
    F[mid] = p[i] + 0.1 * (x[mid] - qa) / np.maximum(qb - qa, eps)
  return np.clip(F, 0.0, 1.0)


def p_qcdf(q, v):
  return cdf_from_quantiles(q, v + 0.5) - cdf_from_quantiles(q, v - 0.5)


def p_laplace(med, scale, v):
  r = np.abs(v - med)
  b = np.maximum(scale, 1e-3)
  hi = (r + 0.5) / b
  lo = np.maximum((r - 0.5) / b, 0.0)
  p = 0.5 * (np.exp(-lo) - np.exp(-hi))
  return np.where(r < 0.5, 1.0 - np.exp(-hi), p)


def p_t3(med, scale, v):
  b = np.maximum(scale, 1e-3)
  z1 = (v + 0.5 - med) / b
  z0 = (v - 0.5 - med) / b
  return student_t.cdf(z1, 3) - student_t.cdf(z0, 3)


def lpc_causal_pmf(x, order=32, adapt=256):
  """Classical expert: global LS-LPC (coeffs in header) + adaptive Laplace.
  Returns (pred, scale) aligned to x[order:]."""
  xf = np.asarray(x, dtype=np.float64)
  X = np.lib.stride_tricks.sliding_window_view(xf, order)[:-1]
  y = xf[order:]
  coef, *_ = np.linalg.lstsq(X, y, rcond=None)
  coef = np.rint(coef * 16384) / 16384
  pred = np.rint(X @ coef)
  ab = np.abs(y - pred)
  run = np.empty(len(ab)); acc = max(ab[:adapt].mean(), 0.5)
  for i in range(len(ab)):
    run[i] = acc; acc += (ab[i] - acc) / adapt
  return pred, run, order


LAP_A = [0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4, 3.2]
T3_A = [0.7, 1.0, 1.4, 2.0]


def score_series(fc, x, ctx=1024, stride=64, batch=32, verbose=False):
  """Returns dict expert -> bits/value, plus MIX, over positions [ctx, N)."""
  x = np.asarray(x, dtype=np.float64)
  N = len(x); span = float(x.max() - x.min() + 1)
  starts = list(range(ctx, N - stride + 1, stride))
  Q = np.zeros((len(starts) * stride, 9), dtype=np.float64)
  V = np.zeros(len(starts) * stride, dtype=np.float64)
  horizon = max(stride, 64)
  t0 = time.time(); nb = 0
  for s in range(0, len(starts), batch):
    chunk = starts[s:s + batch]
    ctxs = [x[t - ctx:t].astype(np.float32) for t in chunk]
    outs = list(fc.predict_batch(contexts=ctxs, horizon=horizon,
                                 return_quantiles=True, sort_quantiles=True))
    for j, t in enumerate(chunk):
      qq = np.asarray(outs[j].quantiles, dtype=np.float64).reshape(-1, 9)[:stride]
      k = (s + j) * stride
      Q[k:k + stride] = qq
      V[k:k + stride] = x[t:t + stride]
    nb += 1
  gpu_s = time.time() - t0

  med = Q[:, 4]
  s_lap = np.maximum((Q[:, 8] - Q[:, 0]) / (2 * LN5), 1e-3)
  s_t3 = np.maximum((Q[:, 8] - Q[:, 0]) / (2 * T3_Q90), 1e-3)

  experts, names = [], []
  experts.append(p_qcdf(Q, V)); names.append("qcdf")
  for a in LAP_A:
    experts.append(p_laplace(med, a * s_lap, V)); names.append(f"lap{a}")
  for a in T3_A:
    experts.append(p_t3(med, a * s_t3, V)); names.append(f"t3_{a}")

  # classical expert, aligned to the same positions
  pred, run, order = lpc_causal_pmf(x, 32)
  pos = np.concatenate([np.arange(t, t + stride) for t in starts])
  li = pos - order
  experts.append(p_laplace(pred[li], run[li], V)); names.append("lpc32")

  P = np.stack(experts)                       # (E, M)
  P = (1 - ESC) * np.clip(P, 0, 1) + ESC / span
  bits = -np.log2(P)
  out = {n: float(b.mean()) for n, b in zip(names, bits)}

  # causal Bayesian mixture with forgetting -> a real, implementable code
  E, M = P.shape
  w = np.full(E, 1.0 / E)
  gamma = 0.998                                # forgetting: allows switching
  mixbits = np.empty(M)
  for i in range(M):
    pm = float(w @ P[:, i])
    mixbits[i] = -np.log2(max(pm, 1e-300))
    w = w * P[:, i]; w /= w.sum()
    w = gamma * w + (1 - gamma) / E
  out["MIX"] = float(mixbits.mean())
  out["_gpu_s"] = gpu_s
  out["_n_scored"] = int(M)
  out["_vals_per_s"] = M / gpu_s
  return out
