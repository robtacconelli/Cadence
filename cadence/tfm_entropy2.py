"""v2 scorer: every expert is causal and implementable in a real codec.

Fixes two flaws in v1:
  * scale was anchored to the model's quantiles; now it is causally
    recalibrated against observed residuals (what a real codec does)
  * adds STACKING: least-squares combination of the model's median with the
    last p samples. This is the key test -- does the model's forecast carry
    information a linear predictor does not already have?
"""
import numpy as np, time
from scipy.stats import t as student_t

ESC = 1e-4
LN5 = np.log(5.0)
T3_Q90 = float(student_t.ppf(0.9, 3))
from tfm_entropy import cdf_from_quantiles, p_qcdf, p_laplace, p_t3


def running_mean(v, adapt=256, init=None):
  """Causal exponential running mean; run[i] uses only v[:i]."""
  out = np.empty(len(v))
  acc = float(init if init is not None else max(np.mean(np.abs(v[:adapt])), 1e-3))
  for i in range(len(v)):
    out[i] = acc
    acc += (v[i] - acc) / adapt
  return np.maximum(out, 1e-3)


def collect(fc, x, ctx, stride, batch, npos, horizon=64):
  """Run the model over npos contiguous positions starting at ctx."""
  x = np.asarray(x, dtype=np.float64)
  starts = list(range(ctx, min(ctx + npos * stride, len(x) - stride + 1), stride))
  Q = np.zeros((len(starts) * stride, 9)); V = np.zeros(len(starts) * stride)
  t0 = time.time()
  for s in range(0, len(starts), batch):
    chunk = starts[s:s + batch]
    outs = list(fc.predict_batch(contexts=[x[t - ctx:t].astype(np.float32) for t in chunk],
                                 horizon=horizon, return_quantiles=True, sort_quantiles=True))
    for j, t in enumerate(chunk):
      k = (s + j) * stride
      Q[k:k + stride] = np.asarray(outs[j].quantiles, dtype=np.float64).reshape(-1, 9)[:stride]
      V[k:k + stride] = x[t:t + stride]
  pos = np.concatenate([np.arange(t, t + stride) for t in starts])
  return Q, V, pos, time.time() - t0


def score(x, Q, V, pos, adapt=256, stack_p=16):
  x = np.asarray(x, dtype=np.float64)
  span = float(x.max() - x.min() + 1)
  M = len(V)
  med = Q[:, 4]
  s_mod = np.maximum((Q[:, 8] - Q[:, 0]) / (2 * LN5), 1e-3)
  r = V - med
  ar = np.abs(r)

  experts, names = [], []
  experts.append(p_qcdf(Q, V)); names.append("qcdf")

  # (1) homoscedastic recalibration: scale = causal running mean|r|
  sA = running_mean(ar, adapt)
  experts.append(p_laplace(med, sA, V)); names.append("lapA")

  # (2) heteroscedastic recalibration: keep the model's shape, fix its gain
  alpha = running_mean(ar, adapt) / running_mean(s_mod, adapt)
  experts.append(p_laplace(med, alpha * s_mod, V)); names.append("lapQcal")
  experts.append(p_t3(med, alpha * s_mod * (LN5 / T3_Q90), V)); names.append("t3Qcal")

  # (3) classical bar: order-32 LS-LPC + adaptive Laplace
  o = 32
  X = np.lib.stride_tricks.sliding_window_view(x, o)[:-1]
  c, *_ = np.linalg.lstsq(X, x[o:], rcond=None); c = np.rint(c * 16384) / 16384
  predL = np.rint(X @ c); rl = x[o:] - predL
  sL = running_mean(np.abs(rl), adapt)
  li = pos - o
  experts.append(p_laplace(predL[li], sL[li], V)); names.append("lpc32")

  # (4) STACK: least-squares blend of the model median with the last p samples
  F = np.column_stack([np.ones(M), med] + [x[pos - k] for k in range(1, stack_p + 1)])
  cs, *_ = np.linalg.lstsq(F, V, rcond=None)
  preds = np.rint(F @ cs); rs = V - preds
  experts.append(p_laplace(preds, running_mean(np.abs(rs), adapt), V)); names.append("stack")

  # (5) STACK without the model -> isolates what the model actually contributes
  F0 = np.column_stack([np.ones(M)] + [x[pos - k] for k in range(1, stack_p + 1)])
  c0, *_ = np.linalg.lstsq(F0, V, rcond=None)
  p0 = np.rint(F0 @ c0); r0 = V - p0
  experts.append(p_laplace(p0, running_mean(np.abs(r0), adapt), V)); names.append("stack_nomodel")

  P = np.stack(experts)
  P = (1 - ESC) * np.clip(P, 0, 1) + ESC / span
  bits = -np.log2(P)
  out = {n: float(b.mean()) for n, b in zip(names, bits)}

  E = P.shape[0]; w = np.full(E, 1.0 / E); g = 0.998; mb = np.empty(M)
  for i in range(M):
    mb[i] = -np.log2(max(float(w @ P[:, i]), 1e-300))
    w = w * P[:, i]; w /= w.sum(); w = g * w + (1 - g) / E
  out["MIX"] = float(mb.mean())
  # mixture WITHOUT any TimesFM expert -> the honest classical-only mixture
  ci = [names.index("lpc32"), names.index("stack_nomodel")]
  Pc = P[ci]; w = np.full(len(ci), 1.0 / len(ci)); mb2 = np.empty(M)
  for i in range(M):
    mb2[i] = -np.log2(max(float(w @ Pc[:, i]), 1e-300))
    w = w * Pc[:, i]; w /= w.sum(); w = g * w + (1 - g) / len(ci)
  out["MIX_classical_only"] = float(mb2.mean())
  out["_stack_coef_model"] = float(cs[1])
  out["_mae_model"] = float(ar.mean()); out["_mae_lpc"] = float(np.abs(rl[li]).mean())
  out["_mae_stack"] = float(np.abs(rs).mean())
  return out
