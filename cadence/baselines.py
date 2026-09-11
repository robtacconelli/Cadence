"""Classical baselines. The point is to build an HONEST bar, not an easy one.

Two families:
  * real codecs on the natural byte serialization (xz/zstd/bz2/brotli,
    with and without delta + byte-shuffle preprocessing)
  * ideal codelengths for predictive models (order-0, delta, LPC+adaptive
    Laplace == the "FLAC/Shorten" class), measured as exact -sum log2 P.
"""
import numpy as np, lzma, bz2, zlib, subprocess, io
import zstandard as zstd, brotli

ESC = 1e-4  # uniform escape mass: guarantees finite codelength, as a real codec must


def serialize(a):
  """Natural fixed-width unsigned little-endian serialization."""
  a = np.asarray(a, dtype=np.int64); lo = int(a.min()); u = a - lo
  span = int(u.max())
  for w, dt in ((1, np.uint8), (2, np.uint16), (4, np.uint32), (8, np.uint64)):
    if span < 2**(8*w):
      return u.astype(dt).tobytes(), w
  raise ValueError


def shuffle_bytes(b, w):
  """Byte transposition (blosc/HDF5 shuffle): groups like-significance bytes."""
  if w == 1: return b
  arr = np.frombuffer(b, dtype=np.uint8).reshape(-1, w)
  return arr.T.copy().tobytes()


def codec_sizes(a):
  """Bytes produced by real compressors, over several preprocessings."""
  raw, w = serialize(a)
  d1 = np.diff(np.asarray(a, dtype=np.int64), prepend=a[0])
  draw, dw = serialize(d1)
  variants = {
    "raw":         raw,
    "shuf":        shuffle_bytes(raw, w),
    "delta":       draw,
    "delta+shuf":  shuffle_bytes(draw, dw),
  }
  zc = zstd.ZstdCompressor(level=22, write_content_size=False)
  out = {}
  for vname, buf in variants.items():
    out[f"xz:{vname}"]     = len(lzma.compress(buf, preset=9 | lzma.PRESET_EXTREME))
    out[f"zstd:{vname}"]   = len(zc.compress(buf))
    out[f"bz2:{vname}"]    = len(bz2.compress(buf, 9))
    out[f"brotli:{vname}"] = len(brotli.compress(buf, quality=11))
  out["_raw_bytes"] = len(raw)
  out["_width"] = w
  return out


# ---------- ideal codelengths for predictive models ----------

def _disc_laplace_bits(resid, scale, span):
  """Exact bits for residuals under a discretized Laplace(0, scale),
  mixed with a uniform escape over `span` so codelength is always finite."""
  b = np.maximum(scale, 1e-6)
  hi = (np.abs(resid) + 0.5) / b
  lo = (np.abs(resid) - 0.5) / b
  # P(|r| in [|r|-.5,|r|+.5]) for two-sided Laplace; r=0 handled by clipping lo>=0
  lo = np.maximum(lo, 0.0)
  p = 0.5 * (np.exp(-lo) - np.exp(-hi))
  p = np.where(resid == 0, 1.0 - np.exp(-hi), p)
  p = (1 - ESC) * p + ESC / span
  return -np.log2(np.maximum(p, 1e-300))


def order0_bits(a):
  v, c = np.unique(a, return_counts=True)
  p = c / c.sum()
  # + model cost for the histogram (values + counts), crudely 32 bits/symbol
  return float(-(c * np.log2(p)).sum()) + 32.0 * len(v)


def delta_order0_bits(a):
  return order0_bits(np.diff(np.asarray(a, dtype=np.int64), prepend=a[0]))


def lpc_adaptive_laplace_bits(a, order=32, adapt=256):
  """The real bar: global least-squares LPC + adaptive-Laplace residual coding.
  This is the ideal-codelength version of FLAC/Shorten and beats them slightly."""
  x = np.asarray(a, dtype=np.float64)
  n = len(x); span = float(x.max() - x.min() + 1)
  if n <= order + 2:
    return order0_bits(a)
  # least-squares LPC on the whole file (coefficients shipped in the header)
  X = np.lib.stride_tricks.sliding_window_view(x, order)[:-1]
  y = x[order:]
  coef, *_ = np.linalg.lstsq(X, y, rcond=None)
  coef = np.rint(coef * 16384) / 16384          # quantize to 14 fractional bits
  pred = np.rint(X @ coef)
  resid = y - pred
  # adaptive Laplace scale from a causal running mean of |resid|
  ab = np.abs(resid)
  run = np.empty(len(ab)); acc = max(ab[:adapt].mean() if len(ab) >= adapt else ab.mean(), 0.5)
  for i in range(len(ab)):
    run[i] = acc
    acc += (ab[i] - acc) / adapt
  bits = _disc_laplace_bits(resid, run, span).sum()
  header = 32.0 * order + 64.0 * order          # coefficients + prologue samples
  return float(bits + header)


def all_baselines(a):
  n = len(a)
  cs = codec_sizes(a)
  res = {k: v * 8.0 / n for k, v in cs.items() if not k.startswith("_")}
  res["ideal:order0"] = order0_bits(a) / n
  res["ideal:delta0"] = delta_order0_bits(a) / n
  res["ideal:lpc32+lap"] = lpc_adaptive_laplace_bits(a, 32) / n
  res["ideal:lpc8+lap"] = lpc_adaptive_laplace_bits(a, 8) / n
  res["_raw_bpv"] = cs["_raw_bytes"] * 8.0 / n
  return res


if __name__ == "__main__":
  import glob, os, json
  out = {}
  for f in sorted(glob.glob("data/synth/*.npy")):
    name = os.path.basename(f)[:-4]
    a = np.load(f)
    r = all_baselines(a)
    best = min((v, k) for k, v in r.items() if not k.startswith("_"))
    out[name] = r
    print(f"{name:18s} raw={r['_raw_bpv']:5.1f}  best={best[1]:18s} {best[0]:7.3f} bpv"
          f"   xz:raw={r['xz:raw']:6.3f}  lpc32={r['ideal:lpc32+lap']:6.3f}")
  json.dump(out, open("results/baselines.json", "w"), indent=1)
