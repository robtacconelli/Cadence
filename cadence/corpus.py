"""Benchmark corpus: integer-valued series with controlled, known structure.

Every series is an int64 numpy array. Controls are included deliberately:
if the model does not lose to them, the measurement is wrong.
"""
import numpy as np, os, json

RNG = np.random.default_rng(12345)


def _q(x, levels):
  """Quantize float array to `levels` uniform steps over its own range."""
  lo, hi = float(x.min()), float(x.max())
  if hi - lo < 1e-12:
    return np.zeros(len(x), dtype=np.int64)
  return np.rint((x - lo) / (hi - lo) * (levels - 1)).astype(np.int64)


def ar1(n, phi=0.99, levels=65536):
  e = RNG.standard_normal(n); x = np.zeros(n)
  for i in range(1, n): x[i] = phi * x[i-1] + e[i]
  return _q(x, levels)


def seasonal_metric(n, levels=65536):
  """Server-metric shaped: daily+weekly seasonality, trend, bursty noise."""
  t = np.arange(n)
  daily = 30 * np.sin(2*np.pi*t/288)            # 5-min samples, 288/day
  weekly = 12 * np.sin(2*np.pi*t/(288*7) + 0.7)
  trend = 0.004 * t
  noise = 3*RNG.standard_normal(n) + 25*(RNG.random(n) < 0.002)
  return _q(100 + daily + weekly + trend + noise, levels)


def random_walk(n, levels=65536):
  """CONTROL: increments are iid -> entropy floor is the increment entropy."""
  return _q(np.cumsum(RNG.standard_normal(n)), levels)


def iid_noise(n, bits=12):
  """CONTROL: pure noise. Nothing may beat log2(range) bits/value."""
  return RNG.integers(0, 2**bits, size=n).astype(np.int64)


def lorenz(n, levels=65536, dt=0.01):
  """Chaotic but smooth & deterministic: locally very predictable."""
  s, r, b = 10.0, 28.0, 8/3
  xyz = np.array([1.0, 1.0, 1.0]); out = np.empty(n)
  for i in range(n):
    x, y, z = xyz
    xyz = xyz + dt*np.array([s*(y-x), x*(r-z)-y, x*y-b*z])
    out[i] = xyz[0]
  return _q(out, levels)


def byte_counter(n):
  """Monotone network counter with wraps: delta coding is near-optimal."""
  rate = 5000 + 4000*np.sin(2*np.pi*np.arange(n)/1440)
  inc = RNG.poisson(np.maximum(rate, 1))
  return (np.cumsum(inc) % (2**31)).astype(np.int64)


def poisson_counts(n, lam=8.0):
  """Low-cardinality count data with slow rate drift."""
  rate = lam * (1 + 0.6*np.sin(2*np.pi*np.arange(n)/500))
  return RNG.poisson(np.maximum(rate, 0.05)).astype(np.int64)


def ecg_like(n, levels=4096, fs=250):
  """Synthetic ECG: sharp quasi-periodic morphology + baseline wander."""
  t = np.arange(n)/fs; out = np.zeros(n); rr = 0.0; i = 0
  while rr < t[-1]:
    c = int(rr*fs)
    for amp, off, wid in ((-0.15,-0.03,0.012),(1.0,0.0,0.010),
                          (-0.25,0.028,0.014),(0.30,0.14,0.055)):
      k = np.arange(max(0,c+int(off*fs)-120), min(n,c+int(off*fs)+120))
      out[k] += amp*np.exp(-((k-(c+off*fs))/(wid*fs))**2)
    rr += 0.8 + 0.06*RNG.standard_normal(); i += 1
  out += 0.12*np.sin(2*np.pi*np.arange(n)/(fs*4))          # baseline wander
  out += 0.006*RNG.standard_normal(n)                       # sensor noise
  return _q(out, levels)


def regime_switch(n, levels=65536):
  """Non-stationary: abrupt regime changes in level and volatility."""
  x = np.zeros(n); lvl = 0.0; vol = 1.0
  for i in range(n):
    if RNG.random() < 1/1500:
      lvl += 20*RNG.standard_normal(); vol = 10**RNG.uniform(-0.5, 1.0)
    x[i] = lvl + vol*RNG.standard_normal()
  return _q(x, levels)


def sparse_spiky(n, levels=65536):
  """Mostly flat with rare large spikes -- IoT / event-driven telemetry."""
  x = np.zeros(n); v = 0.0
  for i in range(n):
    if RNG.random() < 0.01: v = 100*RNG.standard_normal()
    v *= 0.92
    x[i] = v + 0.5*RNG.standard_normal()
  return _q(x, levels)


BUILDERS = {
  "ar1_phi0.99":     lambda: ar1(20000),
  "seasonal_metric": lambda: seasonal_metric(20000),
  "lorenz_chaotic":  lambda: lorenz(20000),
  "ecg_like":        lambda: ecg_like(20000),
  "sparse_spiky":    lambda: sparse_spiky(20000),
  "poisson_counts":  lambda: poisson_counts(20000),
  "byte_counter":    lambda: byte_counter(20000),
  "regime_switch":   lambda: regime_switch(20000),
  "random_walk":     lambda: random_walk(20000),   # CONTROL
  "iid_noise":       lambda: iid_noise(20000),     # CONTROL
}


def build(outdir="data/synth"):
  os.makedirs(outdir, exist_ok=True)
  meta = {}
  for name, fn in BUILDERS.items():
    a = np.asarray(fn(), dtype=np.int64)
    np.save(f"{outdir}/{name}.npy", a)
    meta[name] = dict(n=len(a), lo=int(a.min()), hi=int(a.max()),
                      uniq=int(len(np.unique(a))))
    print(f"{name:18s} n={len(a):6d} range=[{a.min()},{a.max()}] uniq={meta[name]['uniq']}")
  json.dump(meta, open(f"{outdir}/meta.json","w"), indent=1)
  return meta


if __name__ == "__main__":
  build()
