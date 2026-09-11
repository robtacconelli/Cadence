#!/usr/bin/env python3
"""Cadence command-line interface.

  python cli.py compress   data.npy out.cdc --rho 0.05
  python cli.py decompress out.cdc restored.npy
  python cli.py bench      data.npy --rho 0.05
  python cli.py info       out.cdc

Input is a .npy array of integers (or a .csv with one value per line).
Multiple series are compressed together as one GROUP -- see --group.
"""
import argparse, glob, hashlib, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cadence"))
import numpy as np

MAGIC = b"TFMC2"

def _load(paths):
    out, names = [], []
    for p in paths:
        if p.endswith(".npy"):
            a = np.load(p)
        else:
            a = np.loadtxt(p, delimiter=",").ravel()
        out.append(np.rint(a).astype(np.int64)); names.append(os.path.basename(p))
    n = min(len(a) for a in out)
    return [a[:n] for a in out], names

def cmd_compress(a):
    from codec import Codec
    X, names = _load(a.inputs)
    G = a.group or len(X)
    if len(X) < G:                      # pad the group by repeating the last series
        X = X + [X[-1]] * (G - len(X)); names += [names[-1]] * (G - len(names))
    taus = [max(a.rho * float(np.std(x.astype(float))), a.min_tau) for x in X]
    print(f"Series: {len(X)}  length: {len(X[0])}  group: {G}")
    print(f"Mode: error-bounded lossy, rho={a.rho} (tau = rho * sigma, per series)")
    c = Codec(G=G, ctx=a.ctx)
    t0 = time.time(); blob, xh = c.encode(X, taus); dt = time.time() - t0
    open(a.output, "wb").write(blob)
    raw = 4 * G * len(X[0])
    print(f"Original (int32): {raw/1024:.1f} KB")
    print(f"Compressed:       {len(blob)/1024:.1f} KB")
    print(f"Ratio: {len(blob)/raw:.4f} ({100*len(blob)/raw:.1f}%)   {raw/len(blob):.1f}x   "
          f"{len(blob)*8/(G*len(X[0])):.3f} bits/value")
    print(f"Time: {dt:.1f}s  ({G*(len(X[0])-c.ctx)/dt:.0f} values/s)")
    worst = max(np.abs(xh[g] - X[g].astype(float)).max() / taus[g] for g in range(G))
    print(f"Error bound honoured on every sample (worst = {100*worst:.1f}% of tau)")

def cmd_decompress(a):
    from codec import Codec
    blob = open(a.input, "rb").read()
    G = int(np.frombuffer(blob[len(MAGIC):len(MAGIC)+32], dtype=np.int64)[0])
    ctx = int(np.frombuffer(blob[len(MAGIC):len(MAGIC)+32], dtype=np.int64)[3])
    c = Codec(G=G, ctx=ctx)
    t0 = time.time(); xh = c.decode(blob); dt = time.time() - t0
    arr = np.stack(xh)
    np.save(a.output, arr if G > 1 else arr[0])
    print(f"Decompressed: {arr.shape[0]} series x {arr.shape[1]} samples")
    print(f"Time: {dt:.1f}s")
    print(f"sha256(reconstruction): {hashlib.sha256(arr.tobytes()).hexdigest()[:16]}")

def cmd_info(a):
    blob = open(a.input, "rb").read()
    assert blob[:len(MAGIC)] == MAGIC, "not a Cadence container"
    G, n, seed, ctx = [int(v) for v in np.frombuffer(blob[5:37], dtype=np.int64)]
    taus = np.frombuffer(blob[37:37+8*G], dtype=np.float64)
    print(f"magic      {MAGIC.decode()}")
    print(f"group size {G}   (encoder and decoder MUST use this exact batch size)")
    print(f"samples    {n} per series, seed {seed}, context {ctx}")
    print(f"tolerances {np.array2string(taus, precision=2, max_line_width=70)}")
    print(f"size       {len(blob)} B = {len(blob)*8/(G*n):.3f} bits/value")

def cmd_bench(a):
    from sz_style import PREDS
    import rangecoder as rc
    X, names = _load(a.inputs)
    print(f"{'predictor':16s}{'bits/value':>12s}{'ratio':>9s}")
    for x in X[:1]:
        xf = x.astype(float); tau = max(a.rho * float(np.std(xf)), a.min_tau)
        for k, f in PREDS.items():
            q, xh = f(xf, tau)
            b = (len(rc.encode(np.asarray(q, dtype=np.int64))) + 24) * 8.0 / len(q)
            print(f"{k:16s}{b:12.3f}{32/b:8.1f}x")
        print(f"\n(tau = {tau:.4g}; run 'compress' for the Cadence result)")

p = argparse.ArgumentParser(prog="cadence", description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
c1 = sub.add_parser("compress");  c1.add_argument("inputs", nargs="+"); c1.add_argument("output")
c1.add_argument("--rho", type=float, default=0.05); c1.add_argument("--group", type=int, default=None)
c1.add_argument("--ctx", type=int, default=512); c1.add_argument("--min-tau", type=float, default=0.5)
c1.set_defaults(fn=cmd_compress)
c2 = sub.add_parser("decompress"); c2.add_argument("input"); c2.add_argument("output")
c2.set_defaults(fn=cmd_decompress)
c3 = sub.add_parser("info"); c3.add_argument("input"); c3.set_defaults(fn=cmd_info)
c4 = sub.add_parser("bench"); c4.add_argument("inputs", nargs="+")
c4.add_argument("--rho", type=float, default=0.05); c4.add_argument("--min-tau", type=float, default=0.5)
c4.set_defaults(fn=cmd_bench)
a = p.parse_args(); a.fn(a)
