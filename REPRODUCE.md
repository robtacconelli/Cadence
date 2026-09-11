# Reproducing the paper

Paper: [arXiv:2609.06008](https://arxiv.org/abs/2609.06008)

Every number maps to one script. `results/*.json` are committed, so all claims
are checkable **without a GPU**; re-running the model requires one.

Scripts with a GPU stage take an argument: `python <script> gpu` collects model
outputs into `cache/`, then the same script without arguments scores them on CPU.

Set `CADENCE_CACHE` / `CADENCE_DATA` to relocate the cache and data directories.

| Paper | Claim | Script | GPU |
|---|---|---|---|
| Eq. 1, Fig. 1 | log₂ law; 1.51× accuracy → 0.60 bits | `experiments/fair_compare.py` | cached |
| Tab. 2 | **Lossless: +0.03% median** (dead) | `experiments/fair_compare.py` | cached |
| Tab. 2 | Harness validation: `iid_noise` = 12.003 bpv, +0.00% | `experiments/fair_compare.py` | cached |
| §5.2 | Cross-series covariates: no effect | `experiments/run_real.py` | yes |
| Tab. 4, Fig. 2 | **All corpora, one back end** | `experiments/rescore_all.py` | cached |
| Tab. 5 | Grid load +13.3%, 147/147 | `experiments/grid_bench.py` | yes |
| Tab. 6 | Transit +28.3%, 150/150 | `experiments/transit_bench.py` | yes |
| Fig. 3 | Rate–distortion vs classical and SZ3 | `scripts/make_figures.py` | cached |
| Tab. 7 | **SDRBench: −0.8%** (falsification test) | `experiments/sdrbench.py` | yes |
| §5.4 | Predictor noise gain; 15/18 sign prediction | `experiments/noise_gain.py` | yes |
| Tab. 8 | End-to-end container, 2.111 bpv, 15.2× | `experiments/codec_demo.py` | yes |
| Tab. 8 | **Bit-exact round trip** (SHA-256 match) | `experiments/codec_demo.py` | yes |
| Tab. 9 | Determinism: batch size breaks bit-identity | `experiments/determinism.py` | yes |
| Tab. 9 | No config restores invariance | `experiments/determinism2.py` | yes |
| §5.6 | Cross-device: GPU ≠ CPU, ~144k samples to desync | `experiments/portability.py` | yes |
| §5.7 | vs downsampling: 28–56× tighter L∞ | `experiments/hybrid_and_downsample.py` | cached |
| Tab. 10 | Throughput 45 → 442 values/s | `experiments/throughput.py` | yes |
| Tab. 11, Fig. 4 | Context ablation | `experiments/ablation.py` | yes |
| §5.9 | **Quantile head worthless** (−13.8% as AC contexts) | `experiments/ac_bench.py` | cached |
| §5.9 | Arithmetic coder beats xz/zstd +9.7%, 15/15 | `experiments/ac_bench.py` | cached |
| §5.10 | Hybrid: leader picks TimesFM 100%, +0.0% | `experiments/hybrid.py` | yes |
| §5.10 | Interpolation mode fails | `experiments/tfm_interp.py` | yes |
| — | Full experiment registry (40 runs, 8 retractions) | `experiments/registry.py` | no |

## Minimum path to the headline result

```bash
python scripts/fetch_data.py grid transit
python experiments/grid_bench.py gpu && python experiments/grid_bench.py
python experiments/transit_bench.py gpu && python experiments/transit_bench.py
python experiments/rescore_all.py
```

## Superseded and retracted experiments

These scripts produced results that the paper reports as **retracted or
superseded**. They are kept so the retractions can be independently checked —
the methodological argument in §6.2 depends on being able to reproduce what the
earlier measurement said. See [results/README.md](results/README.md) for which
result file is current and which is not.

| Script | Experiment | Status |
|---|---|---|
| `experiments/run_viability.py` | E10 — stride-64 lossless viability | superseded by stride-1 |
| `experiments/run_v2.py` | E12 — lossless with recalibration | retracted: density-family confound |
| `experiments/lossy.py` | E20 — first lossy sweep, "Lorenz +71.4%" | retracted (R1): strawman LPC baseline |
| `experiments/headtohead.py` | E23 — idealized head-to-head | superseded by `final_bench.py` |
| `experiments/final_bench.py` | E25 — real bytes, N=8192 | superseded by `rescore_all.py` |
| `experiments/nab_bench.py` | E30 — NAB on idealized coder | retracted (R5); also generates the NAB cache |
| `experiments/ablation_analyze.py` | E43/E44 scoring half of `ablation.py` | xz back end; see `ablation_ac.json` |
| `experiments/collect_cache.py` | caches model outputs for `fair_compare.py` | current (support script) |

## Cache generation

Scripts marked "cached" above read `cache/*.npz`, produced by the GPU stage of
`grid_bench.py`, `transit_bench.py`, `nab_bench.py`, `ablation.py`, `sdrbench.py`
and `collect_cache.py`. Run those with the `gpu` argument first if you are
regenerating from scratch rather than using the committed JSON.

## Retractions

Eight claims in this work were retracted under better measurement, all recorded
in `EXPERIMENTS.md` with the experiment that overturned each. Two are load-bearing
for anyone extending this:

- **R3** — SZ3 has ~500 B of container overhead that dominates below N≈4k. An
  early comparison at N=2048 overstated our advantage by a wide margin.
- **R8** — "grid and transit trend oppositely in tolerance" was an artifact of
  the xz back end, which compresses simple predictors' long zero runs very well.
  It reversed under a real arithmetic coder.

Every idealized or projected number in this study came in **high** when measured
as real bytes end-to-end. Treat idealized code lengths as upper bounds only.
