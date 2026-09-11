# Results manifest

Backing data for [arXiv:2609.06008](https://arxiv.org/abs/2609.06008).

Every file here is committed so the paper can be verified **without a GPU**.

**Read this before quoting any number.** This repository contains results from
three different entropy back ends, and they do not agree. That is deliberate —
the back end turned out to be part of the experimental design, and two published
claims were retracted because of it (see [../EXPERIMENTS.md](../EXPERIMENTS.md),
R3 and R8). Files below are labelled with which back end produced them.

## Current — these back the paper

| File | Backs | Back end |
|---|---|---|
| `rescore_all.json` | Table 4, Figure 2 — all corpora, one coder | arithmetic |
| `rescore_ac.json` | Tables 5 & 6 — grid and transit per tolerance | arithmetic |
| `e2e_ac.json` | Table 8 — end-to-end container | arithmetic |
| `ablation_ac.json` | Table 11, Figure 4 — context ablation | arithmetic |
| `ac_bench.json` | §5.9 — coder vs xz/zstd; spread contexts | arithmetic |
| `fair_compare.json` | Table 2, Figure 1 — the lossless null result | shared adaptive |
| `sdrbench_ac.json` | Table 7 — SDRBench falsification test | arithmetic |
| `hybrid.json` | §5.10 — hybrid prediction, +0.0% | arithmetic |
| `determinism.json`, `determinism2.json` | Table 9 | back-end independent |
| `portability.json` | §5.6 — cross-device desync rate | back-end independent |
| `noise_gain.json` | §5.4 — predictor noise gain | back-end independent |
| `throughput.json` | Table 10 | back-end independent |
| `tfm_interp.json`, `real_crosschannel.json` | §5.10 — negative results | back-end independent |
| `registry.json` | the 40-experiment registry | — |

## Superseded — kept so the retractions can be checked

These were measured with a general-purpose back end (xz/zstd) or with idealized
code lengths. **Their gain figures do not match the paper.** They are retained
because the paper's methodological argument depends on being able to reproduce
what the earlier measurement said.

| File | What it said | Why superseded |
|---|---|---|
| `grid_bench.json` | grid +17.9% | xz back end; paper reports +13.3% (R8) |
| `transit_bench.json` | transit +26.0% | xz back end; paper reports +28.3% |
| `nab_realbytes.json` | NAB +5.2% | xz back end; paper reports +6.4% |
| `nab_bench.json` | NAB +3.8% | idealized coder — wrong basis entirely (R5) |
| `final_bench.json` | 6-series real-byte benchmark | superseded by `rescore_all.json` |
| `headtohead.json` | idealized head-to-head | superseded by `final_bench.json` |
| `ablation.json` | context ablation | xz back end; use `ablation_ac.json` |
| `hybrid_downsample.json` | §5.7 downsampling ratios | xz-based bitrate matching |
| `sdrbench.json` | SDRBench 3/27 | xz back end; paper reports 0/27 |
| `sz3_real.json` | "+70% vs SZ3" | measured at N=2048, where SZ3's ~500 B container dominates (R3) |
| `lossy.json`, `lossy_parsed.json` | "Lorenz +71.4%" | compared against a strawman LPC-32 baseline (R1) |
| `v2_c1024_s1.json` | early lossless gains | density-family confound, fixed in `fair_compare.json` |
| `viability_c1024_s64.json` | stride-64 viability | superseded by stride-1 |

## Supporting

`baselines.json` (classical codecs on the synthetic corpus), `strong_lpc.json`
(seasonal-lag baselines), `sz_style.json` (SZ3-class predictor family).
