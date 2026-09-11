#!/usr/bin/env python3
"""Check every quantitative claim in the paper against results/*.json.

Run from the repository root:  python scripts/verify_paper.py
Exits non-zero if any claim does not match. Needs no GPU and no data.
"""
import json, re, numpy as np, sys
R="results/"; L=lambda f: json.load(open(R+f))
ok=[];bad=[]
def chk(label, paper, actual, tol=0.15):
    good = abs(paper-actual)<=tol if isinstance(paper,float) else paper==actual
    (ok if good else bad).append((label,paper,actual))

# Table 4 as printed in the paper: (corpus, median gain %, wins/n)
ra=L("rescore_all.json")
TABLE4=[("SDRBench",-0.8,"0/27"),("Synthetic",2.9,"7/12"),("NAB oper.",6.4,"21/24"),
        ("Grid load",13.3,"147/147"),("Transit",28.3,"150/150")]
for key,med,wins in TABLE4:
    v=ra[key]
    chk(f"Table 4 {key} median", med, round(v["median"],1))
    chk(f"Table 4 {key} wins", wins, f"{v['wins']}/{v['n']}")
dc=ra["_demand_combined"]
chk("Table 4 demand median", 21.4, round(dc["median"],1))
chk("Table 4 demand wins", "297/297", f"{dc['wins']}/{dc['n']}")

ac=L("rescore_ac.json")
for dom,tab,vals in [("grid","Tab5",[(0.01,6.4,4.356,4.009),(0.05,13.3,2.398,2.056),(0.2,21.2,1.265,0.980)]),
                     ("transit","Tab6",[(0.01,19.9,5.942,4.734),(0.05,27.7,3.324,2.374),(0.2,51.2,1.834,0.926)])]:
    for rho,g,cl,ca in vals:
        v=ac[f"{dom}|{rho}"]
        chk(f"{tab} {dom} rho={rho} gain", g, round(v["median"],1))
        chk(f"{tab} {dom} rho={rho} classic", cl, round(v["classic"],3), 0.002)
        chk(f"{tab} {dom} rho={rho} cadence", ca, round(v["cadence"],3), 0.002)

e2=L("e2e_ac.json")
chk("Table 8 body", 1.906, round(e2["body"],3), 0.002)
chk("Table 8 classical e2e", 2.244, round(e2["classical"],3), 0.002)
chk("Table 8 total", 2.111, round(e2["cadence"],3), 0.002)

d1=L("determinism.json")
chk("Table 9 batch1v8", 3.9e-3, round(d1["B_maxdiff"],5), 1e-4)
chk("Table 9 batch8v9", 7.8e-3, round(d1["D_maxdiff"],5), 1e-4)
po=L("portability.json")
chk("§5.6 desync samples ~144k", 144000, round(po["expected_samples_to_desync"],-3), 2000)

th={ (r["batch"],r["ctx"],r["dtype"]):round(r["vps"]) for r in L("throughput.json")}
for b,c,dt,v in [(6,1024,"fp32",45),(64,1024,"fp32",113),(256,1024,"fp32",123),(256,512,"fp32",224)]:
    chk(f"Table 10 b{b} ctx{c}", v, th.get((b,c,dt)), 3)

ab=L("ablation_ac.json")
for c,body,seed in [(64,2.183,6.367),(256,2.017,3.611),(512,1.959,3.214),(1024,1.956,2.980)]:
    chk(f"Table 11 ctx={c} body", body, round(ab["body"][str(c)],3), 0.002)
    chk(f"Table 11 ctx={c} seed", seed, round(ab["seed"][str(c)],3), 0.002)

acb=L("ac_bench.json")
ga=[x["gain_ac"] for x in acb]; gs=[x["gain_spread"] for x in acb]
chk("§5.9 AC vs xz median", 9.7, round(float(np.median(ga)),1))
chk("§5.9 AC wins", "15/15", f"{sum(1 for x in ga if x>0)}/{len(ga)}")
chk("§5.9 spread median", -13.8, round(float(np.median(gs)),1))
chk("§5.9 spread wins", "0/15", f"{sum(1 for x in gs if x>0)}/{len(gs)}")

hy=L("hybrid.json")
chk("§5.10 hybrid gain rho=0.05", 0.0, round(hy["0.05"]["gain"],1))
chk("§5.10 tfm block fraction", 1.0, round(hy["0.05"]["tfm_frac"],2), 0.01)

fc=L("fair_compare.json")
g=[v["_gain_pct"] for k,v in fc.items() if "_COND_" not in k]
chk("Table 2 lossless median", 0.03, round(float(np.median(g)),2), 0.005)
chk("Table 2 iid_noise bpv", 12.003, round(fc["iid_noise"]["tfm"],3), 0.002)
r=fc["wikihr_en"]["mae_lpc32"]/fc["wikihr_en"]["mae_tfm"]
chk("Eq.1 MAE ratio", 1.51, round(r,2), 0.005)
chk("Eq.1 bits saved", 0.60, round(float(np.log2(r)),2), 0.005)

sd={k:{"gain":v} for k,v in L("sdrbench_ac.json").items()}
ex=[v["gain"] for k,v in sd.items() if k.startswith("exaalt")]
hu=[v["gain"] for k,v in sd.items() if k.startswith("isabel")]
al=[v["gain"] for v in sd.values()]
chk("Table 7 EXAALT median", -0.6, round(float(np.median(ex)),1))
chk("Table 7 EXAALT wins", "0/18", f"{sum(1 for x in ex if x>0)}/{len(ex)}")
chk("Table 7 Hurricane median", -25.4, round(float(np.median(hu)),1))
chk("Table 7 all wins", "0/27", f"{sum(1 for x in al if x>0)}/{len(al)}")

print(f"VERIFIED {len(ok)}/{len(ok)+len(bad)} claims")
if bad:
    print("\nMISMATCHES:")
    for l,p,a in bad: print(f"  {l:38s} paper={p!r:12s} results={a!r}")
sys.exit(1 if bad else 0)
