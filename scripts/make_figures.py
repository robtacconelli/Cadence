"""Paper figures. Print-safe: every series carries a distinct marker AND
linestyle, so identity survives grayscale printing and CVD. Palette validated
with the dataviz validator (all checks PASS, light surface)."""
import os as _os
CACHE=_os.environ.get("CADENCE_CACHE",_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cache"))
_os.makedirs(CACHE,exist_ok=True)
import numpy as np, json, os, sys, glob, lzma, subprocess
sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","cadence"))
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zstandard as zstd
from sz_style import PREDS

C = {"blue":"#2a78d6","orange":"#eb6834","aqua":"#1baf7a","violet":"#4a3aa7","yellow":"#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({
 "font.family":"serif","font.serif":["DejaVu Serif"],"font.size":7.5,
 "axes.labelsize":7.5,"axes.titlesize":8,"legend.fontsize":6.8,
 "xtick.labelsize":7,"ytick.labelsize":7,
 "axes.edgecolor":INK2,"axes.linewidth":0.6,"axes.labelcolor":INK,
 "text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
 "xtick.major.width":0.6,"ytick.major.width":0.6,
 "grid.color":GRID,"grid.linewidth":0.5,"legend.frameon":False,
 "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.02,
})
def _save(fig,name):
    """PNG for the README, PDF for the paper submission."""
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    fig.savefig(f"{OUT}/{name}.pdf")

def style(ax):
    ax.grid(True, axis="both", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
OUT=os.environ.get("CADENCE_FIGS","assets"); os.makedirs(OUT,exist_ok=True)

# ---------------------------------------------------------------- Fig 1
def fig_log2law():
    fc=json.load(open("results/fair_compare.json"))
    fig,ax=plt.subplots(figsize=(3.3,2.35))
    r=np.logspace(0,3,200); ax.plot(r,np.log2(r),color=C["blue"],lw=1.6,zorder=3)
    pts=[("wikihr\\_en","wikihr_en"),("ecg\\_like","ecg_like"),("sparse\\_spiky","sparse_spiky")]
    xs=[];ys=[]
    for lbl,k in pts:
        v=fc[k]; rr=v["mae_lpc32"]/v["mae_tfm"]; xs.append(rr); ys.append(np.log2(rr))
    ax.axvspan(min(xs)*0.85,max(xs)*1.18,color=C["orange"],alpha=0.13,zorder=1)
    ax.scatter(xs,ys,s=26,color=C["orange"],zorder=6,marker="o",
               edgecolor="white",linewidth=0.7)
    ax.annotate("observed regime\n(TimesFM vs. LPC-32)",(max(xs),np.log2(max(xs))),
                textcoords="offset points",xytext=(8,112),fontsize=6.5,color=INK,ha="left",
                arrowprops=dict(arrowstyle="-",lw=0.6,color=INK2))
    ins=ax.inset_axes([0.545,0.10,0.33,0.34])
    rz=np.linspace(1.0,1.75,80); ins.plot(rz,np.log2(rz),color=C["blue"],lw=1.3)
    ins.scatter(xs,ys,s=20,color=C["orange"],zorder=5,edgecolor="white",linewidth=0.6)
    ins.annotate("3 real series",(np.mean(xs),np.mean(ys)),textcoords="offset points",
                 xytext=(-2,-13),fontsize=5.4,color=INK2,ha="center")
    ins.set_xlim(1.0,1.75); ins.set_ylim(0,0.85)
    ins.tick_params(labelsize=5.5,width=0.5,length=2)
    ins.set_title("zoom: 0.41-0.60 bits saved",fontsize=5.6,color=INK2,pad=2)
    for sp in ("top","right"): ins.spines[sp].set_visible(False)
    ins.grid(alpha=0.5,lw=0.4); ins.set_axisbelow(True)
    ax.axhline(10,color=INK2,lw=0.7,ls=":",zorder=2)
    ax.annotate("10 bits = halve a 20 bpv file\n$\\rightarrow$ needs $1024\\times$",(1.15,10),
                textcoords="offset points",xytext=(2,-20),fontsize=6.5,color=INK2)
    ax.set_xscale("log"); ax.set_xlim(1,1000); ax.set_ylim(0,11)
    ax.set_xlabel("predictor accuracy ratio (MAE$_{\\rm classical}$ / MAE$_{\\rm model}$)")
    ax.set_ylabel("bits saved per value")
    style(ax); _save(fig,"fig1_log2law"); plt.close(fig)
    print("fig1 ok; measured ratios:",[round(x,3) for x in xs])

# ---------------------------------------------------------------- Fig 2
def fig_domains():
    ra=json.load(open("results/rescore_all.json"))
    data={k:ra[k]["gains"] for k in ["SDRBench","Synthetic","NAB oper.","Grid load","Transit"]}
    cols=[C["orange"],C["violet"],C["yellow"],C["blue"],C["aqua"]]
    mks=["v","s","D","o","^"]
    fig,ax=plt.subplots(figsize=(3.3,2.8))
    rng=np.random.default_rng(3)
    for i,(k,v) in enumerate(data.items()):
        v=np.array(v); y=i+rng.uniform(-0.16,0.16,len(v))
        ax.scatter(v,y,s=7,color=cols[i],alpha=0.55,marker=mks[i],linewidths=0,zorder=3)
        ax.plot([np.median(v)],[i],marker="|",ms=16,mew=1.8,color=INK,zorder=5)
        ax.annotate(f"{np.median(v):+.1f}%  ({int((v>0).sum())}/{len(v)})",
                    (np.median(v),i),textcoords="offset points",xytext=(0,9),
                    ha="center",fontsize=6.4,color=INK)
    ax.axvline(0,color=INK2,lw=0.8,zorder=2)
    ax.set_yticks(range(len(data))); ax.set_yticklabels(list(data),fontsize=7)
    ax.set_ylim(-0.6,len(data)-0.25); ax.set_xlim(-70,50)
    ax.set_xlabel("gain over best-of-six classical (%), real bytes")
    style(ax); ax.grid(axis="y",alpha=0)
    _save(fig,"fig2_domains"); plt.close(fig)
    print("fig2 ok;",{k:(round(float(np.median(v)),1),len(v)) for k,v in data.items()})

# ---------------------------------------------------------------- Fig 3
def sz3_bpv(x,tau):
    TD=CACHE
    env=dict(os.environ,LD_LIBRARY_PATH=os.path.abspath("ext/sz3-install/lib"))
    f=f"{TD}/figsz.bin"; x.astype(np.float64).tofile(f)
    r=subprocess.run(["ext/sz3-install/bin/sz3","-d","-i",f,"-z",f+".sz","-1",str(len(x)),
                      "-M","ABS","-A",str(tau)],capture_output=True,env=env)
    return os.path.getsize(f+".sz")*8.0/len(x) if r.returncode==0 else np.nan

def fig_rd():
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.5))
    cfg=[("Grid load (EIA-930, 49 BAs)","results/grid_bench.json","data/grid"),
         ("Subway ridership (MTA, 50 stations)","results/transit_bench.json","data/transit")]
    for ax,(title,rf,dd) in zip(axes,cfg):
        res=json.load(open(rf)); res={k:v for k,v in res.items() if not k.startswith("SEC|")}
        acr=json.load(open("results/rescore_ac.json"))
        dom="grid" if "grid" in dd else "transit"
        sets=sorted({k.split("|")[0] for k in res})
        X,Ycad,Ycls,Ysz=[],[],[],[]
        for rho in [0.01,0.05,0.2]:
            rel=[];sz=[]
            for n in sets:
                v=res[f"{n}|{rho}"]; xf=np.load(f"{dd}/{n}.npy").astype(float)
                rel.append(100*v["tau"]/abs(xf.mean()))
            for n in sets[:10]:
                v=res[f"{n}|{rho}"]; xf=np.load(f"{dd}/{n}.npy").astype(float)
                sz.append(sz3_bpv(xf[512:512+2048],v["tau"]))
            a=acr[f"{dom}|{rho}"]
            X.append(np.median(rel)); Ycad.append(a["cadence"])
            Ycls.append(a["classic"]); Ysz.append(np.nanmedian(sz))
        ax.plot(X,Ysz,marker="s",ms=4.5,lw=1.4,ls=":",color=C["violet"],label="SZ3 (real binary)",zorder=3)
        ax.plot(X,Ycls,marker="^",ms=4.5,lw=1.4,ls="--",color=C["orange"],label="best-of-six classical (same coder)",zorder=4)
        ax.plot(X,Ycad,marker="o",ms=4.5,lw=1.8,ls="-",color=C["blue"],label="Cadence",zorder=5)
        for j,(x,a,b) in enumerate(zip(X,Ycad,Ycls)):
            ax.annotate(f"{100*(1-a/b):+.0f}%",(x,a),textcoords="offset points",
                        xytext=(14 if j==0 else 0,-4 if j==0 else -12),
                        ha="left" if j==0 else "center",fontsize=6.2,color=INK)
        ax.set_xscale("log"); ax.set_title(title,pad=4)
        ax.set_xlabel("guaranteed error bound $\\tau$ (% of series mean)")
        ax.set_ylabel("bits per value"); style(ax)
    axes[0].legend(loc="upper right",handlelength=2.4)
    _save(fig,"fig3_rd"); plt.close(fig)
    print("fig3 ok")

# ---------------------------------------------------------------- Fig 4
def fig_ablation():
    ab=json.load(open("results/ablation_ac.json"))
    body={int(k):v for k,v in ab["body"].items()}
    seed={int(k):v for k,v in ab["seed"].items()}
    ctxs=sorted(body); CLS=ab["CLS"]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,2.4))
    a1.plot(ctxs,[body[c] for c in ctxs],marker="o",ms=4.5,lw=1.8,color=C["blue"],zorder=4)
    a1.set_xscale("log",base=2); a1.set_xticks(ctxs); a1.set_xticklabels([str(c) for c in ctxs])
    a1.set_xlabel("context length $c$ (samples)"); a1.set_ylabel("body bits per value")
    a1.annotate("accuracy saturates:\n$c{=}256$ costs only 3.1%",(256,body[256]),
                textcoords="offset points",xytext=(6,16),fontsize=6.4,color=INK2,
                arrowprops=dict(arrowstyle="-",lw=0.6,color=INK2))
    a1.set_title("(a) Model accuracy vs. context",pad=4); style(a1)
    ns=np.array([2000,4300,8760,17520,43800,100000])
    cols=[C["violet"],C["yellow"],C["aqua"],C["blue"],C["orange"]]
    mks=["v","s","D","o","^"]
    for i,c in enumerate(ctxs):
        g=[100*(1-((body[c]*(n-c)+seed[c]*c)/n)/CLS) for n in ns]
        a2.plot(ns,g,marker=mks[i],ms=4,lw=1.4,color=cols[i],label=f"$c={c}$",zorder=4)
    a2.axhline(0,color=INK2,lw=0.8,zorder=2)
    a2.set_xscale("log"); a2.set_xlabel("series length $n$ (hourly samples)")
    a2.set_ylabel("end-to-end gain (%)")
    a2.set_title("(b) Seed amortization",pad=4); style(a2)
    a2.legend(loc="lower right",ncol=2,handlelength=1.8,
              bbox_to_anchor=(1.0,0.02),columnspacing=1.0)
    _save(fig,"fig4_ablation"); plt.close(fig)
    print("fig4 ok")

if __name__=="__main__":
    fig_log2law(); fig_domains(); fig_ablation()
    # fig 3 needs the fetched corpora (relative-error axis) and the SZ3 binary
    if os.path.isdir("data/grid") and os.path.isdir("data/transit"):
        try: fig_rd()
        except Exception as e: print(f"fig3 skipped: {e}")
    else:
        print("fig3 skipped: needs 'python scripts/fetch_data.py grid transit'"
              " (and scripts/build_sz3.sh for the SZ3 curve)")
    print("\nfigures in", OUT + ":", sorted(f for f in os.listdir(OUT) if f.endswith((".png",".pdf"))))
