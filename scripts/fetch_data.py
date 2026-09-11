#!/usr/bin/env python3
"""Fetch every corpus used in the paper. Data is NOT committed to the repo:
the SDRBench archives alone are ~8 GB compressed.

  python scripts/fetch_data.py --all
  python scripts/fetch_data.py grid transit      # the two headline corpora

Corpora
  synth    generated locally, no network             (~2 MB)
  grid     EIA-930 hourly demand, 50 US BAs, 2026    (~46 MB)
  transit  MTA subway hourly ridership, 2026         (~90 MB, paginated API)
  wiki     Wikimedia hourly pageviews (CONTAMINATED) (~1 MB)
  nab      Numenta Anomaly Benchmark                 (~2 MB)
  sdrb     SDRBench EXAALT + Hurricane ISABEL        (~8 GB)
"""
import argparse, collections, csv, io, json, os, sys, tarfile, urllib.parse, urllib.request
import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.environ.get("CADENCE_DATA",os.path.join(ROOT,"data"))
UA={"User-Agent":"cadence-repro/1.0 (research)"}
def get(u,t=300): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t).read()
def out(sub): 
    d=os.path.join(DATA,sub); os.makedirs(d,exist_ok=True); return d

def synth():
    sys.path.insert(0,os.path.join(ROOT,"cadence")); import corpus
    corpus.build(out("synth")); print("synth: 10 series")

def grid():
    d=out("grid"); f=os.path.join(d,"EIA930_BALANCE_2026_Jan_Jun.csv")
    if not os.path.exists(f):
        open(f,"wb").write(get("https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
                               "EIA930_BALANCE_2026_Jan_Jun.csv"))
    rows=list(csv.DictReader(open(f,newline='',encoding='utf-8-sig')))
    by=collections.defaultdict(list)
    for r in rows:
        v=r.get("Demand (MW)","").replace(",","").strip() or r.get("Demand (MW) (Imputed)","").replace(",","").strip()
        by[r["Balancing Authority"]].append(v)
    k=0
    for ba,vs in by.items():
        a=np.array([float(v) if v not in("","NA") else np.nan for v in vs])
        if np.isnan(a).mean()>0.02 or len(a)<3000: continue
        i=np.arange(len(a)); m=~np.isnan(a)
        a=np.rint(np.interp(i,i[m],a[m])).astype(np.int64)
        if a.max()-a.min()<10: continue
        np.save(os.path.join(d,f"{ba}.npy"),a); k+=1
    print(f"grid: {k} balancing authorities  (NB: SEC is corrupt, excluded in analysis)")

def transit():
    d=out("transit"); rows=[]; last="2026-01-01T00:00:00"
    for _ in range(5):
        q={"$select":"transit_timestamp,station_complex_id,sum(ridership) as r",
           "$group":"transit_timestamp,station_complex_id",
           "$where":f"transit_timestamp > '{last}'","$order":"transit_timestamp","$limit":"500000"}
        chunk=json.loads(get("https://data.ny.gov/resource/5wq4-mkjj.json?"+urllib.parse.urlencode(q)))
        if not chunk: break
        rows+=chunk; last=max(r["transit_timestamp"] for r in chunk)
        print(f"  +{len(chunk)} rows through {last}",flush=True)
    ts=sorted({r["transit_timestamp"] for r in rows}); ti={t:i for i,t in enumerate(ts)}
    by=collections.defaultdict(lambda: np.zeros(len(ts)))
    for r in rows: by[r["station_complex_id"]][ti[r["transit_timestamp"]]]+=float(r["r"])
    full={k:v for k,v in by.items() if (v>0).mean()>0.95 and v.sum()>50000}
    for _,k in sorted(((v.sum(),k) for k,v in full.items()),reverse=True)[:50]:
        np.save(os.path.join(d,f"st{k}.npy"),np.rint(full[k]).astype(np.int64))
    print(f"transit: 50 stations x {len(ts)} hours")

def wiki():
    d=out("real")
    for proj in ["en.wikipedia","de.wikipedia"]:
        v=json.loads(get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
                         f"{proj}/all-access/user/hourly/2024010100/2026080100"))
        a=np.array([i["views"] for i in v["items"]],dtype=np.int64)
        np.save(os.path.join(d,f"wikihr_{proj.split('.')[0]}.npy"),a)
    print("wiki: 2 series  (CONTAMINATED: in TimesFM pretraining to Nov 2023)")

def nab():
    d=out("nab"); base="https://raw.githubusercontent.com/numenta/NAB/master/data/"
    files=["realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv","realAWSCloudwatch/ec2_network_in_5abac7.csv",
           "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv","realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
           "realAWSCloudwatch/grok_asg_anomaly.csv","realKnownCause/nyc_taxi.csv",
           "realKnownCause/ambient_temperature_system_failure.csv",
           "realKnownCause/machine_temperature_system_failure.csv"]
    for f in files:
        v=np.array([float(r["value"]) for r in csv.DictReader(io.StringIO(get(base+f).decode()))])
        sc=10**max(0,4-int(np.floor(np.log10(max(v.max()-v.min(),1e-9)))))
        np.save(os.path.join(d,os.path.basename(f)[:-4]+".npy"),np.rint(v*sc).astype(np.int64))
    print(f"nab: {len(files)} series")

def sdrb():
    d=out("sdrb"); B="https://g-d0cd3f.fd635.8443.data.globus.org/raw-data/"
    for path,name in [("EXAALT/SDRBENCH-EXAALT-2869440.tar.gz","exaalt"),
                      ("Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-Pf.tar.gz","isabel_Pf")]:
        tgz=os.path.join(d,name+".tar.gz")
        if not os.path.exists(tgz):
            print(f"  downloading {name} (this is large)",flush=True)
            open(tgz,"wb").write(get(B+path,900))
        with tarfile.open(tgz) as t: t.extractall(os.path.join(d,name))
    print("sdrb: EXAALT + Hurricane ISABEL")

ALL={"synth":synth,"grid":grid,"transit":transit,"wiki":wiki,"nab":nab,"sdrb":sdrb}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("corpora",nargs="*"); ap.add_argument("--all",action="store_true")
    a=ap.parse_args(); want=list(ALL) if a.all or not a.corpora else a.corpora
    for c in want:
        if c not in ALL: sys.exit(f"unknown corpus {c}; choose from {list(ALL)}")
        print(f"== {c} =="); ALL[c]()
    print("\ndata in", DATA)
