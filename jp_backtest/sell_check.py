"""売りシグナルでも、仲間(窓×上げ相場の週、相関上位200)の推定が後半の成績を予測できるか。SELL=1 python3 sell_check.py"""
import numpy as np
import pandas as pd

import stock_pool as sp
import stock_pool2 as p2
import stock_pool3 as p3

NS, NC, NT = sp.NS, sp.NC, sp.NT
rows = []
for h, frac in p3.SET:
    ts = int(NT * frac)
    E = sp.prep(h)
    tr = E["t"] < ts
    n, sm = sp.stock_mats(E, tr)
    te = (E["t"] >= ts) & ~np.isnan(E["x"])
    nt = np.zeros((NS, NC)); st = np.zeros((NS, NC))
    np.add.at(nt, (E["sid"][te], E["c"][te]), 1); np.add.at(st, (E["sid"][te], E["c"][te]), E["x"][te])
    mt = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
    corr, keep = p3.corr_any(ts, ret="gap", regime="up")
    corr0, keep0 = p3.corr_any(ts)
    zmode = p2.corr_matrix(ts)[1]; same_z = zmode[:, None] == zmode[None, :]
    est = {"自身": np.where(n > 0, sm / np.maximum(n, 1), np.nan),
           "窓×上げ相場・同規模200(採用)": p2.topk_pool(n, sm, corr, same_z, 200),
           "窓×上げ相場・規模問わず200": p2.topk_pool(n, sm, corr, np.ones((NC, NC), bool), 200),
           "終値の相関・規模問わず200(旧)": p2.topk_pool(n, sm, corr0, np.ones((NC, NC), bool), 200)}
    gs = np.zeros((NS, 3)); gn = np.zeros((NS, 3)); ok = tr & ~np.isnan(E["x"])
    np.add.at(gs, (E["sid"][ok], E["z"][ok]), E["x"][ok]); np.add.at(gn, (E["sid"][ok], E["z"][ok]), 1)
    est["同規模の全銘柄"] = (gs / np.maximum(gn, 1))[:, zmode]
    use = (n >= 3) & (nt >= 3) & keep[None, :] & keep0[None, :]
    for v in est.values():
        use &= ~np.isnan(v)
    base = float((mt[use] * nt[use]).sum() / nt[use].sum())
    for m, v in est.items():
        c = float(np.corrcoef(v[use], mt[use])[0, 1])
        top = np.argsort(-np.where(use, v, -1e9), axis=0)[:3]
        sel = np.zeros_like(use)
        for j in range(3):
            sel[top[j], np.arange(NC)] = True
        sel &= use
        rows.append(dict(h=h, frac=frac, method=m, corr=c, top3=float((mt[sel] * nt[sel]).sum() / nt[sel].sum()), allmean=base, pairs=int(use.sum())))
    print(h, frac, int(use.sum()), {r["method"]: round(r["corr"], 4) for r in rows[-5:]}, flush=True)
df = pd.DataFrame(rows); import os; df.to_csv("results_pool_sell.csv" if os.environ.get("SELL") == "1" else "results_pool_buy.csv", index=False, encoding="utf-8-sig")
print(df.groupby("method").agg(corr=("corr", "mean"), mn=("corr", "min"), mx=("corr", "max"), top3=("top3", "mean"), allmean=("allmean", "mean")).round(4))
