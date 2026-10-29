"""市場・業種・銘柄で絞ったときの、シグナルごとの成績(シグナル一覧タブ用)。

基準(同日・同規模の全銘柄平均と、同じ騰落率の日の平均)は全銘柄で作ったまま、シグナルが出た銘柄だけを市場・業種・銘柄で絞って集計する。
使い方: python3 subset_run.py candle:D chart:W ...
出力: report_data/sub_{tf}.json(市場・業種別。行は配列)、report_data/stk_{tf}_{番号}.json(銘柄別、100銘柄ごと)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import backtest as bt
import chart_matched as cm
import timeframe as tfm
from build_report import ZPREF, zname
from hold_run import HOLDS
from mtf_run import signals

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_data")
MK = {"0111": "プライム", "0112": "スタンダード", "0113": "グロース"}
SHARD = 400


def fam_of(name):
    for fam, pre in ZPREF:
        if name.startswith(pre):
            return fam
    return None


def run(P, U, fam0, tf, store):
    holds = HOLDS[tf]
    liq = bt.liquidity_bucket(P)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    prev = np.vstack([np.full((1, P["C"].shape[1]), np.nan), P["C"][:-1]])
    r0 = P["C"] / prev - 1
    nolim = ~P["limit"]
    fr = bt.forward_returns(P, holds)
    master = P["master"].loc[P["codes"]]
    mk = np.array([{"0111": 0, "0112": 1, "0113": 2}.get(m, -1) for m in P["mkt"]])
    secs = sorted(master["S33Nm"].dropna().unique())
    sec = master["S33Nm"].map({s: i for i, s in enumerate(secs)}).fillna(-1).astype(int).to_numpy()
    sigs = signals(P, fam0, tf)
    nstk = len(P["codes"])
    rows, stk = [], {}
    for hi, h in enumerate(holds):
        r = np.where(ok, fr[h], np.nan)
        x = r - bt.baseline(r, liq)
        for zi, (size, lo, hi_) in enumerate(cm.SIZES):
            grp = (liq >= lo) & (liq <= hi_) & nolim & ~np.isnan(x) & ~np.isnan(r0)
            edges = np.unique(np.percentile(r0[grp], cm.QS))
            bins = np.digitize(r0, edges[1:-1])
            base = np.full(len(edges) + 1, np.nan)
            for b in np.unique(bins[grp]):
                base[b] = x[grp & (bins == b)].mean()
            xm = x - base[bins]
            for name, (d, S) in sigs.items():
                t, c = np.nonzero(S & grp)
                if len(t) < 50:
                    continue
                raw, adj = d * x[t, c], d * xm[t, c]
                fam = fam_of(name) or fam0
                nm = zname(name)
                a = stk.setdefault((fam, nm), np.zeros((len(holds), 3, nstk)))
                a[hi, 0] += np.bincount(c, minlength=nstk)
                a[hi, 1] += np.bincount(c, weights=raw, minlength=nstk)
                a[hi, 2] += np.bincount(c, weights=adj, minlength=nstk)
                blk = max(h, 5)
                for uv, um in ((0, np.ones(len(t), bool)), (1, U[t, c])):
                    for kind, gid in ((0, mk[c]), (1, sec[c])):
                        g = np.where(um, gid, -1)
                        order = np.argsort(g, kind="stable")
                        gs = g[order]
                        cuts = np.flatnonzero(np.diff(gs)) + 1
                        for part in np.split(order, cuts):
                            if len(part) < 30 or g[part[0]] < 0:
                                continue
                            rr, aa, tt = raw[part], adj[part], t[part]
                            rows.append([fam, nm, d > 0, uv, kind, int(g[part[0]]), zi, h, len(part),
                                         round(rr.mean() * 1e4, 1), round(float(bt.cluster_t(rr, tt, blk)), 1),
                                         round(aa.mean() * 1e4, 1), round(float(bt.cluster_t(aa, tt, blk)), 1)])
    return rows, stk, secs, holds, master


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    P0 = bt.load_panel()
    U0 = bt.load_margin(P0)
    cache = {"D": (P0, U0)}
    acc = {}
    for job in sys.argv[1:]:
        fam0, tf = job.split(":")
        if tf not in cache:
            cache[tf] = tfm.resample(P0, U0, tf)
        P, U = cache[tf]
        t0 = time.time()
        rows, stk, secs, holds, master = run(P, U, fam0, tf, None)
        a = acc.setdefault(tf, {"rows": [], "stk": {}, "secs": secs, "holds": holds, "master": master, "codes": list(P["codes"])})
        a["rows"] += rows
        a["stk"].update(stk)
        print(f"[{fam0}:{tf}] rows={len(rows)} {time.time() - t0:.0f}s", flush=True)
        # 途中経過を毎回書き出す
        json.dump({"holds": holds, "mkts": list(MK.values()), "secs": secs, "rows": a["rows"]}, open(os.path.join(OUT, f"sub_{tf}.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
        keys = sorted(a["stk"])
        n = len(a["codes"])
        for s0 in range(0, n, SHARD):
            sh = {}
            for j in range(s0, min(s0 + SHARD, n)):
                ent = {}
                for (fam, nm) in keys:
                    arr = a["stk"][(fam, nm)][:, :, j]
                    if arr[:, 0].sum() == 0:
                        continue
                    ent[fam + "|" + nm] = [[int(arr[k, 0]), round(arr[k, 1] / arr[k, 0] * 1e4, 1) if arr[k, 0] else None, round(arr[k, 2] / arr[k, 0] * 1e4, 1) if arr[k, 0] else None] for k in range(len(holds))]
                sh[a["codes"][j]] = ent
            json.dump(sh, open(os.path.join(OUT, f"stk_{tf}_{s0 // SHARD}.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
