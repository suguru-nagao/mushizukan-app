"""保有期間(5/10/20/30日 ≒ 1/2/4/6週)ごとの、同じ騰落率の日との比較。シグナル一覧タブ用。

使い方: python3 hold_run.py candle:D chart:W ...
出力: results_{family}_{tf}_hold.csv(matched と同じ列 + 保有 = 本数)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import backtest as bt
import chart_matched as cm
import timeframe as tfm
from mtf_run import signals

HERE = os.path.dirname(os.path.abspath(__file__))
HOLDS = {"D": [5, 10, 20, 30], "W": [1, 2, 4, 6], "M": [1, 2, 3, 6]}


def hold_table(P, sigs, holds, universe=None, label="全銘柄", k=1):
    liq = bt.liquidity_bucket(P, universe=universe)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    prev = np.vstack([np.full((k, P["C"].shape[1]), np.nan), P["C"][:-k]])
    r0 = P["C"] / prev - 1
    nolim = ~P["limit"]
    fr = bt.forward_returns(P, holds)
    rows = []
    for h in holds:
        r = np.where(ok, fr[h], np.nan)
        x = r - bt.baseline(r, liq)
        for size, lo, hi in cm.SIZES:
            grp = (liq >= lo) & (liq <= hi) & nolim & ~np.isnan(x) & ~np.isnan(r0)
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
                rows.append(dict(対象=label, 基準日数=k, シグナル=name, 方向="買い" if d > 0 else "売り", 規模=size, 保有=h, n=len(t),
                                 素の超過bp=raw.mean() * 1e4, 素のt=bt.cluster_t(raw, t, max(h, 5)),
                                 同じ騰落率との差bp=adj.mean() * 1e4, 差のt=bt.cluster_t(adj, t, max(h, 5))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P0 = bt.load_panel()
    U0 = bt.load_margin(P0)
    cache = {"D": (P0, U0)}
    for job in sys.argv[1:]:
        fam, tf = job.split(":")
        if tf not in cache:
            cache[tf] = tfm.resample(P0, U0, tf)
        P, U = cache[tf]
        t0 = time.time()
        sigs = signals(P, fam, tf)
        df = pd.concat([hold_table(P, sigs, HOLDS[tf], None, "全銘柄"), hold_table(P, sigs, HOLDS[tf], U, "貸借のみ")], ignore_index=True)
        df.to_csv(os.path.join(HERE, f"results_{fam}_{tf}_hold.csv"), index=False, encoding="utf-8-sig")
        print(f"[{fam}:{tf}] hold rows={len(df)} {time.time() - t0:.0f}s", flush=True)
