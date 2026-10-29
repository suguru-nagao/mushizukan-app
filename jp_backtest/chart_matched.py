"""チャートパターンの効果が「その日の大きな値動き」だけで説明できるかの確認。

ネックラインの抜けは、大きく動いた日に起きやすい。同じ流動性グループ・同じ水準のシグナル日の騰落率を持つ「全銘柄・全日」の平均を基準にして、
パターンが出た日の超過リターンを比べる。基準との差が残れば、パターン独自の情報がある。
"""
import os
import pickle

import numpy as np
import pandas as pd

import backtest as bt
import micro_check as mc

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5)]
QS = [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 92.5, 95, 96, 97, 98, 99, 99.5, 100]


def matched_table(P, sigs, universe=None, label="全銘柄"):
    liq = bt.liquidity_bucket(P, universe=universe)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    r0 = P["C"] / bt.ahead(P["C"], 0) - 1  # placeholder shape
    prev = np.vstack([np.full((1, P["C"].shape[1]), np.nan), P["C"][:-1]])
    r0 = P["C"] / prev - 1
    var = {k[0]: v for k, v in mc.variants(P).items()}
    var["G"] = bt.forward_returns(P, [5])[5]
    nolim = ~P["limit"]
    rows = []
    for vname in ["A", "G"]:
        r = np.where(ok, var[vname], np.nan)
        x = r - bt.baseline(r, liq)
        for size, lo, hi in SIZES:
            grp = (liq >= lo) & (liq <= hi) & nolim & ~np.isnan(x) & ~np.isnan(r0)
            edges = np.unique(np.percentile(r0[grp], QS))
            bins = np.digitize(r0, edges[1:-1])
            base = np.full(len(edges) + 1, np.nan)
            for b in np.unique(bins[grp]):
                m = grp & (bins == b)
                base[b] = x[m].mean()
            xm = x - base[bins]
            for name, (d, S) in sigs.items():
                sel = S & grp
                t, c = np.nonzero(sel)
                if len(t) < 50:
                    continue
                raw, adj = d * x[t, c], d * xm[t, c]
                h = 5 if vname == "G" else 1
                rows.append(dict(対象=label, シグナル=name, 方向="買い" if d > 0 else "売り", 規模=size, 変種=vname, n=len(t),
                                 シグナル日の平均騰落率pct=np.nanmean(r0[t, c]) * 100,
                                 素の超過bp=raw.mean() * 1e4, 素のt=bt.cluster_t(raw, t, max(h, 5)),
                                 同じ騰落率との差bp=adj.mean() * 1e4, 差のt=bt.cluster_t(adj, t, max(h, 5))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = bt.load_panel()
    U = bt.load_margin(P)
    sigs = pickle.load(open(os.path.join(bt.CACHE, "chart_sig.pkl"), "rb"))
    res = pd.concat([matched_table(P, sigs), matched_table(P, sigs, U, "貸借のみ")], ignore_index=True)
    res.to_csv(os.path.join(HERE, "results_chart_matched.csv"), index=False, encoding="utf-8-sig")
    print("saved", len(res))
