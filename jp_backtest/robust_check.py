"""signals_extra.py で大きく出たシグナルの頑健性確認。

平均 vs 中央値、外れ値処理、シグナル日のストップ高/安の除外、エントリー遅延、年別を見る。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import micro_check as mc
from signals_extra import all_signals

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["[探索]終値が高値圏(レンジ3%超)", "窓上げ2%超", "窓下げ2%超",
           "首吊り線|出来高急増", "大陽線|出来高急増", "赤三兵|出来高急増"]


def wins(x, p=1):
    lo, hi = np.percentile(x, [p, 100 - p])
    return np.clip(x, lo, hi)


if __name__ == "__main__":
    P = bt.load_panel()
    liq = bt.liquidity_bucket(P)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    sigs = all_signals(P)
    years = P["dates"].year.to_numpy()
    var = mc.variants(P)
    rows, yrows = [], []
    for name in TARGETS:
        d, S = sigs[name]
        for size, lo, hi in [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5)]:
            base_sel = S & (liq >= lo) & (liq <= hi)
            for vname, r in var.items():
                r = np.where(ok, r, np.nan)
                x = r - bt.baseline(r, liq)
                for tag, sel in [("全て", base_sel), ("当日ストップ高/安を除く", base_sel & ~P["limit"])]:
                    s = sel & ~np.isnan(x)
                    t, c = np.nonzero(s)
                    v = d * x[t, c]
                    if len(v) < 50:
                        continue
                    rows.append(dict(シグナル=name, 規模=size, 変種=vname, 条件=tag, n=len(v),
                                     平均bp=v.mean() * 1e4, 中央値bp=np.median(v) * 1e4,
                                     丸め平均bp=wins(v).mean() * 1e4,
                                     t=bt.cluster_t(v, t, 5)))
            # 年別(基準変種A, 当日ストップ高/安を除く)
            r = np.where(ok, var["A 基準: t+1始値→t+2始値"], np.nan)
            x = r - bt.baseline(r, liq)
            s = base_sel & ~P["limit"] & ~np.isnan(x)
            t, c = np.nonzero(s)
            v = d * x[t, c]
            for y in np.unique(years[t]):
                m = years[t] == y
                yrows.append(dict(シグナル=name, 規模=size, 年=int(y), n=int(m.sum()), 平均bp=v[m].mean() * 1e4))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results_robust.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(yrows).to_csv(os.path.join(HERE, "results_robust_year.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for name in TARGETS:
        print(f"\n##### {name}")
        g = res[(res.シグナル == name) & (res.規模 != "大型(分位4-5)")]
        print(g[["規模", "変種", "条件", "n", "平均bp", "中央値bp", "丸め平均bp", "t"]].round(1).to_string(index=False))
        y = pd.DataFrame(yrows)
        y = y[y.シグナル == name].pivot_table(index="規模", columns="年", values="平均bp").round(1)
        print("年別(基準A, 当日ストップ高/安を除く) 平均bp:\n", y.to_string())
