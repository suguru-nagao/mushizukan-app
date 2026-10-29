"""チャートパターンの検証(ローソク足と同じ枠組み)。

1) 全パターン × 流動性分位 × 保有期間のメイン検証(全銘柄 / 貸借のみ)。
2) 変種(翌日始値 / 1日遅らせる / 日中のみ / 5日保有)で、サイン方向の損益と、逆方向(反転)の損益を比べる。
   当日ストップ高/安のシグナルは除く。方向はパターンの抜けた向き(上抜け=買い、下抜け=売り)。
"""
import os
import pickle

import numpy as np
import pandas as pd

import backtest as bt
import micro_check as mc

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5)]


def variants_table(P, universe, label, sigs, split):
    liq = bt.liquidity_bucket(P, universe=universe)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    var = {k[0]: v for k, v in mc.variants(P).items()}
    var["G"] = bt.forward_returns(P, [5])[5]
    nolim = ~P["limit"]
    rows = []
    for vname in ["A", "C", "D", "G"]:
        r = np.where(ok, var[vname], np.nan)
        x = r - bt.baseline(r, liq)
        h = 5 if vname == "G" else 1
        for name, (d, S) in sigs.items():
            for size, lo, hi in SIZES:
                sel = S & nolim & (liq >= lo) & (liq <= hi) & ~np.isnan(x)
                t, c = np.nonzero(sel)
                if len(t) < 50:
                    continue
                v = d * x[t, c]
                s = bt.summarize(v, t, h)
                f, l = v[t < split], v[t >= split]
                rows.append(dict(対象=label, シグナル=name, 方向="買い" if d > 0 else "売り", 規模=size, 変種=vname, n=s["n"],
                                 平均bp=s["mean_bp"], 中央値bp=s["median_bp"], t=s["t"],
                                 前半bp=f.mean() * 1e4 if len(f) > 30 else np.nan, 後半bp=l.mean() * 1e4 if len(l) > 30 else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = bt.load_panel()
    U = bt.load_margin(P)
    sigs = pickle.load(open(os.path.join(bt.CACHE, "chart_sig.pkl"), "rb"))
    split = len(P["dates"]) // 2

    m, _ = bt.run(P, pats=sigs)
    m.to_csv(os.path.join(HERE, "results_chart_main.csv"), index=False, encoding="utf-8-sig")
    mm, _ = bt.run(P, universe=U, pats=sigs)
    mm.to_csv(os.path.join(HERE, "results_chart_margin_main.csv"), index=False, encoding="utf-8-sig")
    both = pd.concat([variants_table(P, None, "全銘柄", sigs, split), variants_table(P, U, "貸借のみ", sigs, split)], ignore_index=True)
    both.to_csv(os.path.join(HERE, "results_chart_variants.csv"), index=False, encoding="utf-8-sig")
    print("saved", len(m), len(mm), len(both))
