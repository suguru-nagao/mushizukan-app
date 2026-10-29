"""貸借銘柄(信用売りが可能な銘柄)だけに絞った再検証。

1) 全パターン × 流動性分位 × 保有期間のメイン検証を、貸借銘柄の中だけでやり直す(results_margin_main.csv)。
2) すべてのシグナルを「ショートした場合」の損益で評価し、全銘柄と貸借銘柄を並べて比べる(results_margin_short.csv)。
   ショートの損益 = -(個別リターン - 同日・同分位の平均)。約定の現実性のため、エントリー変種(翌日始値/1日遅らせる/日中のみ)と、
   シグナル日のストップ高/安の除外も見る。
貸借の判定は、各日付の時点の指定(月初マスター)を使う。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import micro_check as mc
from patterns import detect
from signals_extra import all_signals

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5)]


def signals(P):
    sigs = {k: v for k, v in all_signals(P).items()}
    for k, v in detect(P["O"], P["H"], P["L"], P["C"]).items():
        sigs[k] = v
    return sigs


def short_table(P, universe, label, sigs, split):
    liq = bt.liquidity_bucket(P, universe=universe)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    var = {k[0]: v for k, v in mc.variants(P).items()}
    var["G"] = bt.forward_returns(P, [5])[5]  # 保有5日: t+1始値→t+6始値
    nolim = ~P["limit"]
    rows = []
    for vname in ["A", "C", "D", "G"]:
        r = np.where(ok, var[vname], np.nan)
        x = r - bt.baseline(r, liq)
        for name, (d, S) in sigs.items():
            for size, lo, hi in SIZES:
                sel = S & nolim & (liq >= lo) & (liq <= hi) & ~np.isnan(x)
                t, c = np.nonzero(sel)
                if len(t) < 50:
                    continue
                v = -x[t, c]  # ショートした場合
                h = 5 if vname == "G" else 1
                s = bt.summarize(v, t, h)
                f, l = v[t < split], v[t >= split]
                rows.append(dict(対象=label, シグナル=name, 規模=size, 変種=vname, n=s["n"], 平均bp=s["mean_bp"], 中央値bp=s["median_bp"],
                                 t=s["t"], 前半bp=f.mean() * 1e4 if len(f) > 30 else np.nan,
                                 後半bp=l.mean() * 1e4 if len(l) > 30 else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = bt.load_panel()
    U = bt.load_margin(P)
    split = len(P["dates"]) // 2

    main, sp = bt.run(P, universe=U)
    main.to_csv(os.path.join(HERE, "results_margin_main.csv"), index=False, encoding="utf-8-sig")
    print("main saved", len(main))

    sigs = signals(P)
    both = pd.concat([short_table(P, None, "全銘柄", sigs, split), short_table(P, U, "貸借のみ", sigs, split)], ignore_index=True)
    both.to_csv(os.path.join(HERE, "results_margin_short.csv"), index=False, encoding="utf-8-sig")
    print("short saved", len(both))
