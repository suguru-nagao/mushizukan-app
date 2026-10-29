"""小型株の逆ハンマー(買い)/首吊り線(売り)の効果が、板の構造による見かけでないかの確認。

見かけ(気配のバウンス等)なら、1 日遅らせる/終値で入ると消え、低位株(呼値の比率が大きい)に集中するはず。
本物の価格形成なら、遅らせても残り、株価帯に依存しにくい。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
from patterns import detect

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["逆ハンマー", "首吊り線", "ハンマー", "流れ星"]


def variants(P):
    O, C = P["O"], P["C"]
    a = bt.ahead
    return {
        "A 基準: t+1始値→t+2始値": a(O, 2) / a(O, 1) - 1,
        "B t+1終値→t+2終値": a(C, 2) / a(C, 1) - 1,
        "C 1日遅らせる: t+2始値→t+3始値": a(O, 3) / a(O, 2) - 1,
        "D 日中のみ: t+1始値→t+1終値": a(C, 1) / a(O, 1) - 1,
        "E 窓(参考・約定不可): t終値→t+1始値": a(O, 1) / C - 1,
        "F 2日後から: t+2終値→t+3終値": a(C, 3) / a(C, 2) - 1,
    }


if __name__ == "__main__":
    P = bt.load_panel()
    liq = bt.liquidity_bucket(P)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    pats = detect(P["O"], P["H"], P["L"], P["C"])
    rows = []
    for vname, r in variants(P).items():
        r = np.where(ok, r, np.nan)
        x = r - bt.baseline(r, liq)
        for name in KEY:
            d, S = pats[name]
            for size, lo, hi in [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5)]:
                sel = S & (liq >= lo) & (liq <= hi) & ~np.isnan(x)
                t, c = np.nonzero(sel)
                v = d * x[t, c]
                s = bt.summarize(v, t, 1)
                rows.append(dict(パターン=name, 規模=size, 変種=vname, **s))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results_micro_variants.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for name in KEY[:2]:
        print(f"\n=== {name}: 変種別(平均超過bp / t) ===")
        t = res[res.パターン == name].pivot_table(index="変種", columns="規模", values=["mean_bp", "t"])
        print(pd.DataFrame({k: t["mean_bp"][k].round(1).astype(str) + " (" + t["t"][k].round(1).astype(str) + ")" for k in ["小型(分位1-2)", "中型(分位3)", "大型(分位4-5)"]}).to_string())

    # 株価帯別(未調整終値)。基準変種 A、小型株のみ
    r = variants(P)["A 基準: t+1始値→t+2始値"]
    r = np.where(ok, r, np.nan)
    x = r - bt.baseline(r, liq)
    px = P["Craw"]
    bins = [0, 300, 1000, 3000, 1e9]
    labels = ["<300円", "300-1000円", "1000-3000円", ">3000円"]
    prow = []
    for name in KEY[:2]:
        d, S = pats[name]
        for lab, lo, hi in zip(labels, bins[:-1], bins[1:]):
            sel = S & (liq <= 2) & (liq >= 1) & (px >= lo) & (px < hi) & ~np.isnan(x)
            t, c = np.nonzero(sel)
            prow.append(dict(パターン=name, 株価帯=lab, **bt.summarize(d * x[t, c], t, 1)))
    pr = pd.DataFrame(prow)
    pr.to_csv(os.path.join(HERE, "results_micro_price.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 小型株の株価帯別(基準変種) ===")
    print(pr[["パターン", "株価帯", "n", "mean_bp", "t", "win"]].round(2).to_string(index=False))
