"""パターン定義の閾値を 1 つずつ動かして、結果が変わらないかを確認する。

見るもの: 小型株(売買代金分位1-2)での、方向付き超過リターンと t 値。
基準と同じ結論が閾値に依らず出るか。出ないなら閾値次第の見かけの結果。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import patterns as pt

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["逆ハンマー", "首吊り線", "ハンマー", "流れ星", "大陽線", "大陰線", "陽の包み足", "陰の包み足"]
GRID = [
    ("基準", {}),
    ("TREND_LOOKBACK=3", {"TREND_LOOKBACK": 3}),
    ("TREND_LOOKBACK=10", {"TREND_LOOKBACK": 10}),
    ("TREND_LOOKBACK=20", {"TREND_LOOKBACK": 20}),
    ("トレンド条件なし", {"REQUIRE_TREND": False}),
    ("SMALL_BODY=0.2", {"SMALL_BODY": 0.2}),
    ("SMALL_BODY=0.5", {"SMALL_BODY": 0.5}),
    ("SHADOW_BODY_MULT=1.5", {"SHADOW_BODY_MULT": 1.5}),
    ("SHADOW_BODY_MULT=3", {"SHADOW_BODY_MULT": 3.0}),
    ("SHORT_SHADOW=0.05", {"SHORT_SHADOW": 0.05}),
    ("SHORT_SHADOW=0.2", {"SHORT_SHADOW": 0.2}),
    ("LONG_BODY=0.5", {"LONG_BODY": 0.5}),
    ("LONG_BODY=0.7", {"LONG_BODY": 0.7}),
]
DEFAULTS = {k: getattr(pt, k) for k in ["TREND_LOOKBACK", "SMALL_BODY", "SHADOW_BODY_MULT", "SHORT_SHADOW", "LONG_BODY", "REQUIRE_TREND"]}

if __name__ == "__main__":
    P = bt.load_panel()
    liq = bt.liquidity_bucket(P)
    rets = bt.forward_returns(P, [1, 5])
    xs = {h: rets[h] - bt.baseline(rets[h], liq) for h in (1, 5)}
    small = (liq >= 1) & (liq <= 2)
    rows = []
    for label, over in GRID:
        for k, v in DEFAULTS.items():
            setattr(pt, k, v)
        for k, v in over.items():
            setattr(pt, k, v)
        pats = pt.detect(P["O"], P["H"], P["L"], P["C"])
        for name in KEY:
            d, S = pats[name]
            for h in (1, 5):
                sel = S & small & ~np.isnan(xs[h])
                t, c = np.nonzero(sel)
                rows.append(dict(設定=label, パターン=name, 保有日数=h, **bt.summarize(d * xs[h][t, c], t, max(h, 1))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results_sensitivity.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for h in (1, 5):
        t = res[res.保有日数 == h].pivot_table(index="設定", columns="パターン", values=["mean_bp", "t"], sort=False)
        print(f"\n=== 小型株・保有{h}日: 平均超過bp (t値) ===")
        print(pd.DataFrame({k: t["mean_bp"][k].round(1).astype(str) + " (" + t["t"][k].round(1).astype(str) + ")" for k in KEY}).to_string())
