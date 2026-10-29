"""週足のダブルトップ(売り)が、パターン定義の閾値に依存していないかの確認。

閾値を 1 つずつ動かして検出し直し、全規模・4週保有の方向付き超過(売り=サインの向き)と t 値を見る。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import chart_patterns as cp
import timeframe as tfm

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [("基準", {}), ("折り返し K=3", {"K": 3}), ("折り返し K=4", {"K": 4}), ("折り返し K=6", {"K": 6}),
        ("山の高さの差 2%", {"TOL": .02}), ("山の高さの差 5%", {"TOL": .05}),
        ("谷の深さ 3%以上", {"DEPTH": .03}), ("谷の深さ 8%以上", {"DEPTH": .08}),
        ("事前の上昇 3%", {"PRE": .03}), ("事前の上昇 10%", {"PRE": .10}),
        ("山の間隔 6〜90", {"MIN_SEP": 6}), ("山の間隔 15〜90", {"MIN_SEP": 15}),
        ("抜けを待つ 30本", {"BREAK_WINDOW": 30}), ("抜けを待つ 90本", {"BREAK_WINDOW": 90})]
DEF = {k: getattr(cp, k) for k in ["K", "TOL", "DEPTH", "PRE", "MIN_SEP", "BREAK_WINDOW"]}
H = 4

if __name__ == "__main__":
    P0 = bt.load_panel(); U0 = bt.load_margin(P0)
    P, U = tfm.resample(P0, U0, "W")
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    r = np.where(ok, bt.forward_returns(P, [H])[H], np.nan)
    rows = []
    for uni, universe in (("全銘柄", None), ("貸借のみ", U)):
        liq = bt.liquidity_bucket(P, universe=universe)
        x = r - bt.baseline(r, liq)
        small = (liq >= 1) & (liq <= 5) & ~P["limit"] & ~np.isnan(x)
        for label, over in GRID:
            for k, v in DEF.items():
                setattr(cp, k, v)
            for k, v in over.items():
                setattr(cp, k, v)
            sig = cp.detect_charts(P["H"], P["L"], P["C"])
            _, S = sig["ダブルトップ"]
            t, c = np.nonzero(S & small)
            v = -x[t, c]  # 売り
            rows.append(dict(対象=uni, 設定=label, n=len(t), 平均bp=v.mean() * 1e4, t=bt.cluster_t(v, t, 5),
                             前半bp=v[t < len(P["dates"]) // 2].mean() * 1e4, 後半bp=v[t >= len(P["dates"]) // 2].mean() * 1e4))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results_weekly_sens.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(res.pivot_table(index="設定", columns="対象", values=["n", "平均bp", "t", "前半bp", "後半bp"], sort=False).round(1).to_string())
