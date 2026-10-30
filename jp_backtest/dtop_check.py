"""週足ダブルトップ(売り)の、長期データ(2000年〜)での実コストと、相場局面別の確認。
実コスト: 1ティックのスプレッド(TOPIX100以外の呼値表 ÷ 株価。株価は分割調整後で、古い時期はやや過大 = コストは控えめでなく保守的) + 貸株料(年1.15%を保有の暦日で按分)。
逆日歩は含めない。貸借区分は、現在の指定で代用(先読み)。
相場局面は、全銘柄の等ウェイト指数(週次)から、シグナルが出た時点で分かる値で区分する。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
from chart_patterns import detect_charts
from cost_model import LEND_FEE, tick_size
from long_run import load

HERE = os.path.dirname(os.path.abspath(__file__))
HOLDS = [4, 8, 13]
SIZES = [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5), ("全規模(分位1-5)", 1, 5)]

P, U = load("W")
dates = P["dates"]; T = len(dates); years = dates.year.to_numpy()
C = P["C"]
ret = np.full_like(C, np.nan); ret[1:] = C[1:] / C[:-1] - 1
mret = np.nanmean(np.clip(ret, -0.5, 0.5), axis=1); mret[0] = 0
idx = np.cumprod(1 + np.nan_to_num(mret))
tr26 = np.full(T, np.nan); tr26[26:] = idx[26:] / idx[:-26] - 1
hi52 = pd.Series(idx).rolling(52, min_periods=26).max().to_numpy(); dd = idx / hi52 - 1
vol13 = pd.Series(mret).rolling(13, min_periods=8).std().to_numpy()
vmed = np.nanmedian(vol13)
sig = detect_charts(P["H"], P["L"], P["C"])["ダブルトップ"][1]
tick_rel = tick_size(P["Craw"]) / P["Craw"]
cal = {h: np.array([(dates[min(i + 1 + h, T - 1)] - dates[min(i + 1, T - 1)]).days for i in range(T)]) for h in HOLDS}
REG = {
    "上昇局面(直近26週の市場が上昇)": tr26 > 0, "下落局面(直近26週の市場が下落)": tr26 <= 0,
    "高値圏(52週高値から10%以内)": dd > -0.10, "調整・下落(52週高値から10%超下)": dd <= -0.10,
    "変動が大きい(直近13週)": vol13 > vmed, "変動が小さい(直近13週)": vol13 <= vmed,
}
ERA = [("2000〜2007年", 2000, 2007), ("2008〜2012年", 2008, 2012), ("2013〜2019年", 2013, 2019), ("2020〜2026年", 2020, 2026)]
rows = []
for uni, universe in (("全銘柄", None), ("貸借のみ", U)):
    liq = bt.liquidity_bucket(P, universe=universe)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    for h in HOLDS:
        r = np.where(ok, bt.forward_returns(P, [h])[h], np.nan)
        x = r - bt.baseline(r, liq)
        for size, lo, hi in SIZES:
            sel = sig & ~P["limit"] & (liq >= lo) & (liq <= hi) & ~np.isnan(x) & ~np.isnan(tick_rel)
            t, c = np.nonzero(sel)
            if len(t) < 50:
                continue
            gross = -x[t, c]; tk = tick_rel[t, c]; fee = LEND_FEE * cal[h][t] / 365
            net1, net2 = gross - tk - fee, gross - 2 * tk - fee
            groups = [("全期間", np.ones(len(t), bool))]
            groups += [(f"{y}年", years[t] == y) for y in range(2000, 2027)]
            groups += [(n, (years[t] >= a) & (years[t] <= b)) for n, a, b in ERA]
            groups += [(n, np.nan_to_num(m[t].astype(float)) > 0) for n, m in REG.items()]
            for gname, m in groups:
                if m.sum() < 30:
                    continue
                rows.append(dict(対象=uni, 保有週=h, 規模=size, 区分=gname, n=int(m.sum()), 粗利bp=gross[m].mean() * 1e4, 粗利t=bt.cluster_t(gross[m], t[m], h),
                                 呼値bp=tk[m].mean() * 1e4, 貸株料bp=fee[m].mean() * 1e4, 純1tick_bp=net1[m].mean() * 1e4, 純1tick_t=bt.cluster_t(net1[m], t[m], h),
                                 純2tick_bp=net2[m].mean() * 1e4, 勝率=(net1[m] > 0).mean()))
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "results_dtop_regime.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print(df[(df.区分 == "全期間")].round(1).to_string(index=False))
