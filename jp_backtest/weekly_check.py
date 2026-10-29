"""週足で有望に見えたシグナルの頑健性確認(年別・特定の週への集中・実コスト)。

対象: 週足のダブルトップ(売り)、52週の新安値/新高値、一目の好転(買い)。保有 = 4週(次の週の始値 → 4 週後の始値)。
実コスト: 1 ティックのスプレッド + 貸株料(年 1.15% を保有の暦日で按分)。ショートのみ(売りサイン)。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import timeframe as tfm
from chart_patterns import detect_charts
from cost_model import LEND_FEE, tick_size
from tech_signals import detect_tech
from zones import TF_PARAMS, detect_zones

HERE = os.path.dirname(os.path.abspath(__file__))
H = 4
CAND = [("chart", "ダブルトップ", -1), ("zone", "新安値更新(長期・52本)", -1), ("zone", "新高値更新(長期・52本)", +1), ("tech", "一目: 転換線が基準線を上抜け(好転)", +1)]
SIZES = [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5), ("全規模(分位1-5)", 1, 5)]

if __name__ == "__main__":
    P0 = bt.load_panel(); U0 = bt.load_margin(P0)
    P, U = tfm.resample(P0, U0, "W")
    sigs = {"chart": detect_charts(P["H"], P["L"], P["C"]), "zone": detect_zones(P["H"], P["L"], P["C"], P["Vo"], *TF_PARAMS["W"]),
            "tech": detect_tech(P["O"], P["H"], P["L"], P["C"])}
    dates = P["dates"]; years = dates.year.to_numpy(); T = len(dates)
    tick_rel = tick_size(P["Craw"]) / P["Craw"]
    cal = np.array([(dates[min(i + 1 + H, T - 1)] - dates[min(i + 1, T - 1)]).days for i in range(T)])
    rows, yrows = [], []
    for uni, universe in (("全銘柄", None), ("貸借のみ", U)):
        liq = bt.liquidity_bucket(P, universe=universe)
        ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
        r = np.where(ok, bt.forward_returns(P, [H])[H], np.nan)
        x = r - bt.baseline(r, liq)
        for fam, name, d in CAND:
            _, S = sigs[fam][name]
            for size, lo, hi in SIZES:
                sel = S & ~P["limit"] & (liq >= lo) & (liq <= hi) & ~np.isnan(x) & ~np.isnan(tick_rel)
                t, c = np.nonzero(sel)
                if len(t) < 50:
                    continue
                gross = d * x[t, c]
                tk, fee = tick_rel[t, c], LEND_FEE * cal[t] / 365
                # 買いは貸株料なし(現物)。売りだけ貸株料を引く
                fee_used = fee if d < 0 else 0 * fee
                row = dict(対象=uni, シグナル=name, 方向="売り" if d < 0 else "買い", 規模=size, n=len(t), 粗利bp=gross.mean() * 1e4, t=bt.cluster_t(gross, t, 5),
                           呼値比率bp=tk.mean() * 1e4, 貸株料bp=fee_used.mean() * 1e4)
                for lab, cost in [("コスト20bp固定", 0.002), ("呼値1枚+貸株料", tk + fee_used), ("呼値2枚+貸株料", 2 * tk + fee_used)]:
                    net = gross - cost
                    row[f"{lab}_bp"], row[f"{lab}_t"] = net.mean() * 1e4, bt.cluster_t(net, t, 5)
                # 件数が最も多い 3 週を除いた平均(特定の相場局面への集中を見る)
                wk = pd.Series(t).value_counts()
                keep = ~np.isin(t, wk.index[:3])
                row["上位3週を除く平均bp"] = gross[keep].mean() * 1e4
                row["上位3週の件数比"] = 1 - keep.mean()
                rows.append(row)
                for y in np.unique(years[t]):
                    m = years[t] == y
                    yrows.append(dict(対象=uni, シグナル=name, 規模=size, 年=int(y), n=int(m.sum()), 平均bp=gross[m].mean() * 1e4))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(HERE, "results_weekly_check.csv"), index=False, encoding="utf-8-sig")
    yr = pd.DataFrame(yrows); yr.to_csv(os.path.join(HERE, "results_weekly_check_year.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 30)
    print(res[["対象", "シグナル", "方向", "規模", "n", "粗利bp", "t", "上位3週を除く平均bp", "上位3週の件数比", "呼値比率bp", "貸株料bp", "コスト20bp固定_bp", "呼値1枚+貸株料_bp", "呼値1枚+貸株料_t", "呼値2枚+貸株料_bp"]].round(1).to_string(index=False))
    print("\n年別(粗利bp):")
    print(yr.pivot_table(index=["対象", "シグナル", "規模"], columns="年", values="平均bp").round(1).to_string())
