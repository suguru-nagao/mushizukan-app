"""日足の抵抗帯・新高値/新安値のショート候補に、貸株料・スプレッドを織り込んだ純損益(cost_model.py と同じ方法)。"""
import os

import numpy as np
import pandas as pd

import backtest as bt
from cost_model import LEND_FEE, tick_size
from margin_check import SIZES
from zones import TF_PARAMS, detect_zones

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["新安値更新(長期・250本)", "新安値更新(短期・60本)", "抵抗帯に到達(跳ね返される)", "支持帯を下抜け"]

if __name__ == "__main__":
    P = bt.load_panel(); U = bt.load_margin(P)
    sigs = detect_zones(P["H"], P["L"], P["C"], P["Vo"], *TF_PARAMS["D"])
    liq = bt.liquidity_bucket(P, universe=U)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    tick_rel = tick_size(P["Craw"]) / P["Craw"]
    dates = P["dates"].to_numpy().astype("datetime64[D]"); T = len(dates)
    rows = []
    for h, hv in [(1, "1日保有"), (5, "5日保有")]:
        r = np.where(ok, bt.forward_returns(P, [h])[h], np.nan)
        x = r - bt.baseline(r, liq)
        idx = np.arange(T)
        cal = (dates[np.minimum(idx + 1 + h, T - 1)] - dates[np.minimum(idx + 1, T - 1)]).astype(int)
        for name in NAMES:
            d, S = sigs[name]
            for size, lo, hi in SIZES[:2]:
                sel = S & ~P["limit"] & (liq >= lo) & (liq <= hi) & ~np.isnan(x) & ~np.isnan(tick_rel)
                t, c = np.nonzero(sel)
                if len(t) < 50:
                    continue
                gross = -x[t, c]  # ショート
                tk, fee = tick_rel[t, c], LEND_FEE * cal[t] / 365
                row = dict(シグナル=name, サイン方向="買い" if d > 0 else "売り", 規模=size, 保有=hv, n=len(t), 粗利bp=gross.mean() * 1e4, 呼値比率bp=tk.mean() * 1e4, 貸株料bp=fee.mean() * 1e4)
                for label, cost in [("コスト20bp固定", np.full(len(t), 0.002)), ("呼値1枚+貸株料", tk + fee), ("呼値2枚+貸株料", 2 * tk + fee)]:
                    s = bt.summarize(gross - cost, t, h)
                    row[f"{label}_bp"], row[f"{label}_t"] = s["mean_bp"], s["t"]
                rows.append(row)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(HERE, "results_zone_cost.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 28)
    print(res[["シグナル", "規模", "保有", "n", "粗利bp", "呼値比率bp", "コスト20bp固定_bp", "呼値1枚+貸株料_bp", "呼値1枚+貸株料_t", "呼値2枚+貸株料_bp"]].round(1).to_string(index=False))
