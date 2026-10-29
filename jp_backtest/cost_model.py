"""ショートの実コストを織り込んだ再計算(貸借銘柄・全規模)。

J-Quants Light には貸株料・逆日歩・気配のデータがないため、日足から次の 3 つで見積もる。
  1) 呼値(ティック)による最小スプレッド: 売って買い戻すと、最低でも 1 ティック分の値幅を払う。
     TOPIX100 以外の呼値表(株価 3,000円以下は 1 円、5,000円以下は 5 円、30,000円以下は 10 円 …)。
     対象は小型の貸借銘柄が中心で、TOPIX100 の特例呼値は使わない。比率 = 呼値 / 未調整の終値。
  2) Corwin-Schultz 法による高値・安値からの実効スプレッド推定(直近 20 日平均)。ノイズが大きいので、1) との大きい方を使うシナリオも見る。
  3) 貸株料: 年 1.15%(制度信用の目安)を、保有する暦日数で按分。逆日歩は予測できないので含めない。
評価は margin_check.py と同じ(貸借のみ・当日ストップ高/安を除く・方向はショート)。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
from margin_check import SIZES, signals

HERE = os.path.dirname(os.path.abspath(__file__))
LEND_FEE = 0.0115  # 年率
NAMES = ["[探索]終値が高値圏(レンジ3%超)", "大陽線|出来高急増", "赤三兵|出来高急増", "窓下げ2%超"]


def tick_size(p):
    """TOPIX100 以外の呼値(円)。"""
    edges = [3_000, 5_000, 30_000, 50_000, 300_000, 500_000, 3_000_000, 5_000_000, 30_000_000, 50_000_000]
    vals = [1, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    return np.select([p <= e for e in edges], vals[:-1], default=vals[-1])


def corwin_schultz(H, L, window=20):
    """日次の CS スプレッド(比率)を直近 window 日平均。負は 0 に切り上げ。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.log(H / L) ** 2
        pH, pL = np.vstack([np.full((1, H.shape[1]), np.nan), H[:-1]]), np.vstack([np.full((1, L.shape[1]), np.nan), L[:-1]])
        beta = hl + np.vstack([np.full((1, H.shape[1]), np.nan), hl[:-1]])
        gamma = np.log(np.maximum(H, pH) / np.minimum(L, pL)) ** 2
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.where(np.isfinite(s), np.maximum(s, 0), np.nan)
    return pd.DataFrame(s).rolling(window, min_periods=window // 2).mean().to_numpy()


def stats(v, t, h):
    s = bt.summarize(v, t, h)
    return s["mean_bp"], s["t"]


if __name__ == "__main__":
    P = bt.load_panel()
    U = bt.load_margin(P)
    sigs = signals(P)
    liq = bt.liquidity_bucket(P, universe=U)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)

    tick_rel = tick_size(P["Craw"]) / P["Craw"]
    cs = corwin_schultz(P["H"], P["L"])
    dates = P["dates"].to_numpy().astype("datetime64[D]")

    rows = []
    for h, vname in [(1, "1日保有"), (5, "5日保有")]:
        r = np.where(ok, bt.forward_returns(P, [h])[h], np.nan)
        x = r - bt.baseline(r, liq)
        T = len(dates)
        t_idx = np.arange(T)
        # 保有の暦日数(翌日始値 → h 日後の始値)
        entry = np.minimum(t_idx + 1, T - 1)
        exit_ = np.minimum(t_idx + 1 + h, T - 1)
        cal = (dates[exit_] - dates[entry]).astype(int)
        for name in NAMES:
            d, S = sigs[name]
            for size, lo, hi in SIZES:
                sel = S & ~P["limit"] & (liq >= lo) & (liq <= hi) & ~np.isnan(x) & ~np.isnan(tick_rel)
                t, c = np.nonzero(sel)
                if len(t) < 50:
                    continue
                gross = -x[t, c]
                tk = tick_rel[t, c]
                cs_ = np.nan_to_num(cs[t, c], nan=0.0)
                fee = LEND_FEE * cal[t] / 365
                mx = np.maximum(tk, cs_)
                row = dict(シグナル=name, 規模=size, 保有=vname, n=len(t),
                           粗利bp=gross.mean() * 1e4, 呼値比率bp=tk.mean() * 1e4, CS推定bp=cs_.mean() * 1e4, 貸株料bp=fee.mean() * 1e4)
                for label, cost in [("コスト20bp固定", np.full(len(t), 0.0020)), ("呼値1枚+貸株料", tk + fee),
                                    ("max(呼値,CS)+貸株料", mx + fee), ("呼値2枚+貸株料", 2 * tk + fee)]:
                    m, tt = stats(gross - cost, t, h)
                    row[f"{label}_bp"], row[f"{label}_t"] = m, tt
                rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results_cost_model.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    cols = ["シグナル", "規模", "保有", "n", "粗利bp", "呼値比率bp", "CS推定bp", "貸株料bp", "コスト20bp固定_bp", "呼値1枚+貸株料_bp", "呼値1枚+貸株料_t", "max(呼値,CS)+貸株料_bp", "呼値2枚+貸株料_bp"]
    print(res[cols].round(1).to_string(index=False))
