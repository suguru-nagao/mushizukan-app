"""日足のシグナルを、上位足(週足・月足)のトレンドの向きに合わせると成績が変わるかの確認。

上位足のトレンド: 直前に終わった週(月)の終値が、その 13 本(月は 6 本)移動平均より上なら上向き、下なら下向き。
先読みを避けるため、日足の t 日には「t を含む週(月)より前の、完結した最後の週(月)」の状態を使う。
一致 = 買いサインで上向き、または売りサインで下向き。逆行 = その反対。
分析ごとに、全シグナルをまとめた方向付き超過リターン(翌日始値エントリー、1日・5日保有)を、一致と逆行で比べる。
"""
import os
import pickle

import numpy as np
import pandas as pd

import backtest as bt
import timeframe as tfm
from patterns import detect
from tech_signals import detect_tech
from zones import TF_PARAMS, detect_zones

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5)]
MA = {"W": 13, "M": 6}


def trend_state(P, Q, ma):
    """日足の日付ごとの、直前に完結した上位足の状態(上向き=True)。(日数, 銘柄数) の bool と有効性。"""
    Cq = pd.DataFrame(Q["C"])
    up = (Cq > Cq.rolling(ma, min_periods=ma).mean()).to_numpy()
    valid = Cq.rolling(ma, min_periods=ma).mean().notna().to_numpy()
    qd = Q["dates"]
    # 日足 t に対して、t の日付より前に終わったバーのうち最後のもの
    pos = np.searchsorted(qd.values, P["dates"].values, side="left") - 1   # qd[pos] < dates[t]
    ok = pos >= 0
    pos = np.clip(pos, 0, len(qd) - 1)
    return up[pos] & ok[:, None], valid[pos] & ok[:, None]


def pooled(P, U, sigs_by_fam, trend, label):
    liq = bt.liquidity_bucket(P)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    rows = []
    for h in (1, 5):
        r = np.where(ok, bt.forward_returns(P, [h])[h], np.nan)
        x = r - bt.baseline(r, liq)
        for fam, sigs in sigs_by_fam.items():
            for size, lo, hi in SIZES:
                grp = (liq >= lo) & (liq <= hi) & ~P["limit"] & ~np.isnan(x)
                up, valid = trend
                T_all, V_all, A_all = [], [], []
                for name, (d, S) in sigs.items():
                    sel = S & grp & valid
                    t, c = np.nonzero(sel)
                    if len(t) == 0:
                        continue
                    T_all.append(t); V_all.append(d * x[t, c]); A_all.append(up[t, c] == (d > 0))
                if not T_all:
                    continue
                t, v, a = np.concatenate(T_all), np.concatenate(V_all), np.concatenate(A_all)
                blk = max(h, 5)

                def grpstat(m):
                    vv, tt = v[m], t[m]
                    mu = vv.mean(); s = pd.Series(vv - mu).groupby(tt // blk).sum()
                    return mu, np.sqrt((s ** 2).sum()) / len(vv), len(vv)
                ma_, sa, na = grpstat(a); mb, sb, nb = grpstat(~a)
                rows.append(dict(分析=fam, 上位足=label, 規模=size, 保有日数=h, 一致n=na, 一致bp=ma_ * 1e4, 逆行n=nb, 逆行bp=mb * 1e4,
                                 差bp=(ma_ - mb) * 1e4, 差のt=(ma_ - mb) / np.sqrt(sa ** 2 + sb ** 2)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = bt.load_panel(); U = bt.load_margin(P)
    sigs = {
        "ローソク足": detect(P["O"], P["H"], P["L"], P["C"]),
        "チャートパターン": pickle.load(open(os.path.join(bt.CACHE, "chart_sig.pkl"), "rb")),
        "テクニカル指標": pickle.load(open(os.path.join(bt.CACHE, "tech_sig.pkl"), "rb")),
        "抵抗帯": detect_zones(P["H"], P["L"], P["C"], P["Vo"], *TF_PARAMS["D"]),
    }
    out = []
    for lab, ma in MA.items():
        Q, _ = tfm.resample(P, U, lab)
        out.append(pooled(P, U, sigs, trend_state(P, Q, ma), {"W": "週足", "M": "月足"}[lab]))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(os.path.join(HERE, "results_confluence.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.round(1).to_string(index=False))
