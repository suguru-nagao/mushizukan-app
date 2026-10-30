"""同じ銘柄・同じ週に、推定がコストを超えるシグナルが何個点灯しているかで、実績が変わるか(合意の数)。
使い方: [SELL=1] python3 display_check2.py [buy|sell] [h=4]
"""
import sys

import numpy as np
import pandas as pd

import backtest as bt
import display_check as dc

side = sys.argv[1] if len(sys.argv) > 1 else "buy"
h = int(sys.argv[2]) if len(sys.argv) > 2 else 4
# 制度信用(楽天証券)の実コスト(bp): 売買のスプレッド・約定のズレ 20bp + 金利/貸株料(保有の暦日で按分)
# 買い: 買方金利 年2.80%(配当調整金の受取は見込まない=保守的)。売り: 貸株料 年1.10% + 配当調整金の支払い(配当利回り年2.0%と仮定)。逆日歩は含まない。
LONG_RATE, SHORT_FEE, DIV_YIELD, BASE = 0.0280, 0.0110, 0.020, 20.0
days = 7 * h
cost = BASE + (LONG_RATE if side == "buy" else SHORT_FEE + DIV_YIELD) * days / 365 * 1e4
rows = []
for tag, fr in (("選択60%", 0.6), ("選択50%", 0.5), ("未使用80%", 0.8), ("未使用90%", 0.9)):
    sid, c, t, z, x, est = dc.prepare(h, fr)
    v = est["peer"]
    key = c.astype(np.int64) * 5000 + t
    for lab, m in (("コスト超(推定>コスト)", v > cost),):
        k = pd.Series(np.ones(m.sum()), index=key[m]).groupby(level=0).sum()
        # イベントごとに、同じ銘柄・週の「コスト超」シグナルの数
        cnt = k.reindex(key).to_numpy()
        cnt = np.where(m, cnt, 0)
        for name, sel in (("1個", (cnt == 1)), ("2個", (cnt == 2)), ("3個以上", (cnt >= 3))):
            s = m & sel
            if s.sum() > 100:
                rows.append(dict(set=tag, 個数=name, n=int(s.sum()), 実績bp=float(x[s].mean()), t=float(bt.cluster_t(x[s] / 1e4, t[s], h))))
        # 推定がコスト超のシグナルが1つ以上ある銘柄・週の、平均の推定(複合)で上位20%
        g = pd.DataFrame({"k": key[m], "v": v[m], "x": x[m]}).groupby("k").mean()
        tt = (g.index.to_numpy() % 5000)
        q = np.quantile(g.v, 0.8)
        top = g.v >= q
        rows.append(dict(set=tag, 個数="複合(平均推定)の上位20%", n=int(top.sum()), 実績bp=float(g.x[top].mean()), t=float(bt.cluster_t(g.x[top].to_numpy() / 1e4, tt[top.to_numpy()], h))))
        rows.append(dict(set=tag, 個数="複合の下位80%(コスト超のみ)", n=int((~top).sum()), 実績bp=float(g.x[~top].mean()), t=float(bt.cluster_t(g.x[~top].to_numpy() / 1e4, tt[(~top).to_numpy()], h))))
d = pd.DataFrame(rows); d.to_csv(f"results_display2_{side}_{h}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(d.pivot_table(index="個数", columns="set", values="実績bp").round(1).to_string())
print(d.groupby("個数")[["n", "t"]].mean().round(1).to_string())
