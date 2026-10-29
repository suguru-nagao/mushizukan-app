"""上位足のトレンドの向き(上向き/下向き)だけで、その後の成績に差があるか(シグナルと無関係な基準)。

confluence.py の「一致 − 逆行」の差が、シグナル固有か、単なる勢いの持続かを分けるための基準。
全銘柄・全日(当日ストップ高/安を除く)で、翌日始値エントリーの超過リターンを、上向きと下向きで比べる。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import timeframe as tfm
from confluence import MA, SIZES, trend_state

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    P = bt.load_panel(); U = bt.load_margin(P)
    liq = bt.liquidity_bucket(P)
    ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
    rows = []
    for lab, ma in MA.items():
        Q, _ = tfm.resample(P, U, lab)
        up, valid = trend_state(P, Q, ma)
        for h in (1, 5):
            r = np.where(ok, bt.forward_returns(P, [h])[h], np.nan)
            x = r - bt.baseline(r, liq)
            for size, lo, hi in SIZES:
                grp = (liq >= lo) & (liq <= hi) & ~P["limit"] & ~np.isnan(x) & valid
                st = {}
                for key, m in (("上向き", grp & up), ("下向き", grp & ~up)):
                    t, c = np.nonzero(m)
                    v = x[t, c]; mu = v.mean()
                    s = pd.Series(v - mu).groupby(t // max(h, 5)).sum()
                    st[key] = (mu, np.sqrt((s ** 2).sum()) / len(v), len(v))
                a, b = st["上向き"], st["下向き"]
                rows.append(dict(上位足={"W": "週足", "M": "月足"}[lab], 規模=size, 保有日数=h, 上向きn=a[2], 上向きbp=a[0] * 1e4, 下向きn=b[2], 下向きbp=b[0] * 1e4,
                                 差bp=(a[0] - b[0]) * 1e4, 差のt=(a[0] - b[0]) / np.sqrt(a[1] ** 2 + b[1] ** 2)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results_confluence_base.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(res.round(1).to_string(index=False))
