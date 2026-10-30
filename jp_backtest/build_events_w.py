"""週足(長期、Yahoo Finance 2000年〜)の全買いシグナルの発生履歴を作る。stock_pool.py の入力。
超過リターン = 翌週の始値で買い、h 週後の始値で売ったときの、同週・同流動性分位の全銘柄平均との差。h = 1, 4, 12 週。
出力: cache/stock_events_w.npz
"""
import os
import time

import numpy as np

import backtest as bt
from build_report import zname
from long_run import load
from mtf_run import signals

HOLDS = [int(v) for v in os.environ.get("HOLDS", "1,4,12").split(",")]
SUFFIX = os.environ.get("SUFFIX", "")
SELL = os.environ.get("SELL") == "1"   # 売りシグナル: 超過リターンの符号を反転(空売りの損益)
P, U = load("W")
liq = bt.liquidity_bucket(P)
ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
fr = bt.forward_returns(P, HOLDS)
X = {}
for h in HOLDS:
    r = np.where(ok, fr[h], np.nan)
    X[h] = (r - bt.baseline(r, liq)).astype(np.float32)
size = np.select([liq == 0, liq <= 2, liq == 3], [-1, 0, 1], 2).astype(np.int8)
valid = ~P["limit"] & (liq > 0)
names, fams, T, C, Z = [], [], [], [], []
XS = {h: [] for h in HOLDS}
for fam in ["candle", "chart", "tech", "zone"]:
    t0 = time.time()
    for name, (d, S) in signals(P, fam, "W").items():
        if (d < 0) != SELL:
            continue
        t, c = np.nonzero(S & valid)
        names.append(zname(name)); fams.append(fam)
        T.append(t.astype(np.int16)); C.append(c.astype(np.int16)); Z.append(size[t, c])
        for h in HOLDS:
            XS[h].append(-X[h][t, c] if SELL else X[h][t, c])
    print(f"[{fam}] {len(names)} signals {time.time() - t0:.0f}s", flush=True)
sid = np.concatenate([np.full(len(t), i, np.int16) for i, t in enumerate(T)])
# 銘柄間の相関用: 週次リターン(終値ベース)から、市場平均と、同じ分位の平均を引いた残差
C_ = P["C"]; ret = np.full_like(C_, np.nan); ret[1:] = C_[1:] / C_[:-1] - 1
np.savez_compressed(os.path.join(bt.CACHE, ("stock_events_w_sell" if SELL else "stock_events_w") + SUFFIX + ".npz"), sid=sid, t=np.concatenate(T), c=np.concatenate(C), z=np.concatenate(Z),
                    **{f"x{h}": np.concatenate(XS[h]) for h in HOLDS}, names=np.array(names), fams=np.array(fams),
                    dates=np.array(P["dates"].strftime("%Y-%m-%d")), codes=np.array(P["codes"]), ret=ret.astype(np.float32), size=size)
print("events", len(sid))
