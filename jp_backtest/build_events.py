"""全銘柄・全買いシグナルの発生履歴(日付・銘柄・規模・超過リターン)を作る。stock_signals.py の入力。

超過リターン = 翌日始値で買い、h 日後の始値で売ったときの、同日・同流動性分位の全銘柄平均との差。h = 5, 10, 20。
出力: cache/stock_events.npz
"""
import os
import time

import numpy as np

import backtest as bt
from build_report import zname
from mtf_run import signals

HOLDS = [5, 10, 20]
FAMS = ["candle", "chart", "tech", "zone"]

P = bt.load_panel()
liq = bt.liquidity_bucket(P)
ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
fr = bt.forward_returns(P, HOLDS)
X = {}
for h in HOLDS:
    r = np.where(ok, fr[h], np.nan)
    X[h] = (r - bt.baseline(r, liq)).astype(np.float32)
size = np.select([liq == 0, liq <= 2, liq == 3], [-1, 0, 1], 2).astype(np.int8)   # 0=小型 1=中型 2=大型
valid = ~P["limit"] & (liq > 0)
names, fams, T, C, Z = [], [], [], [], []
XS = {h: [] for h in HOLDS}
for fam in FAMS:
    t0 = time.time()
    sigs = signals(P, fam, "D")
    for name, (d, S) in sigs.items():
        if d <= 0:
            continue
        t, c = np.nonzero(S & valid)
        k = len(names)
        names.append(zname(name)); fams.append(fam)
        T.append(t.astype(np.int16)); C.append(c.astype(np.int16)); Z.append(size[t, c])
        for h in HOLDS:
            XS[h].append(X[h][t, c])
    print(f"[{fam}] {len(names)} signals {time.time() - t0:.0f}s", flush=True)
sid = np.concatenate([np.full(len(t), i, np.int16) for i, t in enumerate(T)])
np.savez_compressed(os.path.join(bt.CACHE, "stock_events.npz"), sid=sid, t=np.concatenate(T), c=np.concatenate(C), z=np.concatenate(Z),
                    **{f"x{h}": np.concatenate(XS[h]) for h in HOLDS}, names=np.array(names), fams=np.array(fams),
                    dates=np.array(P["dates"].strftime("%Y-%m-%d")), codes=np.array(P["codes"]))
print("events", len(sid))
