"""長期(Yahoo Finance、1999年〜)の週足・月足パネルで、4つの分析を実行する。

使い方: python3 jp_backtest/long_run.py chart:W zone:M ...   (family:時間軸)
出力: results_{family}_{時間軸}L_{main|margin_main|variants|matched}.csv   (L = 長期)
貸借のみ(margin_main)は、最新のマスターの区分で過去も代用したもの(先読みを含む)。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import timeframe as tfm
from long_panel import LONG
from mtf_run import evaluate, signals


def load(f):
    z = np.load(os.path.join(LONG, f"{f}.npz"), allow_pickle=True)
    P = {"dates": pd.DatetimeIndex(z["dates"]), "codes": pd.Index(z["codes"]), "mkt": np.array(["0000"] * len(z["codes"])), "master": None,
         "liq_window": tfm.LIQ_WINDOW[f]}
    for k in ("O", "H", "L", "C", "Va", "Vo", "Craw", "limit"):
        P[k] = z[k]
    return P, z["U"]


if __name__ == "__main__":
    cache = {}
    for job in sys.argv[1:]:
        fam, tf = job.split(":")
        if tf not in cache:
            cache[tf] = load(tf)
        P, U = cache[tf]
        t0 = time.time()
        sigs = signals(P, fam, tf)
        counts = {k: int(a.sum()) for k, (d, a) in sigs.items()}
        print(f"[{fam}:{tf}L] bars={len(P['dates'])} stocks={len(P['codes'])} signals={len(sigs)} events={sum(counts.values())} detect={time.time() - t0:.0f}s", flush=True)
        evaluate(P, U, sigs, fam, tf, tag=tf + "L")
        pd.Series(counts).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{fam}_{tf}L_counts.csv"), header=["events"], encoding="utf-8-sig")
        print(f"[{fam}:{tf}L] done in {time.time() - t0:.0f}s", flush=True)
