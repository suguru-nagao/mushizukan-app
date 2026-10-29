"""日足のパネルを、週足・月足のパネルに集約する。

バー(週・月)の始値=最初の営業日の始値、終値=最後の営業日の終値、高値/安値=期間内の最大/最小、売買代金・出来高=合計。
シグナルはバーが終わった時点(最後の営業日の引け後)で判定し、エントリーは次のバーの始値(= 次の週・月の最初の営業日の始値)。
データの最初と最後の、期間が途中のバーは使わない。
"""
import numpy as np
import pandas as pd

LIQ_WINDOW = {"D": 20, "W": 4, "M": 1}   # 売買代金の中央値をとる本数(約20営業日)
HORIZONS = {"D": [0, 1, 2, 3, 5, 10, 20], "W": [0, 1, 2, 4, 8, 13], "M": [0, 1, 2, 3, 6]}
GH = {"D": 5, "W": 4, "M": 3}            # 「保有を延ばした」変種の本数
KS = {"D": (1, 5), "W": (1, 4), "M": (1, 3)}   # 騰落率を揃える期間(本数)
UNIT = {"D": "日", "W": "週", "M": "か月"}


def resample(P, U, freq):
    dates = pd.DatetimeIndex(P["dates"])
    per = dates.to_period(freq)
    ids = per.asi8
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:] - 1, len(dates) - 1]
    keep = np.ones(len(starts), bool)
    # 最初・最後の、期間が途中のバーを除く(平日が欠けていれば途中)
    p0, p1 = per[starts[0]], per[ends[-1]]
    if np.busday_count(p0.start_time.date(), dates[starts[0]].date()) > 0:
        keep[0] = False
    if np.busday_count((dates[ends[-1]] + pd.Timedelta(days=1)).date(), (p1.end_time + pd.Timedelta(days=1)).date()) > 0:
        keep[-1] = False
    gid = np.repeat(np.arange(len(starts)), ends - starts + 1)

    def agg(a, how):
        df = pd.DataFrame(a)
        g = df.groupby(gid)
        out = {"first": g.first, "last": g.last, "max": g.max, "min": g.min, "sum": lambda: g.sum(min_count=1)}[how]()
        return out.to_numpy()[keep]

    Q = {
        "dates": pd.DatetimeIndex(dates[ends][keep]), "codes": P["codes"], "mkt": P["mkt"], "master": P["master"],
        "O": agg(P["O"], "first"), "H": agg(P["H"], "max"), "L": agg(P["L"], "min"), "C": agg(P["C"], "last"),
        "Va": agg(P["Va"], "sum"), "Vo": agg(P["Vo"], "sum"), "Craw": agg(P["Craw"], "last"),
        "limit": agg(P["limit"].astype(np.int8), "max").astype(bool),   # 期間内にストップ高/安の日があれば除外(保守的)
        "liq_window": LIQ_WINDOW[freq],
    }
    Uq = U[ends][keep] if U is not None else None
    return Q, Uq
