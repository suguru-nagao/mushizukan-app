"""Yahoo Finance の長期日足から、(日付 × 銘柄)のパネルを作る。J-Quants と重なる期間で突き合わせ、週足・月足に集約して保存する。

クリーニング: 欠損・出来高 0 の足は欠損扱い、高値/安値を始値・終値で補正、日付は上場銘柄の数が少ない日(祝日の誤記など)を除く。
近似: 売買代金 = 終値 × 出来高(分割調整済み同士なので実際の売買代金に近い)。ストップ高/安 = 「高値=安値で、前日比 ±5% 以上」の足。
貸借の区分は、過去分が取れないので、J-Quants の月初マスターのうち最も新しい区分で代用する(先読みを含む)。全銘柄の結果を主とする。
"""
import gzip
import json
import os

import numpy as np
import pandas as pd

import timeframe as tfm
from fetch_yahoo import CACHE, OUT, candidate_codes

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(CACHE, "long")
MIN_TICKERS_PER_DAY = 200


def read_all():
    codes = candidate_codes()
    series = {}
    for c in codes:
        p = os.path.join(OUT, f"{c}.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt"))
        if len(d["t"]) < 60:
            continue
        idx = (pd.to_datetime(d["t"], unit="s") + pd.Timedelta(hours=9)).normalize()
        df = pd.DataFrame({k: pd.to_numeric(pd.Series(d[k], dtype="float64")) for k in ("o", "h", "l", "c", "v")})
        df.index = idx
        df = df[~df.index.duplicated(keep="last")]
        series[c] = df
    return series


def build_daily(series):
    dates = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series.values()])))
    cnt = pd.Series(0, index=dates)
    for s in series.values():
        cnt.loc[s.index] += 1
    dates = cnt.index[cnt >= MIN_TICKERS_PER_DAY]
    codes = sorted(series)
    T, N = len(dates), len(codes)
    arr = {k: np.full((T, N), np.nan, dtype=np.float32) for k in ("o", "h", "l", "c", "v")}
    for j, c in enumerate(codes):
        s = series[c].reindex(dates)
        for k in arr:
            arr[k][:, j] = s[k].to_numpy(dtype=np.float32)
    # クリーニング
    bad = ~(arr["v"] > 0) | ~(arr["c"] > 0) | ~(arr["o"] > 0) | ~(arr["h"] > 0) | ~(arr["l"] > 0)
    for k in arr:
        arr[k][bad] = np.nan
    arr["h"] = np.fmax(np.fmax(arr["h"], arr["o"]), arr["c"])
    arr["l"] = np.fmin(np.fmin(arr["l"], arr["o"]), arr["c"])
    prev = np.vstack([np.full((1, N), np.nan, dtype=np.float32), arr["c"][:-1]])
    ret = arr["c"] / prev - 1
    limit = (arr["h"] == arr["l"]) & (np.abs(ret) >= 0.05)
    return dates, codes, arr, limit


def margin_proxy(codes):
    """銘柄ごとの貸借区分(最新の月初マスターの Mrgn=='2')。見つからなければ False。"""
    import glob
    mp = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "master_hist", "*.json.gz"))):
        for r in json.load(gzip.open(p, "rt", encoding="utf-8"))["rows"]:
            mp[r["Code"]] = r.get("Mrgn")
    for r in json.load(gzip.open(os.path.join(CACHE, "master.json.gz"), "rt", encoding="utf-8")):
        mp[r["Code"]] = r.get("Mrgn")
    return np.array([mp.get(c) == "2" for c in codes])


def to_panel(dates, codes, arr, limit):
    T, N = len(dates), len(codes)
    c = arr["c"].astype(np.float64)
    return {"dates": dates, "codes": pd.Index(codes), "mkt": np.array(["0000"] * N), "master": None,
            "O": arr["o"].astype(np.float64), "H": arr["h"].astype(np.float64), "L": arr["l"].astype(np.float64), "C": c,
            "Va": (arr["c"] * arr["v"]).astype(np.float64), "Vo": arr["v"].astype(np.float64), "Craw": c, "limit": limit}


def validate(dates, codes, arr):
    """J-Quants(2021-11〜)と同じ日・同じ銘柄の終値・出来高を比べる。"""
    import backtest as bt
    P = bt.load_panel()
    jd = pd.DatetimeIndex(P["dates"])
    common_d = dates.intersection(jd)
    jc = {c: i for i, c in enumerate(P["codes"])}
    yi = {c: i for i, c in enumerate(codes)}
    cs = sorted(set(jc) & set(yi))
    di_y, di_j = dates.get_indexer(common_d), jd.get_indexer(common_d)
    cy = arr["c"][np.ix_(di_y, [yi[c] for c in cs])].astype(np.float64)
    cj = P["C"][np.ix_(di_j, [jc[c] for c in cs])]
    vy = arr["v"][np.ix_(di_y, [yi[c] for c in cs])].astype(np.float64)
    vj = P["Vo"][np.ix_(di_j, [jc[c] for c in cs])]
    m = ~np.isnan(cy) & ~np.isnan(cj)
    rel = np.abs(cy[m] / cj[m] - 1)
    mv = m & ~np.isnan(vy) & ~np.isnan(vj) & (vj > 0)
    vr = vy[mv] / vj[mv]
    return {"共通銘柄数": len(cs), "共通の日数": len(common_d), "比べた足の数": int(m.sum()),
            "終値の差 中央値": float(np.median(rel)), "終値の差0.5%以内": float((rel <= 0.005).mean()), "終値の差5%超": float((rel > 0.05).mean()),
            "出来高比(Yahoo/J-Quants) 中央値": float(np.median(vr)), "出来高比 0.8〜1.25": float(((vr > 0.8) & (vr < 1.25)).mean())}


if __name__ == "__main__":
    os.makedirs(LONG, exist_ok=True)
    series = read_all()
    print("読み込んだ銘柄:", len(series), flush=True)
    dates, codes, arr, limit = build_daily(series)
    print(f"日付 {len(dates)} ({dates[0].date()}〜{dates[-1].date()}) × 銘柄 {len(codes)}", flush=True)
    print("突き合わせ:", json.dumps(validate(dates, codes, arr), ensure_ascii=False, indent=1), flush=True)
    U1 = np.broadcast_to(margin_proxy(codes), (len(dates), len(codes)))
    P = to_panel(dates, codes, arr, limit)
    for f in ("W", "M"):
        Q, Uq = tfm.resample(P, np.ascontiguousarray(U1), f)
        np.savez_compressed(os.path.join(LONG, f"{f}.npz"), dates=Q["dates"].values, codes=np.array(codes), U=Uq,
                            **{k: Q[k] for k in ("O", "H", "L", "C", "Va", "Vo", "Craw", "limit")})
        print(f, "bars", len(Q["dates"]), Q["dates"][0].date(), "→", Q["dates"][-1].date(), flush=True)
