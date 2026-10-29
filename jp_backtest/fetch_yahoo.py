"""Yahoo Finance(非公式の chart API)から、日本株の長期の日足を取得してキャッシュする。

対象: J-Quants の月初マスター(2021-10〜)と現在のマスターに載った、東証の株式(プライム/スタンダード/グロースと、旧市場区分)。
      上場廃止になった銘柄も、Yahoo に残っていれば取れる。ティッカーは「4桁コード.T」(英字入りコードも同様)。
出力: cache/yahoo/{コード}.json.gz = {"t": [日付], "o","h","l","c","v"(分割調整済み), "adj": 配当調整済み終値, "splits": [...]}
O/H/L/C/V は分割だけ調整済み(J-Quants の AdjO/AdjH/AdjL/AdjC と同じ形式)。取得済みはスキップするので、中断しても再開できる。
使い方: python3 jp_backtest/fetch_yahoo.py [--limit N] [--workers 4]
"""
import argparse
import glob
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(CACHE, "yahoo")
EQUITY_MKT = {"0101", "0102", "0104", "0106", "0107", "0111", "0112", "0113"}
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0"
        _local.s = s
    return _local.s


def candidate_codes():
    """J-Quants の月初マスターと現在のマスターから、株式の 5 桁コードの和集合を作る。"""
    codes = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "master_hist", "*.json.gz"))):
        for r in json.load(gzip.open(p, "rt", encoding="utf-8"))["rows"]:
            if r.get("Mkt") in EQUITY_MKT:
                codes[r["Code"]] = r["Mkt"]
    for r in json.load(gzip.open(os.path.join(CACHE, "master.json.gz"), "rt", encoding="utf-8")):
        if r.get("Mkt") in EQUITY_MKT:
            codes[r["Code"]] = r["Mkt"]
    return sorted(codes)


def symbol(code5):
    return code5[:4] + ".T"


def fetch_one(code5, tries=6):
    path = os.path.join(OUT, f"{code5}.json.gz")
    if os.path.exists(path):
        return code5, "cached"
    sym = symbol(code5)
    for i in range(tries):
        try:
            r = session().get(URL.format(sym=sym), params=dict(period1=0, period2=int(time.time()), interval="1d", events="div,splits", includeAdjustedClose="true"), timeout=40)
        except requests.RequestException:
            time.sleep(2 ** (i + 1)); continue
        if r.status_code == 404:
            res = "none"
            break
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** (i + 1) + 1); continue
        if r.status_code != 200:
            return code5, f"http{r.status_code}"
        j = r.json()["chart"]
        if not j.get("result"):
            res = "none"
            break
        d = j["result"][0]
        q = d["indicators"]["quote"][0]
        adj = d["indicators"].get("adjclose", [{}])[0].get("adjclose")
        out = {"t": d.get("timestamp") or [], "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"], "v": q["volume"], "adj": adj,
               "splits": list(((d.get("events") or {}).get("splits") or {}).values())}
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(tmp, path)
        return code5, f"ok{len(out['t'])}"
    else:
        return code5, "failed"
    # 取得できなかった銘柄も、再取得しないよう空で記録する
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"t": [], "o": [], "h": [], "l": [], "c": [], "v": [], "adj": None, "splits": []}, f)
    return code5, "none"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    codes = candidate_codes()
    if a.limit:
        codes = codes[:a.limit]
    print(f"対象 {len(codes)} 銘柄", flush=True)
    done = {"ok": 0, "none": 0, "cached": 0, "other": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (c, st) in enumerate(ex.map(fetch_one, codes), 1):
            key = "ok" if st.startswith("ok") else st if st in done else "other"
            done[key] += 1
            if n % 200 == 0:
                print(f"{n}/{len(codes)} {done} {time.time() - t0:.0f}s", flush=True)
    print("done", done, flush=True)
