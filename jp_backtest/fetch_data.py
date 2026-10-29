"""J-Quants V2 から全銘柄の日足と銘柄マスターを取得し、日付ごとにキャッシュする。

認証: 環境変数 JQUANTS_API_KEY (ヘッダー x-api-key)
使い方: python3 jp_backtest/fetch_data.py [--from 2021-10-29] [--to YYYY-MM-DD]
キャッシュ済みの日付はスキップするので、中断しても再開できる。
"""
import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://api.jquants.com/v2"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BARS_DIR = os.path.join(CACHE, "bars")
LIGHT_START = "2021-10-29"  # Light プランで遡れる最古日


def api_get(path, params=None, retries=6):
    key = os.environ.get("JQUANTS_API_KEY")
    if not key:
        sys.exit("JQUANTS_API_KEY が未設定です")
    qs = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
    for i in range(retries):
        req = urllib.request.Request(url, headers={"x-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(2 ** (i + 1))
                continue
            raise RuntimeError(f"HTTP {e.code} {url}: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, OSError):
            if i < retries - 1:
                time.sleep(2 ** (i + 1))
                continue
            raise


def weekdays(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def fetch_one(d):
    path = os.path.join(BARS_DIR, f"{d.isoformat()}.json.gz")
    if os.path.exists(path):
        return d, None
    data = api_get("/equities/bars/daily", {"date": d.isoformat()}).get("data", [])
    # 休場日は空配列。空でもキャッシュして再取得を避ける
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    return d, len(data)


def fetch_bars(start, end, workers=3):
    os.makedirs(BARS_DIR, exist_ok=True)
    days = list(weekdays(start, end))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, (d, rows) in enumerate(ex.map(fetch_one, days), 1):
            if n % 50 == 0:
                print(f"{n}/{len(days)} {d} rows={rows}", flush=True)


def fetch_master():
    data = api_get("/equities/master").get("data", [])
    with gzip.open(os.path.join(CACHE, "master.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"master: {len(data)} rows")


def fetch_master_history(start, end):
    """月初ごとの銘柄マスター(信用区分 Mrgn: 1=信用, 2=貸借, 3=その他)を取得する。

    貸借銘柄かどうかは時期で変わるので、各日付で「その時点の指定」を使うために必要。
    休場日は空で返るため、最大 6 日先までずらして最初に取れた日を使う。
    """
    out_dir = os.path.join(CACHE, "master_hist")
    os.makedirs(out_dir, exist_ok=True)
    d = start.replace(day=1)
    while d <= end:
        path = os.path.join(out_dir, f"{d.strftime('%Y-%m')}.json.gz")
        if not os.path.exists(path):
            for k in range(7):
                day = max(d + dt.timedelta(days=k), start)
                data = api_get("/equities/master", {"date": day.isoformat()}).get("data", [])
                if data:
                    slim = [{"Code": x["Code"], "Mrgn": x.get("Mrgn"), "Mkt": x.get("Mkt")} for x in data]
                    with gzip.open(path, "wt", encoding="utf-8") as f:
                        json.dump({"date": day.isoformat(), "rows": slim}, f, ensure_ascii=False)
                    print(f"master {day} rows={len(slim)}", flush=True)
                    break
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def fetch_topix(start, end):
    data = api_get("/indices/bars/daily/topix", {"from": start, "to": end}).get("data", [])
    with gzip.open(os.path.join(CACHE, "topix.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"topix: {len(data)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=LIGHT_START)
    ap.add_argument("--to", dest="end", default=dt.date.today().isoformat())
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    fetch_master()
    fetch_topix(a.start, a.end)
    fetch_master_history(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
    fetch_bars(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
    print("done")
