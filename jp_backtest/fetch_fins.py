"""決算短信サマリー(J-Quants fins/summary)を全銘柄分取得して cache/fins/{code}.json.gz に保存する。"""
import gzip
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import threading
import time

from fetch_data import CACHE, api_get as _api_get

_lock = threading.Lock(); _last = [0.0]


def api_get(path, params=None):
    # 速度制限(429)に当たるので、全体で 1.1 秒に1回まで。429 が出たら 30 秒待って再試行
    for _ in range(20):
        with _lock:
            time.sleep(max(0.0, _last[0] + 1.1 - time.time())); _last[0] = time.time()
        try:
            return _api_get(path, params, retries=1)
        except RuntimeError as e:
            if "429" in str(e):
                time.sleep(30)
                continue
            raise
    raise RuntimeError("429が続きました")

OUT = os.path.join(CACHE, "fins")
os.makedirs(OUT, exist_ok=True)
master = json.load(gzip.open(os.path.join(CACHE, "master.json.gz"), "rt", encoding="utf-8"))
import random
codes = sorted({r["Code"] for r in master if r["Mkt"] in ("0111", "0112", "0113")})
random.Random(0).shuffle(codes)   # 途中までの分でも、偏りなく分析できるように、順番をシャッフルする


def one(c):
    p = os.path.join(OUT, f"{c}.json.gz")
    if os.path.exists(p):
        return 0
    rows, pk = [], None
    while True:
        d = api_get("/fins/summary", {"code": c, **({"pagination_key": pk} if pk else {})})
        rows += d.get("data", [])
        pk = d.get("pagination_key")
        if not pk:
            break
    with gzip.open(p + ".tmp", "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    os.replace(p + ".tmp", p)
    return len(rows)


if __name__ == "__main__":
    n = 0
    with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        for i, k in enumerate(ex.map(one, codes), 1):
            n += k
            if i % 500 == 0:
                print(i, len(codes), n, flush=True)
    print("done", n)
