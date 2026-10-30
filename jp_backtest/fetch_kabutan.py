"""株探の信用残(直近5週)を取得して cache/kabutan/{code4}.json に保存する(再開可能)。
サイトへの負担を避けるため 3.2 秒以上あける。使い方: python3 fetch_kabutan.py [銘柄数]"""
import json, os, random, re, subprocess, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "kabutan"); os.makedirs(OUT, exist_ok=True)


def parse(html):
    i = html.find("信用取引&nbsp;(単位")
    if i < 0:
        return None
    seg = html[i:i + 3000]
    rows = re.findall(r'datetime="([\d-]+)">[^<]*</time></th>\s*<td>([\d,\.]+)</td>\s*<td>([\d,\.]+)</td>\s*<td>([\d,\.\-]+|-)</td>', seg)
    return [[d, float(s.replace(",", "")), float(b.replace(",", "")), (float(r.replace(",", "")) if r != "-" else None)] for d, s, b, r in rows]


def main(n):
    ev = pd.read_pickle(os.path.join(HERE, "cache", "zone_char3.pkl"))
    codes = sorted(ev["code"].astype(str).unique()); random.Random(0).shuffle(codes)
    for c5 in codes[:n]:
        c4 = c5[:4]; p = os.path.join(OUT, c4 + ".json")
        if os.path.exists(p):
            continue
        t0 = time.time()
        r = subprocess.run(["curl", "-sL", "-m", "25", "-A", "Mozilla/5.0", f"https://kabutan.jp/stock/?code={c4}"], capture_output=True)
        rows = parse(r.stdout.decode("utf-8", "ignore")) if r.returncode == 0 else None
        if rows is not None:
            json.dump(rows, open(p, "w"))
        time.sleep(max(0, 3.2 - (time.time() - t0)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
