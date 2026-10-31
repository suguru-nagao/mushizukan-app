"""出来高最多の価格帯(抵抗帯・支持帯)を、支持線・抵抗線のように、跳ね返された回数つきで出す。
各銘柄について、直近の日足で「現在値より上で最も出来高が厚い帯(抵抗帯)」「下で最も厚い帯(支持帯)」を、
抵抗帯・支持帯の検出と同じ方法(窓120本、価格帯別出来高、平均の2倍以上)で作り、その帯に、過去 LOOKBACK 本の間に何回触れ、どうなったかを数える。
  触れた = 高値が帯の下端以上かつ安値が帯の上端以下(連続する触れは1回にまとめる。間が3本以内なら同じ回)
  下から来た(直前の終値が帯の下)触れ: 触れ終わって5本後の終値が帯の下 → 「跳ね返された(抵抗として機能)」、帯の上 → 「突破された」
  上から来た(直前の終値が帯の上)触れ: 5本後の終値が帯の上 → 「跳ね返した(支持として機能)」、帯の下 → 「割られた」
  5本後がまだ来ていない、または帯の中 → 「判定中」
"""
import json, os
import numpy as np
import pandas as pd
import backtest as bt
from zones import TF_PARAMS, _clean, make_zones

LOOKBACK, AFTER, MERGE = 250, 5, 3
P = bt.load_panel(); W, R = TF_PARAMS["D"][:2]; dates, codes = P["dates"], P["codes"]


def visits(h, l, c, lo, hi):
    n = len(c); touch = (h >= lo) & (l <= hi); out = []; i = 0; start = max(1, n - LOOKBACK)
    idx = np.nonzero(touch[start:])[0] + start
    if len(idx) == 0:
        return out
    groups = [[idx[0]]]
    for j in idx[1:]:
        (groups[-1].append(j) if j - groups[-1][-1] <= MERGE else groups.append([j]))
    for g in groups:
        s, e = g[0], g[-1]; pre = c[s - 1]; side = "below" if pre < lo else "above" if pre > hi else None
        if side is None:
            continue
        k = e + AFTER; res = None
        if k < n:
            res = "rej" if (side == "below" and c[k] < lo) else "brk" if (side == "below" and c[k] > hi) else "held" if (side == "above" and c[k] > hi) else "brk" if (side == "above" and c[k] < lo) else None
        out.append((side, res, s))
    return out


def one(c):
    cl = _clean(P["H"][:, c], P["L"][:, c], P["C"][:, c], P["Vo"][:, c])
    if cl is None or len(cl[3]) < W + 10:
        return None
    first, h, l, cc, v = cl; n = len(cc); tp = (h + l + cc) / 3
    z = make_zones(tp[n - W:], v[n - W:], cc[-1])
    res = {}
    for nm, (lo, hi) in (("resistance", (z[0], z[1])), ("support", (z[2], z[3]))):
        if np.isnan(lo):
            res[nm] = None; continue
        vs = visits(h, l, cc, lo, hi)
        cnt = lambda sd, r: sum(1 for a, b, _ in vs if a == sd and b == r)
        last = [dates[first + s].strftime("%Y-%m-%d") for a, b, s in vs if b in ("rej", "held")]
        res[nm] = dict(low=round(float(lo), 1), high=round(float(hi), 1), mid=round(float((lo + hi) / 2), 1), dist_pct=round(float((lo + hi) / 2 / cc[-1] - 1) * 100, 1),
                       touches=len(vs), rejected_from_below=cnt("below", "rej"), broken_up=cnt("below", "brk"), held_from_above=cnt("above", "held"), broken_down=cnt("above", "brk"),
                       pending=sum(1 for a, b, _ in vs if b is None), last_bounce=last[-1] if last else None)
    return dict(close=round(float(cc[-1]), 1), **res)


if __name__ == "__main__":
    out = {}
    for c in range(len(codes)):
        r = one(c)
        if r:
            out[codes[c][:4]] = r
    meta = dict(asof=dates[-1].strftime("%Y-%m-%d"), lookback_bars=LOOKBACK, judge_after_bars=AFTER, window_bars=W,
                note="帯=日足の価格帯別出来高で、現在値より上(下)で最も厚い帯。touches=過去250本で帯に触れた回数。rejected_from_below=下から来て跳ね返された回数(抵抗帯)。held_from_above=上から来て跳ね返した回数(支持帯)。broken_*=突破・割れ。pending=判定中。")
    json.dump(dict(meta=meta, stocks=out), open("cache/zone_levels.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(len(out), meta["asof"])
