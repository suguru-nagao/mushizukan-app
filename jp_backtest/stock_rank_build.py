"""銘柄別シグナルタブ用: 全銘柄 × 買いシグナルの、規模別・補正後のランキングデータを report_data/rank_{h}_{k}.json に書く。

出力の形: {コード: [[シグナル番号, 件数, 平均, 補正後, 同規模全銘柄の平均, 同t, 勝率%, 前半, 後半, 直近発生の日付番号, 信頼度], ...]}
meta は report_data/rank_meta.json(シグナル名・分析・日付・銘柄の規模)。シグナル一覧の保有期間と同じく、5/10/20日。
"""
import json
import os
import sys

import numpy as np

import stock_signals as s

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_data")
os.makedirs(OUT, exist_ok=True)
SHARD = 400
N_DAYS = len(s.DATES)
# 銘柄の規模: 直近2年の発生時点の規模の最頻値(なければ全期間)
E0 = s.prep(10)
recent = E0["t"] >= N_DAYS - 500
cnt = np.zeros((s.NC, 3))
np.add.at(cnt, (E0["c"][recent], E0["z"][recent]), 1)
cnt_all = np.zeros((s.NC, 3))
np.add.at(cnt_all, (E0["c"], E0["z"]), 1)
zs = np.where(cnt.sum(1) > 0, cnt.argmax(1), cnt_all.argmax(1))
# 直近の発生日(間引く前の全イベントから)
key = s.SID.astype(np.int64) * s.NC + s.C
order = np.argsort(key, kind="stable")
last_t = {}
uk, first = np.unique(key[order], return_index=True)
ends = np.r_[first[1:], len(order)]
for k, a, b in zip(uk, first, ends):
    last_t[int(k)] = int(s.T[order[a:b]].max())
meta = {"sigs": [[str(n), str(f)] for n, f in zip(s.NAMES, s.FAMS)], "dates": [str(d) for d in s.DATES], "codes": [str(c) for c in s.CODES],
        "size": zs.tolist(), "holds": [5, 10, 20]}
json.dump(meta, open(os.path.join(OUT, "rank_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
for h in (5, 10, 20):
    E = s.prep(h)
    allm = np.ones(len(E["t"]), bool)
    per = {}
    for z in (0, 1, 2):
        F = s.fit(E, allm, z)
        pt = s.pooled_t(E, z, h)
        # 銘柄ごとの詳細(件数・勝率・前半・後半・t)を、(シグナル,銘柄)単位でまとめる
        m = (E["z"] == z) & ~np.isnan(E["x"])
        k2 = E["sid"][m].astype(np.int64) * s.NC + E["c"][m]
        pos = np.searchsorted(F["keys"], k2)
        x = E["x"][m]; tt = E["t"][m]
        n = np.bincount(pos, minlength=len(F["keys"])).astype(float)
        win = np.bincount(pos, (x > 0).astype(float), len(F["keys"])) / np.maximum(n, 1)
        sd = np.sqrt(np.maximum(np.bincount(pos, x * x, len(F["keys"])) / np.maximum(n, 1) - (np.bincount(pos, x, len(F["keys"])) / np.maximum(n, 1)) ** 2, 0) * n / np.maximum(n - 1, 1))
        f_m = tt < s.SPLIT
        na = np.bincount(pos[f_m], minlength=len(F["keys"])); nb = np.bincount(pos[~f_m], minlength=len(F["keys"]))
        fa = np.bincount(pos[f_m], x[f_m], len(F["keys"])) / np.maximum(na, 1)
        fb = np.bincount(pos[~f_m], x[~f_m], len(F["keys"])) / np.maximum(nb, 1)
        tv = np.where(sd > 0, F["mean"] / (sd / np.sqrt(np.maximum(n, 1))), np.nan)
        score = ((n >= 20).astype(int) + (tv >= 2).astype(int) + ((na > 0) & (nb > 0) & (fa > 0) & (fb > 0)).astype(int) + (pt[F["sid"]] >= 2).astype(int) + (F["est"] > s.COST).astype(int))
        per[z] = (F, pt, n, win, tv, fa, fb, score, na, nb)
    # 銘柄ごとにまとめる
    rows_by_c = {}
    for z in (0, 1, 2):
        F, pt, n, win, tv, fa, fb, score, na, nb = per[z]
        for j in range(len(F["keys"])):
            c = int(F["c"][j])
            if zs[c] != z or n[j] < 5:
                continue
            i = int(F["sid"][j])
            r = [i, int(n[j]), round(float(F["mean"][j]), 1), round(float(F["est"][j]), 1), round(float(F["pooled"][j]), 1),
                 None if np.isnan(pt[i]) else round(float(pt[i]), 1), int(round(win[j] * 100)),
                 round(float(fa[j]), 1) if na[j] > 0 else None, round(float(fb[j]), 1) if nb[j] > 0 else None,
                 last_t.get(i * s.NC + c, -1), int(score[j])]
            rows_by_c.setdefault(c, []).append(r)
    for k in range((s.NC + SHARD - 1) // SHARD):
        sh = {s.CODES[c]: rows_by_c.get(c, []) for c in range(k * SHARD, min((k + 1) * SHARD, s.NC))}
        json.dump(sh, open(os.path.join(OUT, f"rank_{h}_{k}.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("h", h, "done", sum(len(v) for v in rows_by_c.values()), flush=True)
