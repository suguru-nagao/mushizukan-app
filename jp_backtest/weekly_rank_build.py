"""週足ベースの銘柄別シグナルタブ用データ(全銘柄・買いシグナル)を report_data/wrank_*.json に書く。

推定値 = 値動きの相関が高い銘柄(上位K=200、規模は問わない)の、同じシグナルの平均(stock_pool3.py の「窓×上げ相場の週」)。銘柄自身の平均は参考として併記する。
出力の形: {コード: [[シグナル番号, 件数, 自身の平均, 推定, 同規模全銘柄の平均, 仲間の件数, 仲間のt, 勝率%, 前半, 後半, 直近発生の週番号, 信頼度, 自身のt], ...]}
"""
import json
import os

import numpy as np

import stock_pool as sp
import stock_pool2 as p2
import stock_pool3 as p3

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_data")
os.makedirs(OUT, exist_ok=True)
K, SHARD, COST = 200, 400, 20.0
NT, NC, NS = sp.NT, sp.NC, sp.NS
corr, keep = p3.corr_any(NT, ret="gap", regime="up")   # 窓(前週終値→始値)の連動を、上げ相場の週だけで測る
_, zmode, _k = p2.corr_matrix(NT)
nn = np.argsort(-corr, axis=1)[:, :K]
valid = np.take_along_axis(corr, nn, 1) > -9
avgcorr = float(np.mean(np.take_along_axis(corr, nn, 1)[valid]))
peer5 = nn[:, :5]
# 直近発生(間引く前)
key = sp.SID.astype(np.int64) * NC + sp.C
last_t = np.full((NS, NC), -1, np.int16)
np.maximum.at(last_t, (sp.SID, sp.C), sp.T.astype(np.int16))
meta = {"sigs": [[str(n), str(f)] for n, f in zip(sp.NAMES, sp.FAMS)], "dates": [str(d) for d in sp.DATES], "codes": [str(c) for c in sp.CODES],
        "size": zmode.tolist(), "peers": peer5.tolist(), "keep": keep.astype(int).tolist(), "holds": [1, 4, 12], "K": K, "avgcorr": round(avgcorr, 3)}
json.dump(meta, open(os.path.join(OUT, "wrank_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))


def sums(E, mask):
    ok = mask & ~np.isnan(E["x"])
    n = np.zeros((NS, NC)); s = np.zeros((NS, NC)); q = np.zeros((NS, NC))
    np.add.at(n, (E["sid"][ok], E["c"][ok]), 1)
    np.add.at(s, (E["sid"][ok], E["c"][ok]), E["x"][ok])
    np.add.at(q, (E["sid"][ok], E["c"][ok]), E["x"][ok] ** 2)
    w = np.zeros((NS, NC))
    np.add.at(w, (E["sid"][ok], E["c"][ok]), (E["x"][ok] > 0).astype(float))
    return n, s, q, w


for h in (1, 4, 12):
    E = sp.prep(h)
    allm = np.ones(len(E["t"]), bool)
    n, s, q, w = sums(E, allm)
    n1, s1, _, _ = sums(E, E["t"] < NT // 2)
    n2, s2, _, _ = sums(E, E["t"] >= NT // 2)
    own = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - own ** 2, 0) * n / np.maximum(n - 1, 1))
    own_t = np.where((n >= 3) & (sd > 0), own / (sd / np.sqrt(np.maximum(n, 1))), np.nan)
    # 規模ごとの全銘柄平均
    gs = np.zeros((NS, 3)); gn = np.zeros((NS, 3))
    ok = ~np.isnan(E["x"])
    np.add.at(gs, (E["sid"][ok], E["z"][ok]), E["x"][ok]); np.add.at(gn, (E["sid"][ok], E["z"][ok]), 1)
    size_mean = (gs / np.maximum(gn, 1))[:, zmode]
    # 仲間(相関が高い同規模の銘柄)
    pn = (n[:, nn] * valid[None]).sum(2); ps = (s[:, nn] * valid[None]).sum(2); pq = (q[:, nn] * valid[None]).sum(2)
    pm = np.where(pn > 0, ps / np.maximum(pn, 1), np.nan)
    psd = np.sqrt(np.maximum(pq / np.maximum(pn, 1) - pm ** 2, 0))
    # 仲間のt値は、同じ週に複数銘柄が重なる分を考慮して、件数を1/3に割り引いた近似
    pt = np.where((pn >= 30) & (psd > 0), pm / (psd / np.sqrt(np.maximum(pn / 3, 1))), np.nan)
    est = np.where(np.isnan(pm), size_mean, pm)
    fa = np.where(n1 > 0, s1 / np.maximum(n1, 1), np.nan); fb = np.where(n2 > 0, s2 / np.maximum(n2, 1), np.nan)
    score = ((n >= 10).astype(int) + (own_t >= 2).astype(int) + ((fa > 0) & (fb > 0)).astype(int) + (pt >= 2).astype(int) + (est > COST).astype(int))
    rows = {}
    for c in range(NC):
        if not keep[c]:
            continue
        sel = np.flatnonzero(n[:, c] >= 3)
        rows[c] = [[int(i), int(n[i, c]), round(float(own[i, c]), 1), round(float(est[i, c]), 1), round(float(size_mean[i, c]), 1), int(pn[i, c]),
                    None if np.isnan(pt[i, c]) else round(float(pt[i, c]), 1), int(round(w[i, c] / n[i, c] * 100)),
                    None if np.isnan(fa[i, c]) else round(float(fa[i, c]), 1), None if np.isnan(fb[i, c]) else round(float(fb[i, c]), 1),
                    int(last_t[i, c]), int(score[i, c]), None if np.isnan(own_t[i, c]) else round(float(own_t[i, c]), 1)] for i in sel]
    for k in range((NC + SHARD - 1) // SHARD):
        sh = {sp.CODES[c]: rows.get(c, []) for c in range(k * SHARD, min((k + 1) * SHARD, NC))}
        json.dump(sh, open(os.path.join(OUT, f"wrank_{h}_{k}.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("h", h, sum(len(v) for v in rows.values()), flush=True)
