"""週足ベースの銘柄別シグナルタブ用データ(全銘柄・買いシグナル)を report_data/wrank_*.json に書く。

推定値 = 値動きの相関が高い銘柄(上位K=200、同じ規模に限る)の、同じシグナルの平均(stock_pool5.py の「窓×上げ相場の週・同規模200」)。銘柄自身の平均は参考として併記する。
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
SELL = os.environ.get("SELL") == "1"
PFX = "wrank_s" if SELL else "wrank"
K, SHARD = 200, 400
# コスト(bp): 買い = 現物(ゼロコース手数料0円)でスプレッド等 20bp。売り = 楽天証券の制度信用: 20bp + 貸株料 年1.10% + 配当調整金(配当利回り年2.0%と仮定)。逆日歩は含まない。
def cost_bp(h):
    return 20.0 + ((0.0110 + 0.020) if SELL else 0.0) * 7 * h / 365 * 1e4

NT, NC, NS = sp.NT, sp.NC, sp.NS
corr, keep = p3.corr_any(NT, ret="gap", regime="up")   # 窓(前週終値→始値)の連動を、上げ相場の週だけで測る
_, zmode, _k = p2.corr_matrix(NT)
corr = np.where(zmode[:, None] == zmode[None, :], corr, -9)   # 仲間は、同じ規模の銘柄に限る
nn = np.argsort(-corr, axis=1)[:, :K]
valid = np.take_along_axis(corr, nn, 1) > -9
avgcorr = float(np.mean(np.take_along_axis(corr, nn, 1)[valid]))
peer5 = nn[:, :5]
# 直近発生(間引く前)
key = sp.SID.astype(np.int64) * NC + sp.C
last_t = np.full((NS, NC), -1, np.int16)
np.maximum.at(last_t, (sp.SID, sp.C), sp.T.astype(np.int16))
has_events = np.zeros(NC, bool); has_events[np.unique(sp.C)] = True   # 長期データに、シグナルの発生履歴がある銘柄
peer5 = np.where(keep[:, None], peer5, -1)
meta = {"sigs": [[str(n), str(f)] for n, f in zip(sp.NAMES, sp.FAMS)], "dates": [str(d) for d in sp.DATES], "codes": [str(c) for c in sp.CODES],
        "size": zmode.tolist(), "peers": peer5.tolist(), "keep": keep.astype(int).tolist(), "fb": ((~keep) & has_events).astype(int).tolist(), "holds": [1, 2, 4, 6, 12], "K": K, "avgcorr": round(avgcorr, 3)}
json.dump(meta, open(os.path.join(OUT, f"{PFX}_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))


def sums(E, mask):
    ok = mask & ~np.isnan(E["x"])
    n = np.zeros((NS, NC)); s = np.zeros((NS, NC)); q = np.zeros((NS, NC))
    np.add.at(n, (E["sid"][ok], E["c"][ok]), 1)
    np.add.at(s, (E["sid"][ok], E["c"][ok]), E["x"][ok])
    np.add.at(q, (E["sid"][ok], E["c"][ok]), E["x"][ok] ** 2)
    w = np.zeros((NS, NC))
    np.add.at(w, (E["sid"][ok], E["c"][ok]), (E["x"][ok] > 0).astype(float))
    return n, s, q, w


for h in [int(v) for v in os.environ.get("HOLDS_BUILD", "1,4,12").split(",")]:
    E = sp.prep(h)
    allm = np.ones(len(E["t"]), bool)
    n, s, q, w = sums(E, allm)
    n1, s1, _, _ = sums(E, E["t"] < NT // 2)
    n2, s2, _, _ = sums(E, E["t"] >= NT // 2)
    own = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - own ** 2, 0) * n / np.maximum(n - 1, 1))
    own_t = np.where((n >= 3) & (sd > 0), own / (sd / np.sqrt(np.maximum(n, 1))), np.nan)
    # 規模ごとの全銘柄平均
    gs = np.zeros((NS, 3)); gn = np.zeros((NS, 3)); gq = np.zeros((NS, 3))
    ok = ~np.isnan(E["x"])
    np.add.at(gs, (E["sid"][ok], E["z"][ok]), E["x"][ok]); np.add.at(gn, (E["sid"][ok], E["z"][ok]), 1); np.add.at(gq, (E["sid"][ok], E["z"][ok]), E["x"][ok] ** 2)
    gm = gs / np.maximum(gn, 1); gsd = np.sqrt(np.maximum(gq / np.maximum(gn, 1) - gm ** 2, 0)); gt = np.where(gsd > 0, gm / (gsd / np.sqrt(np.maximum(gn / 3, 1))), np.nan)
    size_mean = gm[:, zmode]; size_n = gn[:, zmode]; size_t = gt[:, zmode]
    # 仲間(相関が高い同規模の銘柄)
    pn = (n[:, nn] * valid[None]).sum(2); ps = (s[:, nn] * valid[None]).sum(2); pq = (q[:, nn] * valid[None]).sum(2)
    pm = np.where(pn > 0, ps / np.maximum(pn, 1), np.nan)
    psd = np.sqrt(np.maximum(pq / np.maximum(pn, 1) - pm ** 2, 0))
    # 仲間のt値は、同じ週に複数銘柄が重なる分を考慮して、件数を1/3に割り引いた近似
    pt = np.where((pn >= 30) & (psd > 0), pm / (psd / np.sqrt(np.maximum(pn / 3, 1))), np.nan)
    est = np.where(np.isnan(pm), size_mean, pm)
    # 仲間で推定できない銘柄(履歴が短い)は、同じ規模の全銘柄の平均で代用する
    est = np.where(keep[None, :], est, size_mean); pn = np.where(keep[None, :], pn, size_n); pt = np.where(keep[None, :], pt, size_t)
    fa = np.where(n1 > 0, s1 / np.maximum(n1, 1), np.nan); fb = np.where(n2 > 0, s2 / np.maximum(n2, 1), np.nan)
    score = ((n >= 10).astype(int) + (own_t >= 2).astype(int) + ((fa > 0) & (fb > 0)).astype(int) + (pt >= 2).astype(int) + (est > cost_bp(h)).astype(int))
    rows = {}
    for c in range(NC):
        if not (keep[c] or has_events[c]):
            continue
        sel = np.flatnonzero(n[:, c] >= 3)
        rows[c] = [[int(i), int(n[i, c]), round(float(own[i, c]), 1), round(float(est[i, c]), 1), round(float(size_mean[i, c]), 1), int(pn[i, c]),
                    None if np.isnan(pt[i, c]) else round(float(pt[i, c]), 1), int(round(w[i, c] / n[i, c] * 100)),
                    None if np.isnan(fa[i, c]) else round(float(fa[i, c]), 1), None if np.isnan(fb[i, c]) else round(float(fb[i, c]), 1),
                    int(last_t[i, c]), int(score[i, c]), None if np.isnan(own_t[i, c]) else round(float(own_t[i, c]), 1)] for i in sel]
    for k in range((NC + SHARD - 1) // SHARD):
        sh = {sp.CODES[c]: rows.get(c, []) for c in range(k * SHARD, min((k + 1) * SHARD, NC))}
        json.dump(sh, open(os.path.join(OUT, f"{PFX}_{h}_{k}.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("h", h, sum(len(v) for v in rows.values()), flush=True)
