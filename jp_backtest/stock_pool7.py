"""履歴の短い銘柄(現在は推定できない銘柄)の扱いの検討。
  現状: 上げ相場の週のうち60%以上を観測できる銘柄だけ推定(2000年から上場している銘柄に近い)。
  案1: 観測できる上げ相場の週が100週以上(約3.7年の履歴)あれば推定する(仲間は、重なる週の相関で決める)。
  案2: それでも足りない銘柄は、同じ業種 × 同じ規模の銘柄の平均で代用する。
評価: 後半の成績との相関を、グループ別に比べる。
"""
import numpy as np
import pandas as pd

import stock_pool as sp
import stock_pool2 as p2
import stock_pool3 as p3

NS, NC, NT = sp.NS, sp.NC, sp.NT
sec = p2.sec_idx
rows = []
for tag, S in (("選択", p3.SET), ("未使用", [(4, 0.8), (4, 0.9), (1, 0.8), (12, 0.8)])):
    for h, f in S:
        ts = int(NT * f); E = sp.prep(h)
        n, sm = sp.stock_mats(E, E["t"] < ts)
        te = (E["t"] >= ts) & ~np.isnan(E["x"])
        nt = np.zeros((NS, NC)); st = np.zeros((NS, NC))
        np.add.at(nt, (E["sid"][te], E["c"][te]), 1); np.add.at(st, (E["sid"][te], E["c"][te]), E["x"][te])
        mt = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
        zmode = p2.corr_matrix(ts)[1]
        same_z = zmode[:, None] == zmode[None, :]
        c_old, k_old = p3.corr_any(ts, ret="gap", regime="up")
        c_new, k_new = p3.corr_any(ts, ret="gap", regime="up", min_up=100)
        est_old = p2.topk_pool(n, sm, c_old, same_z, 200)
        est_new = p2.topk_pool(n, sm, c_new, same_z, 200)
        fb = p2.group_pool(n, sm, sec * 3 + zmode, sec >= 0)            # 同じ業種×同じ規模
        sz = p2.group_pool(n, sm, zmode, np.ones(NC, bool))             # 同じ規模
        grp = {"現状で推定できる銘柄": k_old, "案1で新たに推定できる銘柄": k_new & ~k_old, "案1でも足りない銘柄": ~k_new}
        for g, mask in grp.items():
            use = (nt >= 3) & mask[None, :]
            for name, v in (("相関上位200(同規模)", est_new), ("同業種×同規模", fb), ("同規模", sz)):
                ok = use & ~np.isnan(v) & (n >= 0)
                if ok.sum() > 500:
                    rows.append(dict(set=tag, h=h, frac=f, group=g, method=name, pairs=int(ok.sum()), stocks=int(mask.sum()), corr=float(np.corrcoef(v[ok], mt[ok])[0, 1])))
    print(tag, flush=True)
df = pd.DataFrame(rows); df.to_csv("results_pool_short.csv", index=False, encoding="utf-8-sig")
print(df.groupby(["group", "method", "set"])["corr"].mean().unstack().round(4).to_string())
print(df.groupby(["group"])["stocks"].mean().round(0))
