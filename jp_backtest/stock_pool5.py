"""異常値を除いたデータでの、仲間の決め方の再検討: 同じ規模に限定 × 相関の取り方 × 仲間の数。
使い方: python3 stock_pool5.py
"""
import numpy as np
import pandas as pd

import stock_pool as sp
import stock_pool2 as p2
import stock_pool3 as p3

NS, NC, NT = sp.NS, sp.NC, sp.NT
CANDS = [
    ("終値・規模平均を引く・規模問わず200(旧基準)", dict(corr=dict(), z=False, K=200)),
    ("終値・規模平均を引く・同規模200", dict(corr=dict(), z=True, K=200)),
    ("終値・市場平均だけ引く・同規模200", dict(corr=dict(how="market"), z=True, K=200)),
    ("窓・市場平均だけ引く・同規模200", dict(corr=dict(ret="gap", how="market"), z=True, K=200)),
    ("窓・規模平均を引く・同規模200", dict(corr=dict(ret="gap"), z=True, K=200)),
    ("窓×上げ相場の週・同規模200", dict(corr=dict(ret="gap", regime="up"), z=True, K=200)),
    ("終値・業種平均を引く・同規模200", dict(corr=dict(how="sector"), z=True, K=200)),
    ("終値・市場平均だけ引く・規模問わず200", dict(corr=dict(how="market"), z=False, K=200)),
    ("終値・規模平均を引く・同規模100", dict(corr=dict(), z=True, K=100)),
    ("終値・規模平均を引く・同規模400", dict(corr=dict(), z=True, K=400)),
    ("終値・市場平均だけ引く・同規模400", dict(corr=dict(how="market"), z=True, K=400)),
]
SET = [(4, 0.6), (4, 0.5), (4, 0.7), (1, 0.6), (12, 0.6)]
HOLD = [(4, 0.8), (4, 0.9), (1, 0.8), (12, 0.8)]      # 比較に使っていない期間(後半20%・10%)


def run(h, frac):
    E = sp.prep(h); ts = int(NT * frac)
    tr = E["t"] < ts
    n, sm = sp.stock_mats(E, tr)
    te = (E["t"] >= ts) & ~np.isnan(E["x"])
    nt = np.zeros((NS, NC)); st = np.zeros((NS, NC))
    np.add.at(nt, (E["sid"][te], E["c"][te]), 1); np.add.at(st, (E["sid"][te], E["c"][te]), E["x"][te])
    mt = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
    zmode = p2.corr_matrix(ts)[1]
    same_z = zmode[:, None] == zmode[None, :]
    est = {}; keeps = []
    cache = {}
    for name, c in CANDS:
        key = tuple(sorted(c["corr"].items()))
        if key not in cache:
            cache[key] = p3.corr_any(ts, **c["corr"])
        corr, keep = cache[key]; keeps.append(keep)
        est[name] = p2.topk_pool(n, sm, corr, same_z if c["z"] else np.ones((NC, NC), bool), c["K"])
    common = np.logical_and.reduce(keeps)
    use = (n >= 3) & (nt >= 3) & common[None, :]
    for v in est.values():
        use &= ~np.isnan(v)
    out = {}
    for name, v in est.items():
        out[name] = float(np.corrcoef(v[use], mt[use])[0, 1])
    own = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
    out["自身の平均"] = float(np.corrcoef(own[use], mt[use])[0, 1])
    return out, int(use.sum())


if __name__ == "__main__":
    rows = []
    for tag, S in (("選択", SET), ("未使用", HOLD)):
        for h, f in S:
            out, pairs = run(h, f)
            for m, c in out.items():
                rows.append(dict(set=tag, h=h, frac=f, method=m, corr=c, pairs=pairs))
            print(tag, h, f, pairs, flush=True)
    df = pd.DataFrame(rows); df.to_csv("results_pool_redo.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["method", "set"])["corr"].agg(["mean", "min", "max"]).round(4).unstack().to_string())
