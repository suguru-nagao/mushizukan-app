"""シグナルへの反応が似ている銘柄を仲間にする方法(値動きの相関との比較・組み合わせ)。
反応ベクトル: 銘柄ごと・シグナルごとの、学習期間の超過リターン平均の、全銘柄平均からのずれ(件数で重みづけしたz値、±3でクリップ)。
推定したいシグナル s については、s を除いたシグナルだけで類似度(コサイン)を測る(自分のデータで自分を推定する循環を避ける)。
  resp : 反応が似ている上位K銘柄
  both : 値動きの相関(窓×上げ相場)の行内z値 + 反応の類似度の行内z値 が高い上位K銘柄
使い方: python3 stock_pool4.py
"""
import numpy as np
import pandas as pd

import stock_pool as sp
import stock_pool2 as p2
import stock_pool3 as p3

NS, NC, NT = sp.NS, sp.NC, sp.NT


def zrow(m, valid):
    mm = np.where(valid, m, np.nan)
    mu = np.nanmean(mm, axis=1, keepdims=True); sd = np.nanstd(mm, axis=1, keepdims=True)
    return np.where(valid, (m - mu) / np.where(sd > 0, sd, 1), -9)


def estimates(E, ts, K=200):
    corr, keep = p3.corr_any(ts, ret="gap", regime="up")
    tr = E["t"] < ts
    ok = tr & ~np.isnan(E["x"])
    n = np.zeros((NS, NC)); sm = np.zeros((NS, NC)); q = np.zeros((NS, NC))
    np.add.at(n, (E["sid"][ok], E["c"][ok]), 1); np.add.at(sm, (E["sid"][ok], E["c"][ok]), E["x"][ok]); np.add.at(q, (E["sid"][ok], E["c"][ok]), E["x"][ok] ** 2)
    mu_s = sm.sum(1) / np.maximum(n.sum(1), 1)
    var_s = q.sum(1) / np.maximum(n.sum(1), 1) - mu_s ** 2
    own = np.where(n > 0, sm / np.maximum(n, 1), 0.0)
    A = np.where(n >= 3, np.clip((own - mu_s[:, None]) / np.sqrt(np.maximum(var_s, 1e-9))[:, None] * np.sqrt(np.maximum(n, 1)), -3, 3), 0.0).astype(np.float32)  # (NS, NC)
    G = A.T @ A                      # (NC, NC)
    nrm2 = (A ** 2).sum(0)
    est = {k: np.full((NS, NC), np.nan) for k in ("corr", "resp", "both")}
    cvalid = corr > -9
    idx_all = np.arange(NC)
    for s in range(NS):
        a = A[s]
        Ns = G - np.outer(a, a)
        nr = np.sqrt(np.maximum(nrm2 - a ** 2, 1e-6))
        sim = Ns / (nr[:, None] * nr[None, :])
        sim[~keep, :] = -9; sim[:, ~keep] = -9
        np.fill_diagonal(sim, -9)
        for name, M in (("resp", sim), ("both", None), ("corr", corr)):
            if name == "both":
                zv = (sim > -9) & cvalid
                M = np.where(zv, zrow(sim, zv) + zrow(corr, zv), -9)
            nn = np.argpartition(-M, K, axis=1)[:, :K]
            v = np.take_along_axis(M, nn, 1) > -9
            pn = (n[s][nn] * v).sum(1); ps = (sm[s][nn] * v).sum(1)
            est[name][s] = np.where(pn >= 10, ps / np.maximum(pn, 1), np.nan)
    return est, n, keep


def evaluate(h, frac, K=200):
    E = sp.prep(h); ts = int(NT * frac)
    est, n, keep = estimates(E, ts, K)
    te = (E["t"] >= ts) & ~np.isnan(E["x"])
    nt = np.zeros((NS, NC)); st = np.zeros((NS, NC))
    np.add.at(nt, (E["sid"][te], E["c"][te]), 1); np.add.at(st, (E["sid"][te], E["c"][te]), E["x"][te])
    mt = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
    use = (n >= 3) & (nt >= 3) & keep[None, :]
    for v in est.values():
        use &= ~np.isnan(v)
    out = {}
    for m, v in est.items():
        c = float(np.corrcoef(v[use], mt[use])[0, 1])
        cs = [np.corrcoef(v[s][use[s]], mt[s][use[s]])[0, 1] for s in range(NS) if use[s].sum() > 100]
        out[m] = (c, float(np.nanmean(cs)))
    return out, int(use.sum())


if __name__ == "__main__":
    rows = []
    for h, f in p3.SET:
        out, pairs = evaluate(h, f)
        for m, (c, w) in out.items():
            rows.append(dict(h=h, frac=f, method=m, corr=c, within=w, pairs=pairs))
        print(h, f, pairs, {m: round(c, 4) for m, (c, w) in out.items()}, flush=True)
    df = pd.DataFrame(rows); df.to_csv("results_pool_resp.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("method")[["corr", "within"]].agg(["mean", "min", "max"]).round(4))
