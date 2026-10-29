"""仲間の作り方(業種・相関・組み合わせ)と、仲間の数・規模の分け方を変えた比較。stock_pool.py の続き。

仲間の作り方:
  size1/size2/size3 : 規模で分けない / 2分割(大型とそれ以外) / 3分割(小型・中型・大型)した、同じシグナルの全銘柄平均
  sec     : 同じ業種(東証33業種)の銘柄
  secz    : 同じ業種 × 同じ規模(3分割)
  corrK   : 値動きの相関が高い上位K銘柄(K=15,30,60,100,200)
  corrzK  : 相関が高い、同じ規模の上位K銘柄
  seccK   : 同じ業種の中で、相関が高い上位K銘柄
  seczK   : 同じ業種 × 同じ規模の中で、相関が高い上位K銘柄
自身は、仲間の合計から引いてある(自分のデータで自分を推定しない)。
使い方: python3 stock_pool2.py [保有週=4] [学習割合=0.6]
"""
import gzip
import json
import sys

import numpy as np

import backtest as bt
import stock_pool as sp

NS, NC, NT = sp.NS, sp.NC, sp.NT
_m = {r["Code"]: r for r in json.load(gzip.open(bt.CACHE + "/master.json.gz", "rt", encoding="utf-8"))}
SEC = np.array([_m[c]["S33"] if c in _m else "" for c in sp.CODES])
SECID = np.unique(SEC[SEC != ""], return_inverse=True)
sec_idx = np.full(NC, -1)
sec_idx[SEC != ""] = SECID[1]


def corr_matrix(t_split, window=None, t_start=None):
    """peers() と同じ手順の相関行列(残差の相関)。(corr, zmode, keep)"""
    a = 1 if window is None else max(1, t_split - window)
    if t_start is not None:
        a = t_start
    r = sp.RET[a:t_split].astype(np.float64); sz = sp.SIZE[a:t_split]
    minobs = 150 if window is None else min(150, int((t_split - a) * 0.7))
    minov = 100 if window is None else min(100, int((t_split - a) * 0.5))
    res = np.full_like(r, np.nan)
    for z in range(3):
        m = (sz == z) & ~np.isnan(r)
        s = np.where(m, r, 0).sum(1); c = m.sum(1)
        res = np.where(m, r - np.where(c > 0, s / np.maximum(c, 1), np.nan)[:, None], res)
    res = np.clip(res, -0.3, 0.3)
    ok = ~np.isnan(res); cnt = ok.sum(0); keep = cnt >= minobs
    x = np.where(ok, res, 0.0); sd = np.sqrt((x ** 2).sum(0) / np.maximum(cnt, 1))
    x = np.clip(x / np.where(sd > 0, sd, 1), -4, 4)
    corr = (x.T @ x) / np.maximum(ok.T.astype(np.float32) @ ok.astype(np.float32), 1)
    ov = ok.T.astype(np.float32) @ ok.astype(np.float32)
    corr[ov < minov] = -9; corr[~keep, :] = -9; corr[:, ~keep] = -9
    np.fill_diagonal(corr, -9)
    zmode = np.array([np.bincount(sz[:, j][sz[:, j] >= 0].astype(int), minlength=3).argmax() if (sz[:, j] >= 0).any() else 1 for j in range(NC)])
    return corr, zmode, keep


def group_pool(n, sm, gid, valid):
    """同じグループ(gid が同じ)の銘柄の合計から、自身を引いた平均。gid<0 は対象外。"""
    out = np.full(n.shape, np.nan)
    g = np.where(valid, gid, -1)
    for k in np.unique(g[g >= 0]):
        cols = np.flatnonzero(g == k)
        tn = n[:, cols].sum(1, keepdims=True); ts = sm[:, cols].sum(1, keepdims=True)
        pn = tn - n[:, cols]; ps = ts - sm[:, cols]
        out[:, cols] = np.where(pn >= 10, ps / np.maximum(pn, 1), np.nan)
    return out


def topk_pool(n, sm, corr, mask, K):
    """mask(銘柄×銘柄の bool)で許された相手のうち、相関が高い上位K銘柄の平均。"""
    c = np.where(mask, corr, -9)
    nn = np.argsort(-c, axis=1)[:, :K]
    valid = np.take_along_axis(c, nn, 1) > -9
    pn = (n[:, nn] * valid[None]).sum(2); ps = (sm[:, nn] * valid[None]).sum(2)
    return np.where(pn >= 10, ps / np.maximum(pn, 1), np.nan)


def methods(E, t_split, Ks=(15, 30, 60, 100, 200), window=None):
    corr, zmode, keep = corr_matrix(t_split, window)
    tr = E["t"] < t_split
    n, sm = sp.stock_mats(E, tr)
    est = {"own": np.where(n > 0, sm / np.maximum(n, 1), np.nan)}
    ok = tr & ~np.isnan(E["x"])
    z3 = E["z"].astype(int); z2 = (z3 == 2).astype(int)
    for name, zz, ng in (("size1", np.zeros_like(z3), 1), ("size2", z2, 2), ("size3", z3, 3)):
        gs = np.zeros((NS, ng)); gn = np.zeros((NS, ng))
        np.add.at(gs, (E["sid"][ok], zz[ok]), E["x"][ok]); np.add.at(gn, (E["sid"][ok], zz[ok]), 1)
        zm = np.minimum(zmode, ng - 1) if ng < 3 else zmode
        if ng == 2:
            zm = (zmode == 2).astype(int)
        est[name] = (gs / np.maximum(gn, 1))[:, zm]
    secv = sec_idx >= 0
    est["sec"] = group_pool(n, sm, sec_idx, secv & keep)
    est["secz"] = group_pool(n, sm, sec_idx * 3 + zmode, secv & keep)
    same_z = zmode[:, None] == zmode[None, :]
    same_s = (sec_idx[:, None] == sec_idx[None, :]) & (sec_idx[:, None] >= 0)
    for K in Ks:
        est[f"corr{K}"] = topk_pool(n, sm, corr, np.ones_like(same_z), K)
        est[f"corrz{K}"] = topk_pool(n, sm, corr, same_z, K)
    for K in (15, 30):
        est[f"secc{K}"] = topk_pool(n, sm, corr, same_s, K)
        est[f"secz{K}"] = topk_pool(n, sm, corr, same_s & same_z, K)
    return est, n, keep


def evaluate(h, frac, boot=0, window=None):
    E = sp.prep(h); ts = int(NT * frac)
    est, n, keep = methods(E, ts, window=window)
    te = (E["t"] >= ts) & ~np.isnan(E["x"])
    nt = np.zeros((NS, NC)); st = np.zeros((NS, NC))
    np.add.at(nt, (E["sid"][te], E["c"][te]), 1); np.add.at(st, (E["sid"][te], E["c"][te]), E["x"][te])
    mt = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
    use = (n >= 3) & (nt >= 3) & keep[None, :]
    out = {}
    cols = np.flatnonzero(use.any(0))
    rng = np.random.default_rng(0)
    for m, v in est.items():
        ok = use & ~np.isnan(v)
        c = float(np.corrcoef(v[ok], mt[ok])[0, 1])
        cs = [np.corrcoef(v[s][ok[s]], mt[s][ok[s]])[0, 1] for s in range(NS) if ok[s].sum() > 100]
        row = {"pairs": int(ok.sum()), "corr": c, "within": float(np.nanmean(cs))}
        if boot:
            bs = []
            for _ in range(boot):
                idx = rng.choice(cols, len(cols))
                vv = v[:, idx]; uu = ok[:, idx]
                bs.append(np.corrcoef(vv[uu], mt[:, idx][uu])[0, 1])
            row["lo"], row["hi"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
        out[m] = row
    return out


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    for m, r in evaluate(h, frac, boot=100).items():
        print(f"{m:10s} pairs={r['pairs']:6d} corr={r['corr']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] within={r['within']:+.4f}")
