"""仲間を決める相関の取り方を変えた比較(上位200銘柄の推定。5設定の平均)。
変えるもの: 期間の重みづけ(指数加重)・リターンの種類(終値/日中/窓/4週/値幅)・基準の引き方(なし/市場/規模/業種)・順位相関・下げ相場だけ/上げ相場だけ。
使い方: python3 stock_pool3.py
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import stock_pool as sp
import stock_pool2 as p2
from long_panel import LONG

NS, NC, NT = sp.NS, sp.NC, sp.NT
W = np.load(os.path.join(LONG, "W.npz"), allow_pickle=True)
assert list(W["codes"]) == list(sp.CODES)
O, H, L, C = (W[k].astype(np.float64) for k in ("O", "H", "L", "C"))
with np.errstate(invalid="ignore", divide="ignore"):
    R_CC = np.full_like(C, np.nan); R_CC[1:] = C[1:] / C[:-1] - 1
    R_OC = C / O - 1
    R_GAP = np.full_like(C, np.nan); R_GAP[1:] = O[1:] / C[:-1] - 1
    R_RNG = (H - L) / C
    R_4W = np.full_like(C, np.nan); R_4W[4:] = C[4:] / C[:-4] - 1
SIZE = sp.SIZE.astype(int)
SEC = p2.sec_idx


def demean(R, a, b, how):
    r = R[a:b].copy()
    if how == "none":
        return r
    if how == "market":
        return r - np.nanmean(r, axis=1, keepdims=True)
    if how == "size":
        sz = SIZE[a:b]
        for z in range(3):
            m = (sz == z) & ~np.isnan(r)
            c = m.sum(1); mean = np.where(c > 0, np.where(m, r, 0).sum(1) / np.maximum(c, 1), np.nan)
            r = np.where(m, r - mean[:, None], r)
        return r
    if how == "sector":
        for g in np.unique(SEC[SEC >= 0]):
            cols = np.flatnonzero(SEC == g)
            r[:, cols] -= np.nanmean(r[:, cols], axis=1, keepdims=True)
        return r
    raise ValueError(how)


def corr_variant(t_split, ret="cc", how="size", hl=None, rank=False, regime=None, window=None, step=1, min_up=None):
    R = {"cc": R_CC, "oc": R_OC, "gap": R_GAP, "rng": R_RNG, "4w": R_4W}[ret]
    a = 1 if window is None else max(1, t_split - window)
    r = demean(R, a, t_split, how)
    if ret == "4w":
        r = r[::4]; step = 4
    mkt = np.nanmean(R_CC[a:t_split], axis=1)
    if regime and ret != "4w":
        sel = mkt < 0 if regime == "down" else mkt > 0
        r = np.where(sel[:, None], r, np.nan)
    if ret == "rng":
        r = r - np.nanmean(r, axis=0, keepdims=True)
    r = np.clip(r, -0.3, 0.3)
    if rank:
        r = pd.DataFrame(r).rank(axis=0, pct=True).to_numpy() - 0.5
    ok = ~np.isnan(r); cnt = ok.sum(0)
    nt = r.shape[0]
    w = np.ones(nt) if hl is None else 0.5 ** ((nt - 1 - np.arange(nt)) / (hl / step))
    x = np.where(ok, r, 0.0)
    sd = np.sqrt(((x ** 2) * w[:, None]).sum(0) / np.maximum((ok * w[:, None]).sum(0), 1e-9))
    x = np.clip(x / np.where(sd > 0, sd, 1), -4, 4)
    xw = x * np.sqrt(w)[:, None]
    okw = ok.astype(np.float32) * np.sqrt(w)[:, None].astype(np.float32)
    corr = (xw.T @ xw) / np.maximum(okw.T @ okw, 1e-9)
    ov = ok.T.astype(np.float32) @ ok.astype(np.float32)
    nsel = int((~np.isnan(r)).any(1).sum()) if regime else nt
    minobs = (int(0.6 * nsel) if min_up is None else min_up) if regime else min(150 // step, int(0.7 * nt))
    keep = cnt >= max(minobs, 30)
    corr[ov < max(min(100 // step, int(0.5 * nt)), 20)] = -9
    corr[~keep, :] = -9; corr[:, ~keep] = -9
    np.fill_diagonal(corr, -9)
    return corr, keep


def corr_any(ts, **kw):
    if kw.get("ret", "cc") == "mix":     # 終値と窓の相関の平均
        kw1 = dict(kw, ret="cc"); kw2 = dict(kw, ret="gap")
        c1, k1 = corr_variant(ts, **kw1); c2, k2 = corr_variant(ts, **kw2)
        c = np.where((c1 > -9) & (c2 > -9), (c1 + c2) / 2, -9)
        return c, k1 & k2
    return corr_variant(ts, **kw)


def evaluate(h, frac, K=200, common=None, **kw):
    E = sp.prep(h); ts = int(NT * frac)
    corr, keep = corr_any(ts, **kw)
    tr = E["t"] < ts
    n, sm = sp.stock_mats(E, tr)
    est = p2.topk_pool(n, sm, corr, np.ones((NC, NC), bool), K)
    te = (E["t"] >= ts) & ~np.isnan(E["x"])
    nt = np.zeros((NS, NC)); st = np.zeros((NS, NC))
    np.add.at(nt, (E["sid"][te], E["c"][te]), 1); np.add.at(st, (E["sid"][te], E["c"][te]), E["x"][te])
    mt = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
    use = (n >= 3) & (nt >= 3) & keep[None, :]
    if common is not None:
        use &= common[None, :]
    ok = use & ~np.isnan(est)
    c = float(np.corrcoef(est[ok], mt[ok])[0, 1])
    cs = [np.corrcoef(est[s][ok[s]], mt[s][ok[s]])[0, 1] for s in range(NS) if ok[s].sum() > 100]
    return c, float(np.nanmean(cs)), int(ok.sum())


VARIANTS = [
    ("基準(終値・週次・市場と規模を引く・全期間)", dict()),
    ("指数加重 半減期10年(520週)", dict(hl=520)), ("指数加重 半減期5年(260週)", dict(hl=260)), ("指数加重 半減期2年(104週)", dict(hl=104)),
    ("リターン: 日中(始値→終値)", dict(ret="oc")), ("リターン: 4週(月次)", dict(ret="4w")), ("値幅(高値-安値)の連動", dict(ret="rng")),
    ("基準を引かない(生のリターン)", dict(how="none")), ("市場平均だけ引く", dict(how="market")), ("業種平均を引く", dict(how="sector")), ("順位相関", dict(rank=True)),
    ("窓(前週終値→始値)", dict(ret="gap")),
    ("下げ相場の週だけ(終値)", dict(regime="down")), ("上げ相場の週だけ(終値)", dict(regime="up")),
    ("窓 × 上げ相場の週だけ", dict(ret="gap", regime="up")), ("窓 × 下げ相場の週だけ", dict(ret="gap", regime="down")),
    ("終値と窓の相関の平均", dict(ret="mix")),
    ("窓 × 市場平均だけ引く", dict(ret="gap", how="market")),
    ("窓 × 指数加重 半減期10年", dict(ret="gap", hl=520)),
    ("窓 × 上げ相場の週 × 指数加重 半減期10年", dict(ret="gap", regime="up", hl=520)),
]
SET = [(4, 0.6), (4, 0.5), (4, 0.7), (1, 0.6), (12, 0.6)]

def common_keep(frac):
    ts = int(NT * frac)
    k = np.ones(NC, bool)
    for kw in (dict(), dict(regime="up"), dict(regime="down"), dict(ret="gap", regime="up"), dict(ret="gap", regime="down")):
        k &= corr_variant(ts, **kw)[1]
    return k


if __name__ == "__main__":
    ck = {f: common_keep(f) for f in (0.5, 0.6, 0.7)}
    rows = []
    for name, kw in VARIANTS:
        res = [evaluate(h, f, common=ck[f], **kw) for h, f in SET]
        cs = [r[0] for r in res]; ws = [r[1] for r in res]
        rows.append(dict(variant=name, avg=np.mean(cs), mn=min(cs), mx=max(cs), within=np.mean(ws), pos=sum(c > 0 for c in cs), pairs=res[0][2]))
        print(f"{name:40s} avg={np.mean(cs):+.4f} [{min(cs):+.4f},{max(cs):+.4f}] within={np.mean(ws):+.4f} pairs={res[0][2]}", flush=True)
    pd.DataFrame(rows).to_csv("results_pool_corrtype.csv", index=False, encoding="utf-8-sig")
