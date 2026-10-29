"""週足ベースの銘柄別シグナル: 1銘柄ではサンプルが少ないので、似た銘柄のデータを借りて推定する方法を比べる。

比べる方法(前半=学習、後半=検証):
  own    : 銘柄自身の平均(補正なし)
  size   : 同じ規模(小型・中型・大型)の全銘柄の、同じシグナルの平均
  peer   : 値動きの相関が高い銘柄(上位K銘柄)の、同じシグナルの平均
  peerz  : peer のうち、同じ規模の銘柄だけ(相関 × 規模の組み合わせ)
  mix    : 自身 + peerz を、件数で重みづけして混ぜる(件数が少ないほど peerz を信じる)
使い方: python3 stock_pool.py [保有週=4] [学習期間の割合=0.6] [K=30]
相関は、学習期間の週次リターンから、市場全体と規模ごとの平均を引いた残差で計算する(全銘柄が市場に連動する影響を除く)。
"""
import os
import sys

import numpy as np

import backtest as bt

Z = np.load(os.path.join(bt.CACHE, "stock_events_w.npz"), allow_pickle=True)
SID, T, C, SZ = Z["sid"].astype(np.int32), Z["t"].astype(np.int32), Z["c"].astype(np.int32), Z["z"].astype(np.int8)
NAMES, FAMS, DATES, CODES = Z["names"], Z["fams"], Z["dates"], Z["codes"]
RET, SIZE = Z["ret"], Z["size"]
NS, NC, NT = len(NAMES), len(CODES), len(DATES)


def peers(t_split, K):
    """学習期間の残差リターンの相関で、銘柄ごとの上位K銘柄(自身を除く)を返す。(NC, K) と、銘柄の規模(学習期間の最頻値)。"""
    r = RET[1:t_split].astype(np.float64)
    sz = SIZE[1:t_split]
    res = np.full_like(r, np.nan)
    for z in range(3):
        m = (sz == z) & ~np.isnan(r)
        s = np.where(m, r, 0).sum(1); c = m.sum(1)
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        res = np.where(m, r - mean[:, None], res)
    res = np.clip(res, -0.3, 0.3)   # 株式分割・データ誤りなどの極端な値で相関が決まらないようにする
    ok = ~np.isnan(res)
    cnt = ok.sum(0)
    keep = cnt >= 150
    x = np.where(ok, res, 0.0)
    sd = np.sqrt((x ** 2).sum(0) / np.maximum(cnt, 1))
    x = x / np.where(sd > 0, sd, 1)
    x = np.clip(x, -4, 4)
    num = x.T @ x
    ov = ok.T.astype(np.float32) @ ok.astype(np.float32)
    corr = num / np.maximum(ov, 1)
    corr[ov < 100] = -9
    corr[~keep, :] = -9; corr[:, ~keep] = -9
    np.fill_diagonal(corr, -9)
    zmode = np.array([np.bincount(sz[:, j][sz[:, j] >= 0].astype(int), minlength=3).argmax() if (sz[:, j] >= 0).any() else 1 for j in range(NC)])
    nn = np.argsort(-corr, axis=1)[:, :K]
    # 同じ規模だけ
    corr2 = corr.copy()
    corr2[zmode[:, None] != zmode[None, :]] = -9
    nn2 = np.argsort(-corr2, axis=1)[:, :K]
    top = np.take_along_axis(corr, nn, 1)
    return nn, nn2, zmode, keep, float(np.nanmean(np.where(top > -9, top, np.nan)))


def thin(h):
    key = (SID.astype(np.int64) * NC + C) * 4000 + T // h
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def prep(h):
    idx = thin(h)
    return dict(sid=SID[idx], c=C[idx], t=T[idx], z=SZ[idx], x=Z[f"x{h}"][idx].astype(np.float64) * 1e4)


def stock_mats(E, mask):
    """(シグナル, 銘柄) ごとの件数・合計(学習データ)。"""
    ok = mask & ~np.isnan(E["x"])
    n = np.zeros((NS, NC)); sm = np.zeros((NS, NC))
    np.add.at(n, (E["sid"][ok], E["c"][ok]), 1)
    np.add.at(sm, (E["sid"][ok], E["c"][ok]), E["x"][ok])
    return n, sm


def estimates(E, t_split, K, kmix=15):
    nn, nn2, zmode, keep, avgcorr = peers(t_split, K)
    tr = E["t"] < t_split
    n, sm = stock_mats(E, tr)
    own = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
    # size: 規模ごとの全銘柄平均
    ok = tr & ~np.isnan(E["x"])
    gs = np.zeros((NS, 3)); gn = np.zeros((NS, 3))
    np.add.at(gs, (E["sid"][ok], E["z"][ok]), E["x"][ok]); np.add.at(gn, (E["sid"][ok], E["z"][ok]), 1)
    size = (gs / np.maximum(gn, 1))[:, zmode]
    # peer / peerz
    def pool(nnx):
        pn = n[:, nnx].sum(2); ps = sm[:, nnx].sum(2)
        return np.where(pn > 0, ps / np.maximum(pn, 1), np.nan), pn
    peer, pn1 = pool(nn); peerz, pn2 = pool(nn2)
    base = np.where(np.isnan(peerz), size, peerz)
    mix = (n * np.nan_to_num(own) + kmix * base) / (n + kmix)
    return dict(own=own, size=size, peer=peer, peerz=peerz, mix=mix, n=n, zmode=zmode, avgcorr=avgcorr, keep=keep, pn1=pn1, pn2=pn2)


def evaluate(h, frac, K):
    E = prep(h)
    t_split = int(NT * frac)
    est = estimates(E, t_split, K)
    te = (E["t"] >= t_split) & ~np.isnan(E["x"])
    nt = np.zeros((NS, NC)); st = np.zeros((NS, NC))
    np.add.at(nt, (E["sid"][te], E["c"][te]), 1); np.add.at(st, (E["sid"][te], E["c"][te]), E["x"][te])
    mt = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
    use = (est["n"] >= 3) & (nt >= 3) & est["keep"][None, :]
    out = {"pairs": int(use.sum()), "avgcorr": round(est["avgcorr"], 3)}
    meth = ["own", "size", "peer", "peerz", "mix"]
    # (a) 全ペアの相関、(b) シグナル内(銘柄間)の相関の平均
    for m in meth:
        v = est[m]
        ok = use & ~np.isnan(v)
        out["corr_" + m] = round(float(np.corrcoef(v[ok], mt[ok])[0, 1]), 4)
        cs = []
        for s in range(NS):
            o = ok[s]
            if o.sum() > 100:
                cs.append(np.corrcoef(v[s][o], mt[s][o])[0, 1])
        out["within_" + m] = round(float(np.nanmean(cs)), 4)
    # (c) 銘柄ごとに推定上位3シグナルを選んだときの、検証期間の平均(件数加重)
    rng = np.random.default_rng(0)
    tot_all = float((mt[use] * nt[use]).sum() / nt[use].sum())
    out["all_mean"] = round(tot_all, 1)
    for m in meth + ["random"]:
        v = est[m] if m != "random" else rng.random((NS, NC))
        v = np.where(use & ~np.isnan(v), v, -1e9)
        top = np.argsort(-v, axis=0)[:3]               # (3, NC)
        sel = np.zeros_like(use)
        for j in range(3):
            sel[top[j], np.arange(NC)] = True
        sel &= use
        out["top3_" + m] = (round(float((mt[sel] * nt[sel]).sum() / nt[sel].sum()), 1), int(sel.sum()))
    return out


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    r = evaluate(h, frac, K)
    for k, v in r.items():
        print(k, v)
