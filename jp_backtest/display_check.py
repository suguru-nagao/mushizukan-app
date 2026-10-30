"""銘柄の参考シグナルを、どのロジックで表示するのが良いかの検証(週足・4週保有 ≒ 20日、買い・売り別)。
表示するのは、その週に点灯したシグナル。各シグナルに付ける「推定」を、次の方法で比べる(学習期間のデータだけで作り、後半の点灯イベントの実績で評価)。
  size : 同じ規模の全銘柄の、同じシグナルの平均
  own  : その銘柄自身の平均
  peer : 相関が高い同じ規模の上位200銘柄の平均(窓×上げ相場の週)(採用中)
  mix  : 自身と peer を件数で混ぜる(自身 n 件 : peer 15件相当)
  none : 推定なし(点灯したシグナルをすべて同じ扱い)
評価: ①イベント単位の相関 ②推定の上位20%と下位20%の実績 ③推定がコストを超えるものだけを表示した場合の実績。
使い方: python3 display_check.py [buy|sell] [h=4]
"""
import sys

import numpy as np
import pandas as pd

import backtest as bt
import stock_pool as sp
import stock_pool2 as p2
import stock_pool3 as p3

NS, NC, NT = sp.NS, sp.NC, sp.NT


def events_test(h, ts):
    lim = {1: 5000.0, 2: 7000.0, 4: 10000.0, 6: 12000.0, 12: 20000.0}.get(h, 20000.0)
    x = np.clip(sp.Z[f"x{h}"].astype(np.float64) * 1e4, -lim, lim)
    m = (sp.T >= ts) & ~np.isnan(x)
    return sp.SID[m].astype(int), sp.C[m].astype(int), sp.T[m].astype(int), sp.SZ[m].astype(int), x[m]


def prepare(h, frac):
    ts = int(NT * frac)
    E = sp.prep(h)
    n, sm = sp.stock_mats(E, E["t"] < ts)
    corr, keep = p3.corr_any(ts, ret="gap", regime="up")
    zmode = p2.corr_matrix(ts)[1]
    same_z = zmode[:, None] == zmode[None, :]
    peer = p2.topk_pool(n, sm, corr, same_z, 200)
    own = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
    ok = (E["t"] < ts) & ~np.isnan(E["x"])
    gs = np.zeros((NS, 3)); gn = np.zeros((NS, 3))
    np.add.at(gs, (E["sid"][ok], E["z"][ok]), E["x"][ok]); np.add.at(gn, (E["sid"][ok], E["z"][ok]), 1)
    size = (gs / np.maximum(gn, 1))[:, zmode]
    mix = (n * np.nan_to_num(own) + 15 * peer) / (n + 15)
    sid, c, t, z, x = events_test(h, ts)
    est = {"size": size[sid, c], "own": own[sid, c], "peer": peer[sid, c], "mix": mix[sid, c]}
    use = keep[c] & (n[sid, c] >= 3)
    for v in est.values():
        use &= ~np.isnan(v)
    sid, c, t, z, x = sid[use], c[use], t[use], z[use], x[use]
    est = {k: v[use] for k, v in est.items()}
    return sid, c, t, z, x, est


def run(h, frac, cost):
    sid, c, t, z, x, est = prepare(h, frac)
    out = []
    base = x.mean()
    out.append(dict(method="推定なし(点灯をすべて表示)", n=len(x), corr=np.nan, top=base, bottom=base, spread=0.0, spread_t=np.nan, above=base, n_above=len(x), t_above=bt.cluster_t(x / 1e4, t, h)))
    for k, v in est.items():
        q = np.quantile(v, [0.2, 0.8])
        top, bot = v >= q[1], v <= q[0]
        d = np.concatenate([x[top], -x[bot]]); tt = np.concatenate([t[top], t[bot]])
        # 上位と下位の差のt: 上位のイベントを +、下位を - として、週ごとのクラスターで
        sp_t = bt.cluster_t(np.concatenate([x[top] - x.mean(), x.mean() - x[bot]]) / 1e4, tt, h)
        above = v > cost
        out.append(dict(method=k, n=len(x), corr=float(np.corrcoef(v, x)[0, 1]), top=x[top].mean(), bottom=x[bot].mean(), spread=x[top].mean() - x[bot].mean(), spread_t=sp_t,
                        above=x[above].mean() if above.sum() > 30 else np.nan, n_above=int(above.sum()), t_above=bt.cluster_t(x[above] / 1e4, t[above], h) if above.sum() > 100 else np.nan))
    return pd.DataFrame(out)


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "buy"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cost = 20.0 if side == "buy" else 30.0
    rows = []
    for tag, fr in (("選択60%", 0.6), ("選択50%", 0.5), ("未使用80%", 0.8), ("未使用90%", 0.9)):
        df = run(h, fr, cost); df.insert(0, "set", tag); rows.append(df)
        print(tag, flush=True)
    d = pd.concat(rows); d.to_csv(f"results_display_{side}_{h}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(d.groupby("method")[["corr", "top", "bottom", "spread", "above", "n_above"]].mean().round(3).to_string())
