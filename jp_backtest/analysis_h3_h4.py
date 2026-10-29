"""仮説3(効きやすい銘柄)・仮説4(効きやすい局面)の検証。

イベント = (シグナル日 t, 銘柄 c, パターン)。値は backtest.py と同じ「方向付き超過リターン」
(翌日始値エントリー、同日・同流動性分位の平均を控除、売りは符号反転)。

H3: 前半(探索)の銘柄別の平均が、後半(検証)でも続くか。逆向きも見る。
    パターン別では銘柄あたりのサンプルが少なすぎるので、パターンを束ねて見る。
H4: 局面(TOPIX のトレンド・ボラティリティ)ごとに平均が違うか。差の t 値と前半/後半での再現を見る。
"""
import gzip
import json
import os

import numpy as np
import pandas as pd

import backtest as bt
from patterns import detect

HERE = os.path.dirname(os.path.abspath(__file__))
HS = [1, 5]
POOLS = {
    "全パターン": None,
    "実体系": ["大陽線", "大陰線", "赤三兵", "黒三兵", "陽の包み足", "陰の包み足", "陽のはらみ足",
              "陰のはらみ足", "切り込み線", "かぶせ線", "明けの明星", "宵の明星"],
    "ヒゲ系": ["ハンマー", "首吊り線", "逆ハンマー", "流れ星", "毛抜き底", "毛抜き天井"],
}


def build_events(P):
    rets = bt.forward_returns(P, HS)
    liq = bt.liquidity_bucket(P)
    xs = {h: rets[h] - bt.baseline(rets[h], liq) for h in HS}
    pats = detect(P["O"], P["H"], P["L"], P["C"])
    parts = []
    for name, (d, S) in pats.items():
        sel = S & (liq > 0) & ~np.isnan(xs[1])
        t, c = np.nonzero(sel)
        df = pd.DataFrame({"t": t, "c": c, "pat": name, "liq": liq[t, c]})
        for h in HS:
            df[f"x{h}"] = d * xs[h][t, c]
        parts.append(df)
    ev = pd.concat(parts, ignore_index=True)
    ev["size"] = pd.cut(ev["liq"], [0, 2, 3, 5], labels=["小型(分位1-2)", "中型(分位3)", "大型(分位4-5)"])
    return ev


def topix_regimes(dates):
    with gzip.open(os.path.join(bt.CACHE, "topix.json.gz"), "rt") as f:
        tp = pd.DataFrame(json.load(f))
    tp["Date"] = pd.to_datetime(tp["Date"])
    c = tp.set_index("Date")["C"].reindex(dates)
    trend = np.where(c > c.rolling(50, min_periods=50).mean(), "上昇(50日線の上)", "下落(50日線の下)")
    trend = pd.Series(trend, index=dates).where(c.rolling(50, min_periods=50).mean().notna())
    vol = c.pct_change().rolling(20, min_periods=20).std()
    # 全期間の三分位(局面ラベルに全期間の分布を使う点は、結果の注意事項として明記)
    q1, q2 = vol.quantile([1 / 3, 2 / 3])
    volreg = pd.Series(np.where(vol <= q1, "低ボラ", np.where(vol >= q2, "高ボラ", "中ボラ")), index=dates).where(vol.notna())
    return trend.to_numpy(), volreg.to_numpy()


def spearman(a, b):
    """順位相関(scipy 不要)。"""
    return a.rank().corr(b.rank())


def mean_se(x, t, block):
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n
    m = x.mean()
    s = pd.Series(x - m).groupby(t // block).sum()
    return m, np.sqrt((s ** 2).sum()) / n, n


# ---------------------------------------------------------------- H3
def h3(ev, P, split_idx):
    out = []
    logva = pd.Series(np.log(np.nanmedian(np.where(P["Va"] > 0, P["Va"], np.nan), axis=0)), index=range(len(P["codes"])))
    for pool, names in POOLS.items():
        e = ev if names is None else ev[ev["pat"].isin(names)]
        for h in HS:
            col = f"x{h}"
            for a, b, label in [("前", "後", "前半→後半"), ("後", "前", "後半→前半")]:
                ia = e[e["t"] < split_idx] if a == "前" else e[e["t"] >= split_idx]
                ib = e[e["t"] >= split_idx] if a == "前" else e[e["t"] < split_idx]
                ga = ia.groupby("c")[col].agg(["mean", "count"])
                gb = ib.groupby("c")[col].agg(["mean", "count"])
                j = ga.join(gb, lsuffix="_a", rsuffix="_b", how="inner")
                j = j[(j["count_a"] >= 20) & (j["count_b"] >= 20)]
                if len(j) < 100:
                    continue
                r = spearman(j["mean_a"], j["mean_b"])
                tstat = r * np.sqrt((len(j) - 2) / (1 - r ** 2))
                # 流動性の影響を除いた相関
                lv = logva.reindex(j.index)
                res = {}
                for k in ["mean_a", "mean_b"]:
                    beta = np.polyfit(lv, j[k], 1)
                    res[k] = j[k] - np.polyval(beta, lv)
                r2 = spearman(res["mean_a"], res["mean_b"])
                j["qa"] = pd.qcut(j["mean_a"], 5, labels=False, duplicates="drop") + 1
                qm = j.groupby("qa")["mean_b"].mean() * 1e4
                out.append(dict(束ね=pool, 保有日数=h, 方向=label, 銘柄数=len(j), 順位相関=r, t=tstat,
                                流動性除去後の相関=r2, 探索期間の下位20pct銘柄の検証期間bp=qm.iloc[0],
                                探索期間の上位20pct銘柄の検証期間bp=qm.iloc[-1]))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- H4
def h4(ev, regimes, split_idx):
    ev = ev.copy()
    for k, v in regimes.items():
        ev[k] = v[ev["t"].to_numpy()]
    rows = []
    for (pat, size), g in ev.groupby(["pat", "size"], observed=True):
        for h in HS:
            col = f"x{h}"
            for rk, (hi, lo) in {"トレンド": ("上昇(50日線の上)", "下落(50日線の下)"), "ボラ": ("高ボラ", "低ボラ")}.items():
                res = {}
                for per, pm in [("全体", g["t"] >= 0), ("前半", g["t"] < split_idx), ("後半", g["t"] >= split_idx)]:
                    ga = g[pm & (g[rk] == hi)]
                    gb = g[pm & (g[rk] == lo)]
                    ma, sa, na = mean_se(ga[col].to_numpy(), ga["t"].to_numpy(), max(h, 5))
                    mb, sb, nb = mean_se(gb[col].to_numpy(), gb["t"].to_numpy(), max(h, 5))
                    diff = ma - mb
                    res[per] = (ma, mb, diff, diff / np.sqrt(sa ** 2 + sb ** 2) if sa == sa and sb == sb else np.nan, na, nb)
                rows.append(dict(
                    パターン=pat, 規模=size, 局面=rk, 保有日数=h, 比較=f"{hi} − {lo}",
                    A_bp=res["全体"][0] * 1e4, B_bp=res["全体"][1] * 1e4, 差_bp=res["全体"][2] * 1e4, 差のt=res["全体"][3],
                    nA=res["全体"][4], nB=res["全体"][5],
                    前半差_bp=res["前半"][2] * 1e4, 後半差_bp=res["後半"][2] * 1e4))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = bt.load_panel()
    T = len(P["dates"])
    split_idx = T // 2
    ev = build_events(P)
    print(f"events: {len(ev):,}  split_idx={split_idx} ({P['dates'][split_idx].date()})")
    ev.to_pickle(os.path.join(bt.CACHE, "events.pkl"))
    r3 = h3(ev, P, split_idx)
    r3.to_csv(os.path.join(HERE, "results_h3.csv"), index=False, encoding="utf-8-sig")
    trend, vol = topix_regimes(P["dates"])
    r4 = h4(ev, {"トレンド": trend, "ボラ": vol}, split_idx)
    r4.to_csv(os.path.join(HERE, "results_h4.csv"), index=False, encoding="utf-8-sig")
    print("saved results_h3.csv, results_h4.csv")
