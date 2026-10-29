"""ローソク足パターンのイベントスタディ型バックテスト。

ルール
- シグナルは t 日の引け後に確定。エントリーは翌営業日(t+1)の始値。
- 保有 h 日: 退出は t+1+h 日の始値(h=0 だけは t+1 日の終値 = 日中)。
- 価格は調整後(分割等を反映)。
- 超過リターン = 個別リターン - 同日・同じ流動性分位(または市場区分)の全銘柄平均。
  市場全体の上げ下げと、規模・流動性による差を取り除くための基準。
- 売りシグナルは符号を反転(ショートした場合の損益)。コストは往復で差し引く。
- t+1 日がストップ高/安で終わった銘柄は約定できない可能性があるので除外。
- t 値は「ブロック(max(h,5) 営業日)単位のクラスター」で計算(同日・保有重複の相関を考慮)。
"""
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

from patterns import detect

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
HORIZONS = [0, 1, 2, 3, 5, 10, 20]
MARKETS = {"0111": "プライム", "0112": "スタンダード", "0113": "グロース"}
COST = 0.002  # 往復コスト(手数料・スリッページの目安)。信用売りの貸株料は含まない。


def load_panel():
    """キャッシュから (日付, 銘柄) のパネルを作る。"""
    with gzip.open(os.path.join(CACHE, "master.json.gz"), "rt", encoding="utf-8") as f:
        master = pd.DataFrame(json.load(f))
    master = master[master["Mkt"].isin(MARKETS)].drop_duplicates("Code").set_index("Code")

    frames = []
    for p in sorted(glob.glob(os.path.join(CACHE, "bars", "*.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rows = json.load(f)
        if rows:
            frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Code"].isin(master.index)]
    cols = ["AdjO", "AdjH", "AdjL", "AdjC", "Va", "AdjVo", "C"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df["limit"] = (df["UL"].astype(str) == "1") | (df["LL"].astype(str) == "1")
    # 出来高 0(取引なし)の日は四本値が信用できないので欠損扱い
    df.loc[df["AdjVo"].fillna(0) <= 0, ["AdjO", "AdjH", "AdjL", "AdjC", "Va"]] = np.nan

    def wide(col):
        return df.pivot(index="Date", columns="Code", values=col).sort_index()

    O, H, L, C, Va = (wide(c) for c in ["AdjO", "AdjH", "AdjL", "AdjC", "Va"])
    Vo, Craw = wide("AdjVo"), wide("C")
    lim = wide("limit").reindex_like(O).fillna(False).astype(bool)
    codes = O.columns
    mkt = master.loc[codes, "Mkt"].to_numpy()
    return {
        "dates": pd.to_datetime(O.index), "codes": codes,
        "O": O.to_numpy(), "H": H.to_numpy(), "L": L.to_numpy(), "C": C.to_numpy(),
        "Va": Va.to_numpy(), "Vo": Vo.to_numpy(), "Craw": Craw.to_numpy(), "limit": lim.to_numpy(), "mkt": mkt, "master": master,
    }


def ahead(a, k):
    """k 日先の値を t の行に持ってくる(末尾は NaN)。"""
    out = np.full_like(a, np.nan, dtype=float)
    if k > 0:
        out[:-k] = a[k:]
    else:
        out[:] = a
    return out


def forward_returns(P, horizons=HORIZONS):
    """シグナル日 t ごとの、翌日始値エントリーの保有 h 日リターン (T,N)。約定不能日は NaN。"""
    O, C = P["O"], P["C"]
    entry = ahead(O, 1)
    ok = ~ahead(P["limit"].astype(float), 1).astype(bool) & ~np.isnan(entry)
    rets = {}
    for h in horizons:
        exit_ = ahead(C, 1) if h == 0 else ahead(O, 1 + h)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = exit_ / entry - 1
        r[~ok] = np.nan
        rets[h] = r
    return rets


def liquidity_bucket(P, n=5, window=20):
    """直近 window 日の売買代金の中央値を、日ごとに n 分位へ(1=最小, n=最大)。"""
    va = pd.DataFrame(P["Va"])
    med = va.rolling(window, min_periods=window // 2).median()
    pct = med.rank(axis=1, pct=True)
    b = np.ceil(pct.to_numpy() * n)
    b[np.isnan(b)] = 0
    return b.astype(int)


def baseline(ret, groups):
    """同日・同グループの平均リターン (T,N)。グループ 0 は基準なし(NaN)。"""
    base = np.full_like(ret, np.nan)
    for g in np.unique(groups):
        if g == 0:
            continue
        m = groups == g
        s = np.where(m, np.nan_to_num(ret), 0).sum(axis=1)
        c = (m & ~np.isnan(ret)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(c > 0, s / c, np.nan)
        base = np.where(m, mean[:, None], base)
    return base


def cluster_t(x, t_idx, block):
    """ブロック単位クラスターの t 値(x: 個別の方向付き超過リターン)。"""
    if len(x) < 30:
        return np.nan
    cid = t_idx // block
    m = x.mean()
    s = pd.Series(x - m).groupby(cid).sum()
    se = np.sqrt((s ** 2).sum()) / len(x)
    return m / se if se > 0 else np.nan


def summarize(x, t_idx, h):
    if len(x) == 0:
        return dict(n=0)
    net = x - COST
    return dict(
        n=len(x),
        mean_bp=x.mean() * 1e4,
        median_bp=np.median(x) * 1e4,
        net_bp=net.mean() * 1e4,
        win=(x > 0).mean(),
        t=cluster_t(x, t_idx, max(h, 5)),
    )


def run(P, split_date=None, horizons=HORIZONS):
    """全パターン × 保有期間 × グループ × 期間(全体/前半/後半)の集計表を返す。"""
    rets = forward_returns(P, horizons)
    liq = liquidity_bucket(P)
    mk = np.broadcast_to(
        np.array([{"0111": 1, "0112": 2, "0113": 3}[m] for m in P["mkt"]]), P["O"].shape
    )
    group_sets = {"流動性分位": liq, "市場": mk}
    bases = {(gs, h): baseline(rets[h], g) for gs, g in group_sets.items() for h in horizons}

    pats = detect(P["O"], P["H"], P["L"], P["C"])
    T = len(P["dates"])
    if split_date is None:
        split_date = P["dates"][T // 2]
    period_of = np.where(P["dates"] < split_date, "前半", "後半")

    rows = []
    for name, (d, S) in pats.items():
        for gs, G in group_sets.items():
            for h in horizons:
                r = rets[h] - bases[(gs, h)]
                sel = S & ~np.isnan(r) & (G > 0)
                ti, ci = np.nonzero(sel)
                if len(ti) == 0:
                    continue
                x = d * r[ti, ci]
                g = G[ti, ci]
                for gv in np.unique(g):
                    for per in ["全体", "前半", "後半"]:
                        m = (g == gv) & (True if per == "全体" else period_of[ti] == per)
                        rows.append(dict(
                            パターン=name, 方向="買い" if d > 0 else "売り", 区分=gs, グループ=int(gv),
                            保有日数=h, 期間=per, **summarize(x[m], ti[m], h)))
    return pd.DataFrame(rows), split_date


if __name__ == "__main__":
    P = load_panel()
    print(f"{len(P['dates'])} 営業日 × {len(P['codes'])} 銘柄 ({P['dates'][0].date()} ~ {P['dates'][-1].date()})")
    res, split = run(P)
    out = os.path.join(HERE, "results_main.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"前半/後半の境: {split.date()}  → {out} ({len(res)} 行)")
