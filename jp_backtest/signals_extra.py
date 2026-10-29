"""追加シグナルの検証: 窓(ギャップ)、終値位置、出来高との組み合わせ。

- 窓系: 窓上げ/窓下げ × 当日の陽線/陰線(窓埋め or 継続)。方向は窓の向きに合わせる(+なら継続が有利)。
- 終値位置(CLV): レンジが大きい日に、終値が安値圏/高値圏。※これは前の検証で見えた傾向から作った
  探索的な仮説なので、全期間で選んだ点に注意し、前半/後半の再現で判断する。
- 出来高: 既存 18 パターンを「出来高が直近 20 日平均の 2 倍以上」で絞った場合と、通常の場合の比較。
基準・コスト・規模区分は backtest.py と同じ。
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
from patterns import detect, shift

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [("小型(分位1-2)", 1, 2), ("中型(分位3)", 3, 3), ("大型(分位4-5)", 4, 5)]


def extra_signals(P):
    O, H, L, C = P["O"], P["H"], P["L"], P["C"]
    with np.errstate(invalid="ignore", divide="ignore"):
        pH, pL = shift(H, 1), shift(L, 1)
        rng = H - L
        clv = (C - L) / rng
        gap_up, gap_dn = L > pH, H < pL
        gu2, gd2 = L > pH * 1.02, H < pL * 0.98
        wide = rng / C >= 0.03
        sig = {
            "窓上げ陽線": (+1, gap_up & (C > O)), "窓上げ陰線": (+1, gap_up & (C < O)),
            "窓下げ陽線": (-1, gap_dn & (C > O)), "窓下げ陰線": (-1, gap_dn & (C < O)),
            "窓上げ2%超": (+1, gu2), "窓下げ2%超": (-1, gd2),
            "窓上げ(全て)": (+1, gap_up), "窓下げ(全て)": (-1, gap_dn),
            "[探索]終値が安値圏(レンジ3%超)": (+1, wide & (clv <= 0.1)),
            "[探索]終値が高値圏(レンジ3%超)": (-1, wide & (clv >= 0.9)),
        }
    return {k: (d, np.asarray(v, bool)) for k, (d, v) in sig.items()}


def volume_ratio(P, window=20):
    vo = pd.DataFrame(P["Vo"])
    base = vo.rolling(window, min_periods=window // 2).mean().shift(1)
    return (vo / base).to_numpy()


def all_signals(P):
    """追加シグナル + 既存 18 パターンの出来高別版。"""
    vr = volume_ratio(P)
    surge = vr >= 2.0
    normal = (vr > 0.5) & (vr < 2.0)
    sigs = extra_signals(P)
    for k, (d, S) in detect(P["O"], P["H"], P["L"], P["C"]).items():
        sigs[f"{k}|出来高急増"] = (d, S & surge)
        sigs[f"{k}|通常出来高"] = (d, S & normal)
    return sigs


if __name__ == "__main__":
    P = bt.load_panel()
    T = len(P["dates"])
    split = T // 2
    liq = bt.liquidity_bucket(P)
    rets = bt.forward_returns(P, [1, 5])
    xs = {h: rets[h] - bt.baseline(rets[h], liq) for h in (1, 5)}
    sigs = all_signals(P)

    rows = []
    for name, (d, S) in sigs.items():
        for size, lo, hi in SIZES:
            for h in (1, 5):
                sel = S & (liq >= lo) & (liq <= hi) & ~np.isnan(xs[h])
                t, c = np.nonzero(sel)
                x = d * xs[h][t, c]
                for per, pm in [("全体", t >= 0), ("前半", t < split), ("後半", t >= split)]:
                    rows.append(dict(シグナル=name, 規模=size, 保有日数=h, 期間=per,
                                     **bt.summarize(x[pm], t[pm], max(h, 1))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results_extra.csv"), index=False, encoding="utf-8-sig")
    print("saved", len(res))
