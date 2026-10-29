"""日足・週足・月足 × ローソク足・チャートパターン・テクニカル指標・抵抗帯 の一括実行。

使い方: python3 jp_backtest/mtf_run.py zone:D candle:W tech:M ...   (family:時間軸 を並べる)
出力: results_{family}_{tf}_{main|margin_main|variants|matched}.csv
main は流動性分位・市場別の全体/前半/後半、variants はエントリー変種、matched は同じ騰落率の日との比較。
日足の candle は main が既存(results_main.csv)なので、variants と matched だけ実行する(candle:D:vm)。
"""
import os
import sys
import time

import pandas as pd

import backtest as bt
import chart_check as cc
import chart_matched as cm
import timeframe as tfm
from chart_patterns import detect_charts
from patterns import detect
from tech_signals import detect_tech
from zones import SR_PARAMS, TF_PARAMS, detect_zones

HERE = os.path.dirname(os.path.abspath(__file__))


def save(df, fam, tag, kind):
    df.to_csv(os.path.join(HERE, f"results_{fam}_{tag}_{kind}.csv"), index=False, encoding="utf-8-sig")


def signals(P, fam, tf):
    if fam == "candle":
        return detect(P["O"], P["H"], P["L"], P["C"])
    if fam == "chart":
        return detect_charts(P["H"], P["L"], P["C"])
    if fam == "tech":
        return detect_tech(P["O"], P["H"], P["L"], P["C"])
    if fam == "zone":
        return detect_zones(P["H"], P["L"], P["C"], P["Vo"], *TF_PARAMS[tf], sr=SR_PARAMS[tf])
    raise ValueError(fam)


def evaluate(P, U, sigs, fam, tf, main=True, var=True, matched=True, tag=None):
    tag = tag or tf
    hz, gh, ks = tfm.HORIZONS[tf], tfm.GH[tf], tfm.KS[tf]
    split = len(P["dates"]) // 2
    if main:
        m, _ = bt.run(P, horizons=hz, pats=sigs)
        save(m, fam, tag, "main")
        mm, _ = bt.run(P, universe=U, horizons=hz, pats=sigs)
        save(mm, fam, tag, "margin_main")
    if var:
        save(pd.concat([cc.variants_table(P, None, "全銘柄", sigs, split, gh), cc.variants_table(P, U, "貸借のみ", sigs, split, gh)], ignore_index=True), fam, tag, "variants")
    if matched:
        parts = [cm.matched_table(P, sigs, None, "全銘柄", k, gh) for k in ks] + [cm.matched_table(P, sigs, U, "貸借のみ", k, gh) for k in ks]
        save(pd.concat(parts, ignore_index=True), fam, tag, "matched")


if __name__ == "__main__":
    P0 = bt.load_panel()
    U0 = bt.load_margin(P0)
    cache = {"D": (P0, U0)}
    for job in sys.argv[1:]:
        parts = job.split(":")
        fam, tf = parts[0], parts[1]
        only_vm = len(parts) > 2 and parts[2] == "vm"
        if tf not in cache:
            cache[tf] = tfm.resample(P0, U0, tf)
        P, U = cache[tf]
        t0 = time.time()
        sigs = signals(P, fam, tf)
        counts = {k: int(a.sum()) for k, (d, a) in sigs.items()}
        print(f"[{fam}:{tf}] bars={len(P['dates'])} signals={len(sigs)} events={sum(counts.values())} detect={time.time() - t0:.0f}s", flush=True)
        evaluate(P, U, sigs, fam, tf, main=not only_vm)
        pd.Series(counts).to_csv(os.path.join(HERE, f"results_{fam}_{tf}_counts.csv"), header=["events"], encoding="utf-8-sig")
        print(f"[{fam}:{tf}] done in {time.time() - t0:.0f}s", flush=True)
