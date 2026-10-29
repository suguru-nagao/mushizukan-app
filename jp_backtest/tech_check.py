"""テクニカル指標シグナルの検証(ローソク足・チャートパターンと同じ枠組み)。

main: 全パターン × 流動性分位 × 保有期間(全銘柄 / 貸借のみ)
variants: 翌日始値 / 1日遅らせる / 日中のみ / 5日保有(サイン方向の損益)
matched: シグナル日(または直近5日)の騰落率が同じ水準の日と比べた差
"""
import os
import pickle
import time

import pandas as pd

import backtest as bt
import chart_check as cc
import chart_matched as cm
from tech_signals import detect_tech

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    P = bt.load_panel()
    U = bt.load_margin(P)
    t0 = time.time()
    sigs = detect_tech(P["O"], P["H"], P["L"], P["C"])
    print("detect secs", round(time.time() - t0))
    for k, (d, S) in sigs.items():
        print(f"{k:34s} {d:+d} events={int(S.sum()):8d}")
    pickle.dump(sigs, open(os.path.join(bt.CACHE, "tech_sig.pkl"), "wb"))
    split = len(P["dates"]) // 2

    m, _ = bt.run(P, pats=sigs)
    m.to_csv(os.path.join(HERE, "results_tech_main.csv"), index=False, encoding="utf-8-sig")
    mm, _ = bt.run(P, universe=U, pats=sigs)
    mm.to_csv(os.path.join(HERE, "results_tech_margin_main.csv"), index=False, encoding="utf-8-sig")
    both = pd.concat([cc.variants_table(P, None, "全銘柄", sigs, split), cc.variants_table(P, U, "貸借のみ", sigs, split)], ignore_index=True)
    both.to_csv(os.path.join(HERE, "results_tech_variants.csv"), index=False, encoding="utf-8-sig")
    mt = pd.concat([cm.matched_table(P, sigs, None, "全銘柄", 1), cm.matched_table(P, sigs, U, "貸借のみ", 1),
                    cm.matched_table(P, sigs, None, "全銘柄", 5), cm.matched_table(P, sigs, U, "貸借のみ", 5)], ignore_index=True)
    mt.to_csv(os.path.join(HERE, "results_tech_matched.csv"), index=False, encoding="utf-8-sig")
    print("saved", len(m), len(mm), len(both), len(mt))
