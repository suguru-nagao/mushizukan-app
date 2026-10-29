"""長期データ(Yahoo Finance)の、株式分割の調整漏れ・一時的な誤値を取り除く。
- 前の足から終値が4倍超(+300%超)に動いた足が、次の足で元に戻る → その足だけを欠損にする(一時的な誤値)。
- 戻らない → 水準が恒久的にずれたもの(調整漏れ)とみなし、その銘柄の、ずれより前の足をすべて欠損にする。
- 始値と終値が4倍以上ずれている足(始値の誤値)は、その足だけを欠損にする。
元データは W_raw.npz / M_raw.npz に残す。
"""
import os
import shutil

import numpy as np

from long_panel import LONG

THR = 3.0
for f in ("W", "M"):
    path = os.path.join(LONG, f"{f}.npz"); raw = os.path.join(LONG, f"{f}_raw.npz")
    if not os.path.exists(raw):
        shutil.copy(path, raw)
    z = np.load(raw, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    O, H, L, C = d["O"].copy(), d["H"].copy(), d["L"].copy(), d["C"].copy()
    T, N = C.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.full_like(C, np.nan); r[1:] = C[1:] / C[:-1] - 1
        r2 = np.full_like(C, np.nan); r2[:-1] = r[1:]
        gap_bad = np.abs(O / C - 1) > THR
        jump = np.abs(r) > THR
        revert = jump & (np.abs((1 + r) * (1 + r2) - 1) < 0.5)
    bar_bad = revert | gap_bad
    perm = jump & ~revert & ~gap_bad
    n_perm = n_bar = 0
    kill = bar_bad.copy()
    for c in np.flatnonzero(perm.any(0)):
        t = np.flatnonzero(perm[:, c]).max()
        kill[:t, c] = True            # ずれより前を捨てる
        n_perm += 1
    n_bar = int(bar_bad.sum())
    for k in ("O", "H", "L", "C", "Va", "Vo", "Craw"):
        a = d[k].astype(np.float64); a[kill] = np.nan; d[k] = a
    d["limit"] = np.where(kill, False, d["limit"])
    np.savez_compressed(path, **d)
    print(f, "一時的な誤値の足", n_bar, "恒久的なずれの銘柄", n_perm, "欠損にした足", int(kill.sum()), "全体", int((~np.isnan(C)).sum()))
