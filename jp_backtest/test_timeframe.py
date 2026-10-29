"""timeframe.py の単体テスト。実行: python3 jp_backtest/test_timeframe.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timeframe import resample  # noqa: E402

# 2021-10-27(水)〜2021-11-19(金)の平日(祝日は無視)。最初の週は水曜から、最後は金曜まで
dates = pd.bdate_range("2021-10-27", "2021-11-19")
T = len(dates)
rng = np.random.default_rng(2)
C = 100 + np.cumsum(rng.normal(0, 1, (T, 2)), axis=0)
O = C + rng.normal(0, .3, (T, 2)); H = np.maximum(O, C) + .5; L = np.minimum(O, C) - .5
Va = rng.uniform(1, 2, (T, 2)) * 1e6
P = {"dates": dates, "codes": ["a", "b"], "mkt": np.array(["0111", "0111"]), "master": None, "O": O, "H": H, "L": L, "C": C,
     "Va": Va, "Vo": Va / 100, "Craw": C, "limit": np.zeros((T, 2), bool)}
P["limit"][dates.get_loc(pd.Timestamp("2021-11-10")), 1] = True   # 2 週目の水曜に銘柄 b がストップ高
U = np.ones((T, 2), bool)

# --- 週足: 最初の週(水曜始まり=途中)は除き、11/1 の週から 11/19 の週まで 3 本
Q, Uq = resample(P, U, "W")
assert len(Q["dates"]) == 3, Q["dates"]
assert list(Q["dates"].strftime("%m-%d")) == ["11-05", "11-12", "11-19"]
i0, i1 = dates.get_loc(pd.Timestamp("2021-11-01")), dates.get_loc(pd.Timestamp("2021-11-05"))
assert np.allclose(Q["O"][0], O[i0]) and np.allclose(Q["C"][0], C[i1])
assert np.allclose(Q["H"][0], H[i0:i1 + 1].max(axis=0)) and np.allclose(Q["L"][0], L[i0:i1 + 1].min(axis=0))
assert np.allclose(Q["Va"][0], Va[i0:i1 + 1].sum(axis=0))
assert Q["limit"][1, 1] and not Q["limit"][1, 0] and not Q["limit"][0, 1]   # 期間内にストップ高があった週だけ True

# --- 月足: 10 月(27 日から)は途中なので除き、11 月(1日〜19日)は月末まで揃っていないので除く → 0 本
Q2, _ = resample(P, U, "M")
assert len(Q2["dates"]) == 0, Q2["dates"]
# 12 月末まで伸ばすと、11 月が完全な 1 本になる
dates3 = pd.bdate_range("2021-10-27", "2022-01-14"); T3 = len(dates3)
C3 = 100 + np.cumsum(rng.normal(0, 1, (T3, 2)), axis=0)
P3 = {**P, "dates": dates3, "O": C3, "H": C3 + 1, "L": C3 - 1, "C": C3, "Va": np.ones((T3, 2)), "Vo": np.ones((T3, 2)), "Craw": C3, "limit": np.zeros((T3, 2), bool)}
Q3, _ = resample(P3, np.ones((T3, 2), bool), "M")
assert list(Q3["dates"].strftime("%Y-%m-%d")) == ["2021-11-30", "2021-12-31"], Q3["dates"]
n_nov = ((dates3 >= "2021-11-01") & (dates3 <= "2021-11-30")).sum()
assert np.allclose(Q3["Va"][0], n_nov)
print("ok: all timeframe tests passed")
