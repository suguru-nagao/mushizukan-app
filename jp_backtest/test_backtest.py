"""backtest.py の単体テスト(人工データ)。実行: python3 jp_backtest/test_backtest.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

# --- forward_returns: 翌日始値エントリー、h 日後の始値退出 ---
T = 8
O = np.array([[100, 200, 300, 400, 500, 600, 700, 800]], float).T  # 1 銘柄
C = O + 1
P = {"O": O, "C": C, "limit": np.zeros((T, 1), bool)}
r = bt.forward_returns(P, [0, 1, 2])
# t=0: エントリー = O[1]=200
assert np.isclose(r[0][0, 0], C[1, 0] / O[1, 0] - 1)   # h=0 は翌日終値
assert np.isclose(r[1][0, 0], O[2, 0] / O[1, 0] - 1)   # h=1 は翌々日始値
assert np.isclose(r[2][0, 0], O[3, 0] / O[1, 0] - 1)
assert np.isnan(r[2][T - 3, 0]) and not np.isnan(r[2][T - 4, 0])  # 末尾は NaN

# --- ストップ高張り付きのエントリー日は除外 ---
lim = np.zeros((T, 1), bool)
lim[1, 0] = True
r2 = bt.forward_returns({"O": O, "C": C, "limit": lim}, [1])
assert np.isnan(r2[1][0, 0]) and not np.isnan(r2[1][1, 0])

# --- baseline: 同日・同グループ平均 ---
ret = np.array([[0.01, 0.03, 0.05, np.nan]])
groups = np.array([[1, 1, 2, 2]])
b = bt.baseline(ret, groups)
assert np.allclose(b[0], [0.02, 0.02, 0.05, 0.05], equal_nan=True)

# --- 仕込んだ優位性を回収できるか(シグナル後に +2% 上がる世界) ---
rng = np.random.default_rng(0)
T, N = 600, 200
ret_day = rng.normal(0, 0.01, (T, N))
sig = rng.random((T, N)) < 0.02
sig[-30:] = False
# シグナル翌日の始値→翌々日始値に +2% を乗せる(h=1 のリターン)
bump = np.zeros((T, N))
bump[1:] = sig[:-1]
ret_day += 0.02 * bump
price = 100 * np.exp(np.cumsum(ret_day, axis=0))
P = {"O": price, "C": price, "limit": np.zeros((T, N), bool)}
rets = bt.forward_returns(P, [1])
base = bt.baseline(rets[1], np.ones((T, N), int))
ti, ci = np.nonzero(sig & ~np.isnan(rets[1]))
x = (rets[1] - base)[ti, ci]
s = bt.summarize(x, ti, 1)
# ret_day[t+1] の +2% は「t+1 日の終値/始値」ではなく O[t+1]→O[t+2] に乗る。
# 始値=終値の人工価格では O[t+2]/O[t+1] = exp(ret_day[t+2]) なので、バンプは 1 日ずれて乗る点に注意。
sig2 = np.zeros((T, N), bool)
sig2[:-1] = False
bump2 = np.zeros((T, N)); bump2[2:] = sig[:-2]
# 上のシグナルとバンプの位置が 1 日ずれているため、優位性はほぼ回収されないはず(ずれの検出テスト)
assert abs(s["mean_bp"]) < 30, s
# ずれを直した世界で回収できること
ret2 = rng.normal(0, 0.01, (T, N)) + 0.02 * bump2
price2 = 100 * np.exp(np.cumsum(ret2, axis=0))
P2 = {"O": price2, "C": price2, "limit": np.zeros((T, N), bool)}
rr = bt.forward_returns(P2, [1])
bb = bt.baseline(rr[1], np.ones((T, N), int))
ti, ci = np.nonzero(sig & ~np.isnan(rr[1]))
s2 = bt.summarize((rr[1] - bb)[ti, ci], ti, 1)
assert 170 < s2["mean_bp"] < 230 and s2["t"] > 5, s2

# --- 優位性なしの世界では t 値が小さい(偽陽性の確認) ---
ret3 = rng.normal(0, 0.01, (T, N))
price3 = 100 * np.exp(np.cumsum(ret3, axis=0))
P3 = {"O": price3, "C": price3, "limit": np.zeros((T, N), bool)}
r3 = bt.forward_returns(P3, [5])
b3 = bt.baseline(r3[5], np.ones((T, N), int))
ti, ci = np.nonzero(sig & ~np.isnan(r3[5]))
s3 = bt.summarize((r3[5] - b3)[ti, ci], ti, 5)
assert abs(s3["t"]) < 3.5, s3

print("ok: all backtest tests passed", {k: round(float(v), 2) for k, v in s2.items()})
