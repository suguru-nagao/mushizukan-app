"""chart_patterns.py の単体テスト。実行: python3 jp_backtest/test_chart_patterns.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_patterns import detect_charts  # noqa: E402


def series(knots, n):
    xs, ys = zip(*knots)
    c = np.interp(np.arange(n), xs, ys)
    return c * 1.002, c * 0.998, c


def run(knots, n):
    h, l, c = series(knots, n)
    return detect_charts(h[:, None], l[:, None], c[:, None]), (h, l, c)


def days(sig, name):
    return [int(t) for t in np.nonzero(sig[name][1][:, 0])[0]] if name in sig else []


# --- ダブルトップ: 80→100(山1)→90→100(山2)→80。ネックライン(約90)の下抜けで確定
dt = [(0, 80), (20, 80), (60, 100), (75, 90), (90, 100), (120, 80), (130, 80)]
sig, (h, l, c) = run(dt, 131)
d = days(sig, "ダブルトップ")
assert d and all(100 <= t <= 110 for t in d), d
assert sig["ダブルトップ"][0] == -1
# 山2(90日)の確定(95日)より前には出ない
assert min(d) >= 95

# --- ダブルボトム(価格を反転)
sigb, _ = run([(a, 180 - b) for a, b in dt], 131)
db = days(sigb, "ダブルボトム")
assert db and all(100 <= x <= 115 for x in db), (db, d)
assert sigb["ダブルボトム"][0] == +1

# --- 事前トレンドがない(最初から横ばい)と、ダブルトップは出ない
flat = [(0, 100), (40, 100), (55, 90), (70, 100), (90, 100), (120, 80), (130, 80)]
s2, _ = run(flat, 131)
assert days(s2, "ダブルトップ") == [], days(s2, "ダブルトップ")

# --- 三尊: 左肩100 / 頭112 / 右肩100、ネックライン約92
hs = [(0, 80), (20, 80), (45, 100), (60, 92), (75, 112), (90, 92), (105, 100), (140, 80), (150, 80)]
s3, _ = run(hs, 151)
dh = days(s3, "三尊")
assert dh and all(110 <= t <= 130 for t in dh), dh
s3b, _ = run([(a, 180 - b) for a, b in hs], 151)
assert days(s3b, "逆三尊") and all(110 <= x <= 130 for x in days(s3b, "逆三尊")), days(s3b, "逆三尊")

# --- 上昇チャネル: 高値・安値とも 1 日 0.25 ずつ切り上がる平行な 2 本のライン。最後に上抜け
ch = [(0, 100), (20, 106), (30, 101), (40, 111), (50, 106), (60, 116), (70, 111), (80, 121), (90, 116), (105, 135), (120, 135)]
s4, _ = run(ch, 121)
names = [k for k in s4 if k.startswith("上昇チャネル")]
assert any(k.endswith("上抜け") for k in names), list(s4)

# --- 先読みがないこと: データを途中で切っても、それまでの日のシグナルは変わらない
rng = np.random.default_rng(1)
T, N = 600, 40
c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
h, l = c * (1 + np.abs(rng.normal(0, 0.008, (T, N)))), c * (1 - np.abs(rng.normal(0, 0.008, (T, N))))
full = detect_charts(h, l, c)
assert sum(v[1].sum() for v in full.values()) > 50, "乱数でもある程度は検出される"
for X in (300, 420, 500):
    part = detect_charts(h[:X], l[:X], c[:X])
    for name in set(full) | set(part):
        a = full[name][1][:X] if name in full else np.zeros((X, N), bool)
        b = part[name][1] if name in part else np.zeros((X, N), bool)
        assert (a == b).all(), f"先読みの疑い: {name} X={X} 差={int((a != b).sum())}"

print("ok: all chart pattern tests passed;", "検出名:", sorted(full)[:4], "...", f"({len(full)}種)")
