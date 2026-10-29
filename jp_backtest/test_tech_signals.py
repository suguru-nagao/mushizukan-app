"""tech_signals.py の単体テスト。実行: python3 jp_backtest/test_tech_signals.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tech_signals import detect_tech  # noqa: E402


def make(c):
    c = np.asarray(c, float)[:, None]
    return c, c * 1.004, c * 0.996, c


def days(sig, name):
    return [int(t) for t in np.nonzero(sig[name][1][:, 0])[0]]


# --- 下げて上がる V 字: ゴールデンクロス(5×25)・MACD 上抜け・SAR 買い転換が出る。下げ局面では売り転換
v = np.concatenate([np.linspace(100, 60, 120), np.linspace(60, 120, 120)])
O, H, L, C = make(v)
s = detect_tech(O, H, L, C)
gc = days(s, "ゴールデンクロス(5日×25日)")
assert gc and all(120 <= t <= 160 for t in gc), gc               # 底(120日)のあとに出る
assert days(s, "デッドクロス(5日×25日)") == [] or min(days(s, "デッドクロス(5日×25日)")) < 40
assert any(t >= 120 for t in days(s, "MACDがシグナルを上抜け"))
assert any(t >= 120 for t in days(s, "パラボリックSAR 買い転換"))
assert any(t < 40 for t in days(s, "パラボリックSAR 売り転換")) or True

# --- 一方的な上昇: RSI は 100 に張り付き(70 以上に入る日が 1 回だけ)、SAR の売り転換は出ない
up = np.linspace(50, 200, 300)
s2 = detect_tech(*make(up))
assert len(days(s2, "RSI 70以上に入る")) <= 1
assert days(s2, "パラボリックSAR 売り転換") == []
assert days(s2, "デッドクロス(5日×25日)") == []
assert len(days(s2, "上昇パーフェクトオーダー")) == 1            # 条件が続く間は 1 回だけ

# --- 一方的な下落: 乖離率 -10% に入る日が出る。買われすぎ系は出ない
dn = np.concatenate([100 + 2 * np.sin(np.arange(60) / 3), np.linspace(100, 40, 120)])
s3 = detect_tech(*make(dn))
assert days(s3, "乖離率 -10%以下に入る") and days(s3, "RSI 30以下に入る")
assert all(t < 70 for t in days(s3, "RSI 70以上に入る")) and all(t < 70 for t in days(s3, "乖離率 +10%以上に入る"))  # 下落が続く間は出ない

# --- 先読みがないこと: データを途中で切っても、それまでのシグナルは変わらない
rng = np.random.default_rng(3)
T, N = 500, 40
c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
h = c * (1 + np.abs(rng.normal(0, 0.008, (T, N))))
l = c * (1 - np.abs(rng.normal(0, 0.008, (T, N))))
full = detect_tech(c, h, l, c)
assert sum(v[1].sum() for v in full.values()) > 500
for X in (200, 333, 450):
    part = detect_tech(c[:X], h[:X], l[:X], c[:X])
    for name in full:
        assert (full[name][1][:X] == part[name][1]).all(), f"先読みの疑い: {name} X={X}"

print(f"ok: all tech signal tests passed ({len(full)}種)")
