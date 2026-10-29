"""zones.py の単体テスト。実行: python3 jp_backtest/test_zones.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zones import detect_zones, make_zones  # noqa: E402

# --- make_zones: 103 付近に出来高が集中していれば、現在値 100 の上に抵抗帯ができる
rng = np.random.default_rng(0)
tp = rng.uniform(92, 108, 200)
v = np.ones(200)
v[(tp > 102.5) & (tp < 104)] = 12          # 103 付近の厚い出来高
rlo, rhi, slo, shi = make_zones(tp, v, 100.0)
assert rlo <= 103.3 <= rhi and rhi / rlo < 1.08, (rlo, rhi)
# 下側(97 付近)に厚い出来高があれば支持帯
v2 = np.ones(200)
v2[(tp > 96.2) & (tp < 97.8)] = 12
rlo, rhi, slo, shi = make_zones(tp, v2, 100.0)
assert slo <= 97 <= shi, (slo, shi)
# 出来高が一様なら、帯は作られない
rlo, rhi, slo, shi = make_zones(tp, np.ones(200), 100.0)
assert np.isnan(rlo) and np.isnan(slo)
# 現在値のすぐ上の帯も、(現在値より上なら)抵抗帯になる。現在値を含む帯は対象外
v3 = np.ones(200); v3[(tp > 101.0) & (tp < 101.8)] = 30
z3 = make_zones(tp, v3, 100.0)
assert z3[0] >= 100.0 and z3[0] <= 101.4 <= z3[1], z3

# --- detect_zones: 上(99.3 付近)に出来高の山。株価は 92〜93 まで下げて横ばい → 帯に向けて上昇 → 跳ね返される → 106 まで上抜け
n = 200
c = np.concatenate([np.full(60, 99.3) + np.sin(np.arange(60)) * .2, np.linspace(99.3, 92, 50), 93 + np.sin(np.arange(40)) * .3,
                    np.linspace(93, 99.6, 20), np.linspace(99.6, 96.5, 15), np.linspace(96.5, 106, 15)])
assert len(c) == n
h, l = c * 1.003, c * 0.997
vol = np.where(np.arange(n) < 60, 20.0, 1.0)  # 最初の 60 本が 99.3 付近で大商い
sg = detect_zones(h[:, None], l[:, None], c[:, None], vol[:, None], W=150, R=5, hi_long=60, hi_short=20)
touch = np.nonzero(sg["抵抗帯に到達(跳ね返される)"][1][:, 0])[0]
brk = np.nonzero(sg["抵抗帯を上抜け"][1][:, 0])[0]
assert len(touch) and all(t >= 150 for t in touch) and touch.min() < 175, touch   # 上昇の途中で帯に到達
assert len(brk) and brk.min() >= 150 and brk.max() >= 185, brk                   # 最後の上昇で上抜け
nh = np.nonzero(sg["新高値更新(短期・20本)"][1][:, 0])[0]
assert len(nh) and nh.max() >= 190, nh                       # 最後の上昇で新高値
assert not sg["抵抗帯に到達(跳ね返される)"][1][:150].any() and not sg["抵抗帯を上抜け"][1][:150].any()  # 窓が埋まる前は出ない

# --- 先読みがないこと
rng = np.random.default_rng(5)
T, N = 400, 25
cc = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
hh, ll = cc * (1 + np.abs(rng.normal(0, .008, (T, N)))), cc * (1 - np.abs(rng.normal(0, .008, (T, N))))
vv = rng.lognormal(10, 1, (T, N))
full = detect_zones(hh, ll, cc, vv, 60, 5, 120, 30)
assert sum(a.sum() for _, a in full.values()) > 200
for X in (200, 300, 360):
    part = detect_zones(hh[:X], ll[:X], cc[:X], vv[:X], 60, 5, 120, 30)
    for name in full:
        assert (full[name][1][:X] == part[name][1]).all(), f"先読みの疑い: {name} X={X}"
print("ok: all zone tests passed;", {k: int(a.sum()) for k, (_, a) in full.items()})
