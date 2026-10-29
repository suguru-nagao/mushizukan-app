"""zones.py のサポート・レジスタンスライン部分の単体テスト。実行: python3 jp_backtest/test_zones_sr.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zones import detect_zones, sr_lines, zone_sets  # noqa: E402

# --- sr_lines: 近い価格どうしをまとめ、2 回以上触れたものだけをラインにする
ls = sr_lines(np.array([104.0, 104.5, 110.0, 98.4, 98.6, 98.5, 120.0]))
assert [round(l, 1) for l, _ in ls] == [98.5, 104.2], ls          # 110 と 120 は 1 回だけなので除く
assert dict((round(l, 1), n) for l, n in ls) == {98.5: 3, 104.2: 2}
# 幅が 1.5% を超えて離れた価格はまとめない
assert sr_lines(np.array([100.0, 102.0])) == []

# --- zone_sets: 重なり / 出来高のみ / 線のみ の振り分け
z = zone_sets((103.0, 104.5, 96.0, 97.5), [(104.0, 3), (110.0, 2), (98.5, 2)], 100.0)
assert abs(z["線"][0] - 104.0 * 0.9925) < 1e-9 and abs(z["線"][1] - 104.0 * 1.0075) < 1e-9          # 現在値より上で最も近いライン
assert z["重なり"][0] <= 103.0 and z["重なり"][1] >= 104.5 and not np.isnan(z["重なり"][0])           # 出来高の帯にラインが重なる(抵抗側)
assert np.isnan(z["出来高のみ"][0]) and np.isnan(z["線のみ"][0])                                       # 重なっているので、どちらか一方だけの集合には入らない
assert np.isnan(z["重なり"][2])                                                                      # 支持側は重ならない
assert z["出来高のみ"][2:] == (96.0, 97.5)                                                           # 支持側は出来高の帯だけ
assert abs(z["線のみ"][2] - 98.5 * 0.9925) < 1e-9                                                    # 支持側はラインだけ

# --- detect_zones(sr): 95〜105 を往復する株価。ラインが両端にでき、到達が繰り返し検出される。大商いが 105 付近なら、重なりにも出る
n = 400
t = np.arange(n)
c = 100 + 5 * np.sin(2 * np.pi * t / 40)
h, l = c * 1.002, c * 0.998
vol = np.where(c > 103, 20.0, 1.0)
sg = detect_zones(h[:, None], l[:, None], c[:, None], vol[:, None], W=100, R=5, hi_long=200, hi_short=50, sr=(3, 200))
cnt = lambda k: int(sg[k][1][:, 0].sum())
assert cnt("線: 抵抗線に到達(跳ね返される)") >= 4 and cnt("線: 支持線に到達(跳ね返す)") >= 4, {k: cnt(k) for k in sg}
assert cnt("重なり: 抵抗帯に到達(跳ね返される)") >= 1, "出来高の山とラインが重なる"
assert not sg["線: 抵抗線に到達(跳ね返される)"][1][:100].any()                                         # 窓が埋まる前は出ない

# --- sr=None は従来どおり(出来高だけ、名前も同じ)
sg0 = detect_zones(h[:, None], l[:, None], c[:, None], vol[:, None], W=100, R=5, hi_long=200, hi_short=50)
assert "抵抗帯に到達(跳ね返される)" in sg0 and not any(k.startswith("線") for k in sg0)
assert (sg0["抵抗帯に到達(跳ね返される)"][1] == sg["抵抗帯に到達(跳ね返される)"][1]).all()               # sr の有無で出来高だけの結果は変わらない

# --- 先読みがないこと
rng = np.random.default_rng(7)
T, N = 450, 25
cc = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
hh, ll = cc * (1 + np.abs(rng.normal(0, .008, (T, N)))), cc * (1 - np.abs(rng.normal(0, .008, (T, N))))
vv = rng.lognormal(10, 1, (T, N))
full = detect_zones(hh, ll, cc, vv, 60, 5, 120, 30, sr=(4, 150))
assert sum(a.sum() for _, a in full.values()) > 300
for X in (250, 330, 400):
    part = detect_zones(hh[:X], ll[:X], cc[:X], vv[:X], 60, 5, 120, 30, sr=(4, 150))
    for name in full:
        assert (full[name][1][:X] == part[name][1]).all(), f"先読みの疑い: {name} X={X}"
print("ok: all zone/line tests passed;", {k: int(a.sum()) for k, (_, a) in full.items() if k.startswith(("線", "重なり", "出来高のみ", "線のみ"))})
