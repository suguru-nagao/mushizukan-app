"""patterns.py の単体テスト。人工データで定義どおりに検出されるかを確認する。
実行: python3 jp_backtest/test_patterns.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patterns import detect  # noqa: E402


def run(bars):
    """bars: [(O,H,L,C), ...] を 1 銘柄の時系列として検出し、最終日に立ったパターン名の集合を返す。"""
    a = np.array(bars, dtype=float)
    O, H, L, C = (a[:, i:i + 1] for i in range(4))
    res = detect(O, H, L, C)
    return {k for k, (_, v) in res.items() if v[-1, 0]}


# 下落トレンドの前置き(終値が下がり続ける 8 本)
DOWN = [(110 - i, 110 - i + 0.5, 110 - i - 1.5, 110 - i - 1) for i in range(8)]  # 終値 109..102
UP = [(100 + i, 100 + i + 1.5, 100 + i - 0.5, 100 + i + 1) for i in range(8)]     # 終値 101..108

# ハンマー: 下落後、小さい実体 + 長い下ヒゲ + 上ヒゲなし
assert "ハンマー" in run(DOWN + [(100, 100.2, 95, 100.1)]), "ハンマー"
# 同じ形でも上昇後なら首吊り線
assert "首吊り線" in run(UP + [(110, 110.2, 105, 110.1)]), "首吊り線"
assert "ハンマー" not in run(UP + [(110, 110.2, 105, 110.1)])
# 流れ星: 上昇後、小さい実体 + 長い上ヒゲ
assert "流れ星" in run(UP + [(110, 115, 109.9, 110.1)]), "流れ星"

# 陽の包み足: 下落後、陰線を包む陽線
assert "陽の包み足" in run(DOWN + [(102, 102.2, 100.8, 101), (100.8, 103.5, 100.5, 103.2)]), "陽の包み足"
# 包んでいない(終値が前日始値に届かない)なら検出しない
assert "陽の包み足" not in run(DOWN + [(102, 102.2, 100.8, 101), (100.8, 102.0, 100.5, 101.9)])
# 陰の包み足: 上昇後
assert "陰の包み足" in run(UP + [(108, 109, 107.8, 108.8), (109, 109.2, 106.5, 107.0)]), "陰の包み足"

# 明けの明星: 大陰線 → 小さい実体(ギャップダウン) → 陽線(大陰線の中心超え)
assert "明けの明星" in run(DOWN + [(104, 104.1, 99.9, 100), (99, 99.4, 98.4, 98.9), (99.5, 103.5, 99.4, 103)]), "明けの明星"
# 赤三兵
assert "赤三兵" in run([(100, 102.1, 99.9, 102), (101, 104.1, 100.9, 104), (103, 106.1, 102.9, 106)]), "赤三兵"
assert "黒三兵" in run([(106, 106.1, 103.9, 104), (105, 105.1, 101.9, 102), (103, 103.1, 99.9, 100)]), "黒三兵"

# レンジ 0(高値=安値)の足では何も検出しない
assert run(DOWN + [(100, 100, 100, 100)]) == set()

# 先読みがないこと: 最終日を書き換えても、それ以前の日のシグナルは変わらない
base = DOWN + [(100, 100.2, 95, 100.1), (100, 101, 99, 100.5)]
alt = DOWN + [(100, 100.2, 95, 100.1), (50, 200, 10, 190)]
a1, a2 = np.array(base, float), np.array(alt, float)
r1 = detect(*(a1[:, i:i + 1] for i in range(4)))
r2 = detect(*(a2[:, i:i + 1] for i in range(4)))
for k in r1:
    assert (r1[k][1][:-1] == r2[k][1][:-1]).all(), f"先読みあり: {k}"

print("ok: all pattern tests passed")
