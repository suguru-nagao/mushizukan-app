"""ローソク足パターン検出。

入力は (日数 T, 銘柄数 N) の 2 次元配列 O/H/L/C(調整後価格)。
返り値は {パターン名: (方向, bool 配列 (T, N))}。方向は +1(買い)/ -1(売り)。
シグナルは「その足の引け後に確定」するものだけを使う(先読みなし)。
"""
import numpy as np

# 判定に使う閾値。バックテストで感度を見られるようにここへ集約する。
TREND_LOOKBACK = 5      # 直前トレンドの判定に使う日数
SMALL_BODY = 0.35       # 「小さい実体」= 実体 / レンジ がこの値以下
LONG_BODY = 0.6         # 「大きい実体」= 実体 / レンジ がこの値以上
SHADOW_BODY_MULT = 2.0  # 「長い影」= 影が実体のこの倍数以上
SHORT_SHADOW = 0.1      # 「ほぼ無い影」= 影 / レンジ がこの値以下


def shift(a, k):
    """k 日前の値を返す(先頭 k 行は NaN)。"""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    else:
        out[:] = a
    return out


def detect(O, H, L, C):
    with np.errstate(invalid="ignore", divide="ignore"):
        rng = H - L
        body = np.abs(C - O)
        top = np.maximum(O, C)
        bot = np.minimum(O, C)
        upper = H - top
        lower = bot - L
        valid = rng > 0
        bull = C > O
        bear = C < O
        body_r = body / rng

        down = C < shift(C, TREND_LOOKBACK)   # 直前は下落
        up = C > shift(C, TREND_LOOKBACK)     # 直前は上昇
        # トレンド判定は「パターンの足の手前」で行うため、1 本前を基準にする
        down1 = shift(down.astype(float), 1) == 1
        up1 = shift(up.astype(float), 1) == 1
        down2 = shift(down.astype(float), 2) == 1
        up2 = shift(up.astype(float), 2) == 1

        small = valid & (body_r <= SMALL_BODY)
        longb = valid & (body_r >= LONG_BODY)

        def s(a, k):  # bool 配列の k 日前(NaN は False)
            return shift(a.astype(float), k) == 1

        pat = {}

        # --- 1 本足 ---
        hammer_shape = small & (lower >= SHADOW_BODY_MULT * body) & (upper <= SHORT_SHADOW * rng)
        shoot_shape = small & (upper >= SHADOW_BODY_MULT * body) & (lower <= SHORT_SHADOW * rng)
        pat["ハンマー"] = (+1, hammer_shape & down1)
        pat["首吊り線"] = (-1, hammer_shape & up1)
        pat["逆ハンマー"] = (+1, shoot_shape & down1)
        pat["流れ星"] = (-1, shoot_shape & up1)
        pat["大陽線"] = (+1, longb & bull)
        pat["大陰線"] = (-1, longb & bear)

        # --- 2 本足(前日=1, 当日=0) ---
        pO, pC, pH, pL = shift(O, 1), shift(C, 1), shift(H, 1), shift(L, 1)
        p_bull, p_bear = s(bull, 1), s(bear, 1)
        p_long = s(longb, 1)

        engulf_up = p_bear & bull & (O <= pC) & (C >= pO) & (body > np.abs(pC - pO))
        engulf_dn = p_bull & bear & (O >= pC) & (C <= pO) & (body > np.abs(pC - pO))
        pat["陽の包み足"] = (+1, engulf_up & down1)
        pat["陰の包み足"] = (-1, engulf_dn & up1)

        harami_up = p_bear & p_long & bull & (top <= shift(np.maximum(O, C), 1)) & (bot >= shift(np.minimum(O, C), 1)) & (body < np.abs(pC - pO))
        harami_dn = p_bull & p_long & bear & (top <= shift(np.maximum(O, C), 1)) & (bot >= shift(np.minimum(O, C), 1)) & (body < np.abs(pC - pO))
        pat["陽のはらみ足"] = (+1, harami_up & down1)
        pat["陰のはらみ足"] = (-1, harami_dn & up1)

        mid_p = (pO + pC) / 2
        pat["切り込み線"] = (+1, p_bear & p_long & bull & (O < pL) & (C > mid_p) & (C < pO) & down1)
        pat["かぶせ線"] = (-1, p_bull & p_long & bear & (O > pH) & (C < mid_p) & (C > pO) & up1)

        pat["毛抜き底"] = (+1, down1 & p_bear & bull & (np.abs(L - pL) <= 0.001 * pL))
        pat["毛抜き天井"] = (-1, up1 & p_bull & bear & (np.abs(H - pH) <= 0.001 * pH))

        # --- 3 本足(2 日前=2, 前日=1, 当日=0) ---
        O2, C2 = shift(O, 2), shift(C, 2)
        mid2 = (O2 + C2) / 2
        star_small = s(small, 1)
        morning = s(bear, 2) & s(longb, 2) & star_small & (np.maximum(pO, pC) < C2) & bull & (C > mid2)
        evening = s(bull, 2) & s(longb, 2) & star_small & (np.minimum(pO, pC) > C2) & bear & (C < mid2)
        pat["明けの明星"] = (+1, morning & down2)
        pat["宵の明星"] = (-1, evening & up2)

        soldiers = bull & p_bull & s(bull, 2) & (C > pC) & (pC > C2) & (O > pO) & (pO > O2) & valid & s(valid, 1) & s(valid, 2)
        crows = bear & p_bear & s(bear, 2) & (C < pC) & (pC < C2) & (O < pO) & (pO < O2) & valid & s(valid, 1) & s(valid, 2)
        pat["赤三兵"] = (+1, soldiers)
        pat["黒三兵"] = (-1, crows)

    # NaN 比較は False になるが、念のため bool 化して返す
    return {k: (d, np.asarray(v, dtype=bool)) for k, (d, v) in pat.items()}
