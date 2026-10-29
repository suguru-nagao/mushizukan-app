"""テクニカル指標のシグナル検出(日足・標準パラメータ)。

入力は (日数, 銘柄数) の調整後 O/H/L/C。返り値は {名前: (方向 +1=買い/-1=売り, 日数×銘柄の bool)}。
シグナルは「状態が切り替わった日」(クロスした日、しきい値に入った日)に 1 日だけ立つ。判定には当日の引け値までしか使わない。
方向は教科書の読み方(逆張り系は逆張りの向き)。順張りで使う場合は、結果の符号が逆になる。

指標と標準パラメータ
- 移動平均: 5/25/75 日の単純移動平均。ゴールデン/デッドクロス(5×25、25×75)、パーフェクトオーダー(終値>5>25>75、逆も)。
- RSI(14, ワイルダー): 30 以下に入った日=買い、70 以上に入った日=売り(逆張り)。30 を上抜けた日=買い、70 を下抜けた日=売り。
- MACD(12, 26, 9): MACD がシグナルを上(下)抜け=買い(売り)。MACD がゼロを上(下)抜け=買い(売り)。
- ボリンジャーバンド(20日, ±2σ): 終値が -2σ を割った日=買い、+2σ を超えた日=売り(逆張り)。
- ストキャスティクス(14, 3, 3, スロー): %K が %D を上抜け、かつ %D ≤ 20=買い。下抜け、かつ %D ≥ 80=売り。
- 移動平均乖離率(25日): -10% 以下に入った日=買い、+10% 以上に入った日=売り(逆張り)。
- 一目均衡表(9, 26, 52): 雲の上抜け/下抜け、転換線×基準線(好転/逆転)、三役好転/逆転。
- DMI/ADX(14, ワイルダー): +DI が -DI を上(下)抜け、かつ ADX > 25 =買い(売り)。
- パラボリック SAR(加速 0.02, 最大 0.2): 買い転換/売り転換。
"""
import numpy as np
import pandas as pd


def _df(a):
    return pd.DataFrame(a)


def cross_up(a, b):
    return ((a > b) & (a.shift(1) <= b.shift(1))).to_numpy()


def cross_dn(a, b):
    return ((a < b) & (a.shift(1) >= b.shift(1))).to_numpy()


def enter_below(x, th):
    return ((x <= th) & (x.shift(1) > th)).to_numpy()


def enter_above(x, th):
    return ((x >= th) & (x.shift(1) < th)).to_numpy()


def first_true(cond):
    """条件が False(または未計算) → True に変わった日。"""
    prev = cond.shift(1, fill_value=False)
    return (cond & ~prev).to_numpy()


def sar_flips(H, L, C, af0=0.02, af_max=0.2):
    """パラボリック SAR の転換日(買い転換, 売り転換)。H/L/C は (T, N)。銘柄ごとに状態を持つ。"""
    T, N = H.shape
    up_flip = np.zeros((T, N), bool)
    dn_flip = np.zeros((T, N), bool)
    for c in range(N):
        h, l, cl = H[:, c], L[:, c], C[:, c]
        ok = np.nonzero(~(np.isnan(h) | np.isnan(l) | np.isnan(cl)))[0]
        if len(ok) < 5:
            continue
        i0 = ok[0]
        h, l, cl = h.tolist(), l.tolist(), cl.tolist()
        trend = 1 if cl[i0 + 1] >= cl[i0] else -1 if not np.isnan(cl[i0 + 1]) else 1
        sar = l[i0] if trend == 1 else h[i0]
        ep = h[i0] if trend == 1 else l[i0]
        af = af0
        for t in range(i0 + 1, T):
            if np.isnan(h[t]) or np.isnan(l[t]):
                continue
            sar = sar + af * (ep - sar)
            if trend == 1:
                sar = min(sar, l[t - 1] if not np.isnan(l[t - 1]) else sar, l[t - 2] if t - 2 >= i0 and not np.isnan(l[t - 2]) else sar)
                if l[t] < sar:
                    trend, sar, ep, af = -1, ep, l[t], af0
                    dn_flip[t, c] = True
                elif h[t] > ep:
                    ep = h[t]
                    af = min(af + af0, af_max)
            else:
                sar = max(sar, h[t - 1] if not np.isnan(h[t - 1]) else sar, h[t - 2] if t - 2 >= i0 and not np.isnan(h[t - 2]) else sar)
                if h[t] > sar:
                    trend, sar, ep, af = 1, ep, h[t], af0
                    up_flip[t, c] = True
                elif l[t] < ep:
                    ep = l[t]
                    af = min(af + af0, af_max)
    return up_flip, dn_flip


def detect_tech(O, H, L, C):
    Cd, Hd, Ld = _df(C), _df(H), _df(L)
    sig = {}

    def put(name, d, arr):
        sig[name] = (d, np.asarray(arr, bool))

    # --- 移動平均
    sma = {n: Cd.rolling(n, min_periods=n).mean() for n in (5, 25, 75)}
    put("ゴールデンクロス(5日×25日)", +1, cross_up(sma[5], sma[25]))
    put("デッドクロス(5日×25日)", -1, cross_dn(sma[5], sma[25]))
    put("ゴールデンクロス(25日×75日)", +1, cross_up(sma[25], sma[75]))
    put("デッドクロス(25日×75日)", -1, cross_dn(sma[25], sma[75]))
    po_up = (Cd > sma[5]) & (sma[5] > sma[25]) & (sma[25] > sma[75])
    po_dn = (Cd < sma[5]) & (sma[5] < sma[25]) & (sma[25] < sma[75])
    put("上昇パーフェクトオーダー", +1, first_true(po_up))
    put("下降パーフェクトオーダー", -1, first_true(po_dn))

    # --- RSI(14)
    diff = Cd.diff()
    gain, loss = diff.clip(lower=0), (-diff).clip(lower=0)
    ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    rsi = rsi.where(~((al == 0) & (ag > 0)), 100.0)
    put("RSI 30以下に入る", +1, enter_below(rsi, 30))
    put("RSI 70以上に入る", -1, enter_above(rsi, 70))
    put("RSI 30を上抜け", +1, cross_up(rsi, pd.DataFrame(30.0, index=rsi.index, columns=rsi.columns)))
    put("RSI 70を下抜け", -1, cross_dn(rsi, pd.DataFrame(70.0, index=rsi.index, columns=rsi.columns)))

    # --- MACD(12, 26, 9)
    macd = Cd.ewm(span=12, adjust=False, min_periods=12).mean() - Cd.ewm(span=26, adjust=False, min_periods=26).mean()
    msig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    zero = pd.DataFrame(0.0, index=macd.index, columns=macd.columns)
    put("MACDがシグナルを上抜け", +1, cross_up(macd, msig))
    put("MACDがシグナルを下抜け", -1, cross_dn(macd, msig))
    put("MACDがゼロを上抜け", +1, cross_up(macd, zero))
    put("MACDがゼロを下抜け", -1, cross_dn(macd, zero))

    # --- ボリンジャーバンド(20, ±2σ)
    m20, s20 = Cd.rolling(20, min_periods=20).mean(), Cd.rolling(20, min_periods=20).std(ddof=0)
    lo_b, up_b = m20 - 2 * s20, m20 + 2 * s20
    put("ボリンジャー -2σ割れ", +1, ((Cd < lo_b) & (Cd.shift(1) >= lo_b.shift(1))).to_numpy())
    put("ボリンジャー +2σ超え", -1, ((Cd > up_b) & (Cd.shift(1) <= up_b.shift(1))).to_numpy())

    # --- ストキャスティクス(14, 3, 3)
    ll, hh = Ld.rolling(14, min_periods=14).min(), Hd.rolling(14, min_periods=14).max()
    raw = (Cd - ll) / (hh - ll).replace(0, np.nan) * 100
    k = raw.rolling(3, min_periods=3).mean()
    d = k.rolling(3, min_periods=3).mean()
    put("ストキャスティクス 売られすぎでクロス上抜け", +1, (cross_up(k, d) & (d <= 20).to_numpy()))
    put("ストキャスティクス 買われすぎでクロス下抜け", -1, (cross_dn(k, d) & (d >= 80).to_numpy()))

    # --- 移動平均乖離率(25日)
    dev = (Cd / sma[25] - 1) * 100
    put("乖離率 -10%以下に入る", +1, enter_below(dev, -10))
    put("乖離率 +10%以上に入る", -1, enter_above(dev, 10))

    # --- 一目均衡表
    ten = (Hd.rolling(9, min_periods=9).max() + Ld.rolling(9, min_periods=9).min()) / 2
    kij = (Hd.rolling(26, min_periods=26).max() + Ld.rolling(26, min_periods=26).min()) / 2
    spa = ((ten + kij) / 2).shift(26)
    spb = ((Hd.rolling(52, min_periods=52).max() + Ld.rolling(52, min_periods=52).min()) / 2).shift(26)
    ctop, cbot = np.maximum(spa, spb), np.minimum(spa, spb)
    put("一目: 雲を上抜け", +1, ((Cd > ctop) & (Cd.shift(1) <= ctop.shift(1))).to_numpy())
    put("一目: 雲を下抜け", -1, ((Cd < cbot) & (Cd.shift(1) >= cbot.shift(1))).to_numpy())
    put("一目: 転換線が基準線を上抜け(好転)", +1, cross_up(ten, kij))
    put("一目: 転換線が基準線を下抜け(逆転)", -1, cross_dn(ten, kij))
    sanyaku_up = (ten > kij) & (Cd > ctop) & (Cd > Cd.shift(26))
    sanyaku_dn = (ten < kij) & (Cd < cbot) & (Cd < Cd.shift(26))
    put("一目: 三役好転", +1, first_true(sanyaku_up))
    put("一目: 三役逆転", -1, first_true(sanyaku_dn))

    # --- DMI / ADX(14)
    up_move, dn_move = Hd.diff(), -Ld.diff()
    pdm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0).where(up_move.notna())
    mdm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0).where(dn_move.notna())
    pc = Cd.shift(1)
    tr = pd.DataFrame(np.maximum(np.maximum((Hd - Ld).to_numpy(), (Hd - pc).abs().to_numpy()), (Ld - pc).abs().to_numpy()))
    w = dict(alpha=1 / 14, adjust=False, min_periods=14)
    atr = tr.ewm(**w).mean()
    pdi, mdi = 100 * pdm.ewm(**w).mean() / atr, 100 * mdm.ewm(**w).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(**w).mean()
    put("DMI: +DIが-DIを上抜け(ADX>25)", +1, cross_up(pdi, mdi) & (adx > 25).to_numpy())
    put("DMI: +DIが-DIを下抜け(ADX>25)", -1, cross_dn(pdi, mdi) & (adx > 25).to_numpy())

    # --- パラボリック SAR
    up_f, dn_f = sar_flips(H, L, C)
    put("パラボリックSAR 買い転換", +1, up_f)
    put("パラボリックSAR 売り転換", -1, dn_f)
    return sig
