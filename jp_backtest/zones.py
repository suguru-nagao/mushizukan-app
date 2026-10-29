"""抵抗帯(価格帯別出来高の厚い価格帯)・支持帯・新高値/新安値のシグナル。

どの時間軸(日足・週足・月足)でも同じ関数が使える。バー数で数える(W=窓、R=帯を作り直す間隔)。

抵抗帯の作り方(バー τ の引け後に、τ までのデータだけで作る)
1. 直近 W 本の「典型価格 (高値+安値+終値)/3」に、そのバーの出来高を割り当てて、価格帯別の出来高を作る。
   価格帯は、価格に固定した対数の格子(1 本 約1.5%)で、τ の終値の ±35% の範囲。終値に合わせて区切り直さないので、帯の位置は再計算でずれない。
2. 3 本の移動平均で滑らかにする。現在値より上で最も出来高が厚い帯が、平均の 2 倍以上なら「抵抗帯」。
   同様に、現在値より下で最も厚い帯が「支持帯」。ピークの 60% 以上の隣接帯(両側 2 本まで)は帯に含める。
3. 帯は τ+1 〜 τ+R 本目まで固定して使い、R 本ごとに作り直す。新しい帯ができなければ、終値で上抜けられる(支持帯は下抜けられる)まで前の帯を保つ。
   τ の時点で分かるデータだけを使うので先読みなし。

サポート・レジスタンスライン(チャート上の水平線)も同じ枠組みで作る(sr=True)
- 折り返し点: 前後 K 本の高値(安値)の中で最も高い(安い)バー。右側の K 本が過ぎてから確定する。
- 直近 Wsr 本の折り返し点(高値側・安値側の両方)の価格を、価格の近いもの(幅 1.5% 以内)どうしでまとめ、2 回以上触れた価格をラインとする。
  ラインは中心価格の ±0.75% の細い帯として扱う。現在値より上で最も近いものが抵抗線、下で最も近いものが支持線。
3 つの考え方を比べる: 価格帯別出来高だけ(従来)/ ラインだけ / 両方が重なるとき(出来高の帯の中にラインがある)。
さらに、出来高の帯だけでラインが重ならないもの、ラインだけで出来高の帯が重ならないものも別に評価して、それぞれの寄与を見る。

シグナル(バー t、帯は t より前に確定したもの)
- 抵抗帯に到達(売り): 高値が抵抗帯の下端に初めて届き、終値は帯の上端を超えなかった(跳ね返された)。
- 抵抗帯を上抜け(買い): 終値が抵抗帯の上端を初めて超えた。
- 支持帯に到達(買い): 安値が支持帯の上端に初めて届き、終値は帯の下端を割らなかった。
- 支持帯を下抜け(売り): 終値が支持帯の下端を初めて割った。
- 新高値更新(買い) / 新安値更新(売り): 終値が、直前 N 本の高値(安値)を更新。N は長期・短期の 2 通り。
"""
import numpy as np
import pandas as pd

from chart_patterns import pivot_idx

NB = 46            # 価格帯の数
SPAN = 0.35        # 現在値から ±35%
GAP = 0.0         # 現在値より上(下)にある帯を対象にする(近づいてきた帯も外さない)
PEAK_MULT = 2.0    # 平均の何倍以上なら「厚い」帯か
EXT = 0.6          # ピークの何割以上の隣接帯を含めるか
EXT_N = 2
SR_TOL = 0.015     # 同じラインとみなす価格の幅(まとめた折り返し点の幅)
SR_HALF = 0.0075   # ラインの帯の半幅
SR_MIN = 2         # ラインとみなす最小の接触数


def _clean(H, L, C, V):
    Cs = pd.Series(C).ffill()
    first = Cs.first_valid_index()
    if first is None:
        return None
    c = Cs.to_numpy()[first:]
    h = np.where(np.isnan(H[first:]), c, H[first:])
    l = np.where(np.isnan(L[first:]), c, L[first:])
    v = np.nan_to_num(V[first:], nan=0.0)
    return first, np.maximum(h, c), np.minimum(l, c), c, v


def make_zones(tp, v, close):
    """典型価格 tp と出来高 v(窓内)から、現在値 close に対する (抵抗帯の下端, 上端, 支持帯の下端, 上端)。無ければ NaN。"""
    # 価格に固定した対数の格子(1 本 = 2*SPAN/NB)。終値に合わせて区切り直さないので、帯の位置が再計算でずれない
    step = 2 * SPAN / NB
    k0 = int(np.floor((np.log(close) - SPAN) / step))
    edges = np.exp((k0 + np.arange(NB + 1)) * step)
    h, _ = np.histogram(tp, bins=edges, weights=v)
    if h.sum() <= 0:
        return (np.nan,) * 4
    hs = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    # 「厚い」の基準は、実際に価格が通った範囲の帯の平均(取引のない価格帯を含めない)
    i0 = max(int(np.searchsorted(edges, tp.min())) - 1, 0)
    i1 = min(int(np.searchsorted(edges, tp.max())), NB - 1)
    thr = PEAK_MULT * hs[i0:i1 + 1].mean()
    out = []
    for above in (True, False):
        if above:
            ok = np.nonzero(edges[:-1] >= close * (1 + GAP))[0]
        else:
            ok = np.nonzero(edges[1:] <= close * (1 - GAP))[0]
        if len(ok) == 0:
            out += [np.nan, np.nan]
            continue
        j = ok[np.argmax(hs[ok])]
        if hs[j] < thr:
            out += [np.nan, np.nan]
            continue
        a = j
        while a - 1 >= 0 and j - a < EXT_N and hs[a - 1] >= EXT * hs[j] and (a - 1) in ok:
            a -= 1
        b = j
        while b + 1 < NB and b - j < EXT_N and hs[b + 1] >= EXT * hs[j] and (b + 1) in ok:
            b += 1
        out += [edges[a], edges[b + 1]]
    return tuple(out)


def sr_lines(levels):
    """価格のリスト(折り返し点)を、近いものどうしでまとめ、(中心価格, 接触数) のリストを返す(接触数 SR_MIN 以上)。"""
    if len(levels) < SR_MIN:
        return []
    lv = np.sort(levels)
    out, start = [], 0
    for i in range(1, len(lv) + 1):
        if i == len(lv) or lv[i] > lv[start] * (1 + SR_TOL):
            if i - start >= SR_MIN:
                out.append((float(lv[start:i].mean()), i - start))
            start = i
    return out


def overlap(a_lo, a_hi, b_lo, b_hi):
    return a_lo <= b_hi and a_hi >= b_lo


def zone_sets(vz, lines, close):
    """価格帯別出来高の帯 vz=(抵抗lo, 抵抗hi, 支持lo, 支持hi) と、ライン群から、考え方ごとの抵抗・支持の帯を作る。

    返り値: {集合名: (抵抗lo, 抵抗hi, 支持lo, 支持hi)}。集合名: 出来高(従来) / 線 / 重なり / 出来高のみ / 線のみ。
    """
    nan = np.nan
    bands = [(lv * (1 - SR_HALF), lv * (1 + SR_HALF)) for lv, _ in lines]
    res_l = min((b for b in bands if b[0] >= close), default=None, key=lambda b: b[0])
    sup_l = max((b for b in bands if b[1] <= close), default=None, key=lambda b: b[1])
    out = {"出来高": list(vz), "線": [res_l[0] if res_l else nan, res_l[1] if res_l else nan, sup_l[0] if sup_l else nan, sup_l[1] if sup_l else nan],
           "重なり": [nan] * 4, "出来高のみ": [nan] * 4, "線のみ": [nan] * 4}
    for side, (vlo, vhi, line) in enumerate(((vz[0], vz[1], res_l), (vz[2], vz[3], sup_l))):
        o = 2 * side
        ov = [b for b in bands if not np.isnan(vlo) and overlap(b[0], b[1], vlo, vhi)]
        if not np.isnan(vlo):
            if ov:
                out["重なり"][o], out["重なり"][o + 1] = min([vlo] + [b[0] for b in ov]), max([vhi] + [b[1] for b in ov])
            else:
                out["出来高のみ"][o], out["出来高のみ"][o + 1] = vlo, vhi
        if line is not None and (np.isnan(vlo) or not overlap(line[0], line[1], vlo, vhi)):
            out["線のみ"][o], out["線のみ"][o + 1] = line
    return {k: tuple(v) for k, v in out.items()}


def events_from(h, l, cc, rlo, rhi, slo, shi):
    ph, pl, pc = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]], np.r_[np.nan, cc[:-1]]
    with np.errstate(invalid="ignore"):
        return ((h >= rlo) & (ph < rlo) & (cc <= rhi), (cc > rhi) & (pc <= rhi), (l <= shi) & (pl > shi) & (cc >= slo), (cc < slo) & (pc >= slo))



SET_NAMES = {"出来高": ("", "帯"), "線": ("線: ", "線"), "重なり": ("重なり: ", "帯"), "出来高のみ": ("出来高のみ: ", "帯"), "線のみ": ("線のみ: ", "線")}


def detect_zones(H, L, C, V, W, R, hi_long, hi_short, sr=None):
    """(バー数, 銘柄数) の配列から、抵抗帯・支持帯・新高値/新安値のシグナルを返す。

    sr=(K, Wsr) を渡すと、サポート・レジスタンスライン(折り返し点の前後 K 本、直近 Wsr 本)も作り、
    価格帯別出来高だけ / ラインだけ / 重なり / 出来高のみ / 線のみ の 5 つの考え方でシグナルを作る。sr=None なら従来どおり(出来高だけ)。
    """
    T, N = H.shape
    sets = list(SET_NAMES) if sr else ["出来高"]
    ev_names = ["{}に到達(跳ね返される)", "{}を上抜け", "{}に到達(跳ね返す)", "{}を下抜け"]
    dirs = [-1, +1, +1, -1]
    sig, keymap = {}, {}
    for k in sets:
        pre, unit = SET_NAMES[k]
        for j, (en, d) in enumerate(zip(ev_names, dirs)):
            side = "抵抗" if j < 2 else "支持"
            name = pre + en.format(side + unit)
            sig[name] = (d, np.zeros((T, N), bool))
            keymap[(k, j)] = name
    nh = {}
    for lab, n in (("長期", hi_long), ("短期", hi_short)):
        nh[("高", lab)] = (f"新高値更新({lab}・{n}本)", +1, np.zeros((T, N), bool))
        nh[("安", lab)] = (f"新安値更新({lab}・{n}本)", -1, np.zeros((T, N), bool))
    for c in range(N):
        cl = _clean(H[:, c], L[:, c], C[:, c], V[:, c])
        if cl is None:
            continue
        first, h, l, cc, v = cl
        n = len(cc)
        if n < W + R + 2:
            continue
        tp = (h + l + cc) / 3
        arrs = {k: [np.full(n, np.nan) for _ in range(4)] for k in sets}
        cur = {k: [np.nan] * 4 for k in sets}
        if sr:
            Ks, Wsr = sr
            hp, lp = pivot_idx(h, Ks), pivot_idx(-l, Ks)
        for tau in range(W - 1, n - 1, R):
            vz = make_zones(tp[tau - W + 1:tau + 1], v[tau - W + 1:tau + 1], cc[tau])
            if sr:
                a0, a1 = np.searchsorted(hp, tau - Wsr), np.searchsorted(hp, tau - Ks, side="right")
                b0, b1 = np.searchsorted(lp, tau - Wsr), np.searchsorted(lp, tau - Ks, side="right")
                zs = zone_sets(vz, sr_lines(np.concatenate([h[hp[a0:a1]], l[lp[b0:b1]]])), cc[tau])
            else:
                zs = {"出来高": vz}
            e = min(tau + R, n - 1)
            for k in sets:
                z, cu = zs[k], cur[k]
                # 抵抗帯は、新しい帯ができるか、終値で上抜けられるまで保つ(支持帯は下抜けられるまで)
                r_lo, r_hi = (z[0], z[1]) if not np.isnan(z[0]) else ((cu[0], cu[1]) if cc[tau] <= cu[1] else (np.nan, np.nan))
                s_lo, s_hi = (z[2], z[3]) if not np.isnan(z[2]) else ((cu[2], cu[3]) if cc[tau] >= cu[2] else (np.nan, np.nan))
                cur[k] = [r_lo, r_hi, s_lo, s_hi]
                for q in range(4):
                    arrs[k][q][tau + 1:e + 1] = cur[k][q]
        for k in sets:
            for j, ev in enumerate(events_from(h, l, cc, *arrs[k])):
                sig[keymap[(k, j)]][1][np.nonzero(ev)[0] + first, c] = True
        # 新高値・新安値(直前 N 本の高値/安値を終値で更新)
        for lab, kk in (("長期", hi_long), ("短期", hi_short)):
            if n <= kk:
                continue
            roll_hi = pd.Series(h).shift(1).rolling(kk, min_periods=kk).max().to_numpy()
            roll_lo = pd.Series(l).shift(1).rolling(kk, min_periods=kk).min().to_numpy()
            with np.errstate(invalid="ignore"):
                nh[("高", lab)][2][np.nonzero(cc > roll_hi)[0] + first, c] = True
                nh[("安", lab)][2][np.nonzero(cc < roll_lo)[0] + first, c] = True
    for _, (name, d, arr) in nh.items():
        sig[name] = (d, arr)
    return sig


# 時間軸ごとの標準設定: (窓 W, 作り直し間隔 R, 新高値の長期本数, 短期本数)
TF_PARAMS = {"D": (120, 5, 250, 60), "W": (52, 1, 52, 13), "M": (36, 1, 12, 6)}
# サポレジラインの設定: (折り返し点の前後 K 本, 直近 Wsr 本)
SR_PARAMS = {"D": (5, 250), "W": (3, 104), "M": (2, 60)}
