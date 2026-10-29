"""チャートパターン(ダブルトップ/ボトム、トリプル、三尊/逆三尊、三角保ち合い、ウェッジ、チャネル)の検出。

仕様書(KabuFull の画面)から読み取れた考え方に沿った実装。画面に写っていない細部は、下の定数にした仮定による。
- 折り返し点(スイングポイント): 前後 K=5 日の高値(安値)の中で最も高い(安い)日。右側の K 日が過ぎてから確定。
- シグナル: ネックライン(または保ち合いのライン)を「終値で」初めて抜けた日。ここで「確定」とみなす。
  折り返し点が確定する(K 日後)より前の抜けは、その時点では形が分からないので採用しない(先読み防止)。
- ダブル/トリプル/三尊系は、事前のトレンドを条件にする: 1つ目の山(谷)より前の40日以内に、そこから5%以上離れた安値(高値)がある。
- 保ち合い系: 直近2つずつの高値・安値を結んだ2本のラインで、値幅の変化率(終点/始点)が 0.75 未満=収束、0.75〜1.25=平行。
- 山を作る側(トップ)で実装し、価格の逆数をとった系列に同じ関数をかけて谷(ボトム)を得る。
返り値: {名前: (方向 +1=買い/-1=売り, 日数×銘柄の bool)}。方向は抜けた向き。
"""
import numpy as np
import pandas as pd

K = 5               # 折り返し点の確認日数(左右)
TOL = 0.03          # ダブル/トリプルで「ほぼ同じ高さ」とみなす幅
DEPTH = 0.05        # ダブルの谷(山)の深さの下限(山の高さに対する比)
DEPTH3 = 0.04       # トリプル
PRE = 0.05          # 事前トレンド: 40日以内に 5% 以上離れた安値(高値)
PRE3 = 0.02         # トリプル: ネックラインの外側 2% 以上から来ている
SHOULDER = 0.10     # 三尊の左右の肩の高さの差(頭に対する比)
HEAD = 1.03         # 頭は肩より 3% 以上高い
MIN_SEP, MAX_SEP = 10, 90
MAX_SPAN = 120
BREAK_WINDOW = 60   # 形が確定してから、抜けを待つ最大日数
FLAT = 0.0004       # 保ち合いの傾きが「水平」とみなす 1 日あたりの割合
CONV, PARA = 0.75, 1.25
MIN_WIDTH = 0.03    # 保ち合いの始点の値幅(価格比)の下限
SCAN = 30           # 保ち合いの抜けを待つ最大日数


def pivot_idx(Hs, k=K):
    """高値の折り返し点の位置(右側 k 日が過ぎた確定済みのもののみ)。"""
    n = len(Hs)
    if n < 2 * k + 1:
        return np.array([], dtype=int)
    win = np.lib.stride_tricks.sliding_window_view(Hs, 2 * k + 1)
    center = win[:, k]
    left = win[:, :k].max(axis=1)
    right = win[:, k + 1:].max(axis=1)
    ok = (center > left) & (center >= right)
    return np.nonzero(ok)[0] + k


def first_break(C, line, start, stop, fresh_from, H=None, cap=None):
    """C が line を下に抜ける最初の日を返す(上に抜ける場合は呼び出し側で符号を反転した系列を渡す)。

    fresh_from〜start-1 の間に抜けていたら None(すでに抜けている)。cap を H が超えたら無効。
    line はスカラーまたは日ごとの配列(長さ n)。
    """
    n = len(C)
    stop = min(stop, n - 1)
    if start > stop:
        return None
    seg_end = stop + 1
    ln = line[fresh_from:seg_end] if hasattr(line, "__len__") else line
    below = C[fresh_from:seg_end] < ln
    if not below.any():
        return None
    p = int(np.argmax(below)) + fresh_from
    if p < start:
        return None  # 確定前にすでに抜けている
    if cap is not None and H[fresh_from:p + 1].max() > cap:
        return None
    return p


def tops(H, L, C, hi):
    """山の形(ダブルトップ・トリプルトップ・三尊)の、ネックライン下抜けの日を返す。"""
    n = len(H)
    out = {"double": set(), "triple": set(), "hs": set()}
    for j, h2 in enumerate(hi):
        P2 = H[h2]
        # --- ダブルトップ
        for h1 in hi[max(0, j - 3):j]:
            sep = h2 - h1
            if sep < MIN_SEP or sep > MAX_SEP:
                continue
            P1 = H[h1]
            if abs(P1 - P2) / max(P1, P2) > TOL:
                continue
            nl = L[h1:h2 + 1].min()
            top = min(P1, P2)
            if (top - nl) / top < DEPTH:
                continue
            pre = L[max(0, h1 - 40):h1]
            if len(pre) < 20 or pre.min() > P1 * (1 - PRE):
                continue
            t = first_break(C, nl, h2 + K, h2 + K + BREAK_WINDOW, h2 + 1, H, max(P1, P2) * 1.03)
            if t is not None:
                out["double"].add(t)
        # --- トリプルトップ
        if j >= 2:
            h1, hm, h3 = hi[j - 2], hi[j - 1], h2
            if hm - h1 >= 8 and h3 - hm >= 8 and h3 - h1 <= MAX_SPAN:
                ps = np.array([H[h1], H[hm], H[h3]])
                mean = ps.mean()
                if (ps.max() - ps.min()) / mean <= TOL:
                    nl = L[h1:h3 + 1].min()
                    if (mean - nl) / mean >= DEPTH3:
                        pre = C[max(0, h1 - 40):h1]
                        if len(pre) >= 20 and pre.min() <= nl * (1 - PRE3) and L[max(0, h1 - 40):h1].min() <= H[h1] * (1 - PRE):
                            t = first_break(C, nl, h3 + K, h3 + K + BREAK_WINDOW, h3 + 1, H, ps.max() * 1.03)
                            if t is not None:
                                out["triple"].add(t)
            # --- 三尊(左肩 h1, 頭 hm, 右肩 h3)
            a, b, c = h1, hm, h3
            Pa, Pb, Pc = H[a], H[b], H[c]
            if b - a >= 8 and c - b >= 8 and c - a <= MAX_SPAN and Pb > max(Pa, Pc) * HEAD and abs(Pa - Pc) / Pb <= SHOULDER:
                i1 = a + int(np.argmin(L[a:b + 1]))
                i2 = b + int(np.argmin(L[b:c + 1]))
                l1, l2 = L[i1], L[i2]
                if i2 > i1 and abs(l1 - l2) / ((l1 + l2) / 2) <= 0.08 and (min(Pa, Pc) - max(l1, l2)) / Pb >= 0.04:
                    pre = L[max(0, a - 40):a]
                    if len(pre) >= 20 and pre.min() <= Pa * (1 - PRE):
                        line = l1 + (l2 - l1) * (np.arange(n) - i1) / (i2 - i1)
                        t = first_break(C, line, c + K, c + K + BREAK_WINDOW, c + 1, H, Pb * 1.01)
                        if t is not None:
                            out["hs"].add(t)
    return out


def channels(H, L, C, hi, lo):
    """保ち合い系(三角・ウェッジ・チャネル)。(日, 名前, 方向) の集合を返す。"""
    n = len(H)
    ev = set()
    piv = sorted([(i, 1) for i in hi] + [(i, -1) for i in lo])
    hs = [i for i, _ in piv if _ == 1]
    ls = [i for i, _ in piv if _ == -1]
    ph = pl = 0  # 現在までに含まれる個数
    for k, (idx, typ) in enumerate(piv):
        if typ == 1:
            ph += 1
        else:
            pl += 1
        if ph < 2 or pl < 2:
            continue
        a2, a1 = hs[ph - 1], hs[ph - 2]
        b2, b1 = ls[pl - 1], ls[pl - 2]
        t0, t1 = min(a1, b1), max(a2, b2)
        if t1 - t0 < 15 or t1 - t0 > 100 or a2 - a1 < 6 or b2 - b1 < 6:
            continue
        tau = idx + K
        if tau >= n - 1:
            continue
        su = (H[a2] - H[a1]) / (a2 - a1)
        sl = (L[b2] - L[b1]) / (b2 - b1)
        up = lambda t: H[a1] + su * (t - a1)
        lw = lambda t: L[b1] + sl * (t - b1)
        w0, w1 = up(t0) - lw(t0), up(t1) - lw(t1)
        mean = (H[a1] + H[a2] + L[b1] + L[b2]) / 4
        if w0 <= 0 or w1 <= 0 or w0 / mean < MIN_WIDTH:
            continue
        r = w1 / w0
        nu, nl_ = su / mean, sl / mean
        uf, lf = abs(nu) <= FLAT, abs(nl_) <= FLAT
        uu, ud = nu > FLAT, nu < -FLAT
        lu, ld = nl_ > FLAT, nl_ < -FLAT
        name = None
        if r < CONV:
            if uf and lu: name = "上昇三角"
            elif ud and lf: name = "下降三角"
            elif ud and lu: name = "対称三角"
            elif uu and lu: name = "上昇ウェッジ"
            elif ud and ld: name = "下降ウェッジ"
        elif r <= PARA:
            if uu and lu: name = "上昇チャネル"
            elif ud and ld: name = "下降チャネル"
        if name is None:
            continue
        # 確定時点で、まだラインの内側にいること(すでに抜けていない)
        if not (lw(tau) < C[tau] < up(tau)):
            continue
        nxt = piv[k + 1][0] + K if k + 1 < len(piv) else n
        for t in range(tau + 1, min(tau + SCAN, nxt, n - 1) + 1):
            if C[t] > up(t):
                ev.add((t, name, 1)); break
            if C[t] < lw(t):
                ev.add((t, name, -1)); break
    return ev


def _clean(H, L, C):
    """欠損を直前の終値で埋め、有効期間(最初の有効日〜)を返す。"""
    C = pd.Series(C).ffill()
    first = C.first_valid_index()
    if first is None:
        return None
    C = C.to_numpy()[first:]
    H = np.where(np.isnan(H[first:]), C, H[first:])
    L = np.where(np.isnan(L[first:]), C, L[first:])
    H, L = np.maximum(H, C), np.minimum(L, C)
    return first, H, L, C


NAMES_TOP = {"double": "ダブルトップ", "triple": "トリプルトップ", "hs": "三尊"}
NAMES_BOT = {"double": "ダブルボトム", "triple": "トリプルボトム", "hs": "逆三尊"}


def detect_charts(H, L, C):
    """(日数, 銘柄数) の配列から、チャートパターンのシグナルを返す。"""
    T, N = H.shape
    sig = {}

    def put(name, d, t, c):
        if name not in sig:
            sig[name] = (d, np.zeros((T, N), bool))
        sig[name][1][t, c] = True

    for c in range(N):
        cl = _clean(H[:, c], L[:, c], C[:, c])
        if cl is None:
            continue
        first, h, l, cc = cl
        if len(h) < 120:
            continue
        hi = pivot_idx(h)
        lo = pivot_idx(-l)
        # 山(売り)
        for key, ts in tops(h, l, cc, hi).items():
            for t in ts:
                put(NAMES_TOP[key], -1, first + t, c)
        # 谷(買い): 価格を反転して同じ関数に
        # 割合の判定が符号で壊れないよう、符号反転ではなく逆数で反転する
        for key, ts in tops(1e4 / l, 1e4 / h, 1e4 / cc, lo).items():
            for t in ts:
                put(NAMES_BOT[key], +1, first + t, c)
        for t, name, d in channels(h, l, cc, hi, lo):
            put(f"{name}・{'上抜け' if d > 0 else '下抜け'}", d, first + t, c)
    return sig
