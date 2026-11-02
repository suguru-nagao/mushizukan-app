"""市場全体のシグナルと個別銘柄のシグナルが一致している場合だけを対象にした時の、勝率・損益。
  python3 market_regime.py W|D|M
市場全体 = 全銘柄の等ウェイトの指数(各バーの、前の終値に対する始値・高値・安値・終値の変化率の平均から作る。出来高は売買代金の合計)。
市場のシグナル = 個別銘柄と同じ検出(ローソク足・チャート・テクニカル・抵抗帯)を、この指数にかけたもの。
市場の状態 = 同じグループ(ローソク足・チャート・抵抗帯は種類、テクニカルは指標)の、直近の市場シグナルの向き。
  買いシグナルが点灯して、それを否定する売りシグナルが出ていない間は「上」、逆は「下」。点灯したバーの引け後から有効(先読みなし)。
対象: 個別銘柄の買いシグナルは、市場の状態が「上」のときだけ、売りシグナルは「下」のときだけ(シグナルが出た日と、市場シグナルの日は、同時でなくてよい)。
比較: 絞り込まない場合 / 絞り込んだ場合 / 市場の状態だけで入った場合(個別のシグナルなしの、全銘柄・全バー)
"""
import os, pickle, sys
import numpy as np, pandas as pd
import backtest as bt
from build_report import zname
from mtf_run import signals

TF = sys.argv[1] if len(sys.argv) > 1 else "W"
HOLD = {"W": 4, "D": 20, "M": 1}[TF]; AGE = {"W": 4, "D": 10, "M": 2}[TF]
if TF in ("W", "M"):
    from long_run import load
    P, _ = load(TF)
else:
    P = bt.load_panel()
O, H, L, C, Va = (P[k].astype(float) for k in ("O", "H", "L", "C", "Va")); T, N = O.shape
meta = pickle.load(open(f"cache/v2_{TF}_meta.pkl", "rb")); names = [s[0] for s in meta["sigs"]]; fam = np.array([s[1] for s in meta["sigs"]]); dr = np.array([s[2] for s in meta["sigs"]])
# 指数
prev = np.vstack([np.full((1, N), np.nan), C[:-1]])
def rr(a): return np.nanmean(np.clip(a / prev - 1, -0.5, 0.5), axis=1)
ro, rh, rl, rc = rr(O), rr(H), rr(L), rr(C)
ro, rh, rl, rc = (np.nan_to_num(x) for x in (ro, rh, rl, rc))
Ic = np.cumprod(1 + rc) * 100; Ip = np.concatenate([[100], Ic[:-1]])
Io = Ip * (1 + ro); Ih = np.maximum(Ip * (1 + rh), np.maximum(Io, Ic)); Il = np.minimum(Ip * (1 + rl), np.minimum(Io, Ic))
vol = np.nansum(Va, axis=1)
PI = {"O": Io[:, None], "H": Ih[:, None], "L": Il[:, None], "C": Ic[:, None], "Va": vol[:, None], "Vo": vol[:, None], "Craw": Ic[:, None], "limit": np.zeros((T, 1), bool)}
msig = {}
for f in ("candle", "chart", "tech", "zone"):
    s = signals(PI, f, TF)
    for nm, (d, v) in s.items():
        msig[zname(nm)] = np.asarray(v, bool)[:, 0]
    print("市場シグナル", f, len(s), flush=True)
def group(i):
    nm = names[i]
    if fam[i] != "tech": return fam[i]
    if nm.startswith(("ゴールデンクロス", "デッドクロス")) or "パーフェクトオーダー" in nm: return "移動平均"
    for k in ("RSI", "MACD", "ボリンジャー", "ストキャスティクス", "乖離率", "一目", "DMI", "パラボリックSAR"):
        if nm.startswith(k): return k
GR = [group(i) for i in range(len(names))]; gnames = sorted(set(GR)); gid = np.array([gnames.index(g) for g in GR])
STATE = np.zeros((len(gnames), T), np.int8); AGEARR = np.full((len(gnames), T), 9999, np.int32)
for k, g in enumerate(gnames):
    ids = [i for i in range(len(names)) if GR[i] == g]
    b = np.zeros(T, bool); s = np.zeros(T, bool)
    for i in ids:
        v = msig.get(names[i], np.zeros(T, bool))
        if dr[i] > 0: b |= v
        else: s |= v
    st = 0; age = 9999
    for t in range(T):
        if b[t] and not s[t]: st, age = 1, 0
        elif s[t] and not b[t]: st, age = -1, 0
        else: age += 1
        STATE[k, t] = st; AGEARR[k, t] = age
print("市場の状態「上」の割合", {g: round(float((STATE[k] > 0).mean() * 100)) for k, g in enumerate(gnames)}, flush=True)
HHd = pd.read_pickle(f"cache/v2_{TF}_H.pkl"); OO = pd.read_pickle(f"cache/v2_{TF}_O.pkl"); OO = OO[OO.closed]
col = f"n{HOLD}"; xc = f"x{HOLD}"
def stat(df, c, xcol, lab, d, f):
    if len(df) < 100: return None
    return dict(見方=lab, 種類=f, 売買="買い" if d > 0 else "売り", 件数=len(df), 勝率=(df[c] > 0).mean() * 100, 平均bp=df[c].mean() * 1e4, 中央値bp=df[c].median() * 1e4, 市場超過bp=df[xcol].mean() * 1e4)
rows = []
for lab, df, c, xcol in ((f"①{HOLD}{ {'W':'週','D':'日','M':'か月'}[TF] }保有", HHd[np.isfinite(HHd[col])], col, xc), ("②反対のシグナルで出る", OO, "net", "netex")):
    d_ = dr[df.sid.to_numpy()]; g_ = gid[df.sid.to_numpy()]; t_ = df.t.to_numpy().astype(int)
    st = STATE[g_, t_]; ag = AGEARR[g_, t_]; f_ = fam[df.sid.to_numpy()]
    same = st == d_; recent = same & (ag <= AGE)
    for d in (1, -1):
        for f in ("すべて", "candle", "chart", "tech", "zone"):
            m0 = (d_ == d) & ((f_ == f) if f != "すべて" else True)
            for nm2, mask in (("絞り込まない", m0), ("市場と一致(期限なし)", m0 & same), (f"市場と一致(市場シグナルから{AGE}本以内)", m0 & recent), ("市場と逆(参考)", m0 & (st == -d_))):
                r = stat(df[mask], c, xcol, lab + " / " + nm2, d, f)
                if r: rows.append(r)
R = pd.DataFrame(rows).round(1); R.to_csv(f"results_market_regime_{TF}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R[R.種類 == "すべて"].to_string(index=False))
# 市場の状態だけで入った場合(個別のシグナルなし): 全銘柄・全バー、約1か月保有
r = np.full((T, N), np.nan); r[:T - HOLD - 1] = O[HOLD + 1:T] / O[1:T - HOLD] - 1; r = np.clip(np.where(np.abs(r) < 10, r, np.nan), -0.9, 3.0)
print("\n市場の状態だけで入った場合の、全銘柄の平均リターン(約1か月保有、コスト前 bp)")
for k, g in enumerate(gnames):
    up = STATE[k] > 0; dn = STATE[k] < 0
    print(f"  {g:10s} 市場が上: {np.nanmean(r[up]) * 1e4:6.1f}bp / 市場が下: {np.nanmean(r[dn]) * 1e4:6.1f}bp / 全体 {np.nanmean(r) * 1e4:6.1f}bp  (上 {up.mean() * 100:.0f}% 下 {dn.mean() * 100:.0f}%)")
