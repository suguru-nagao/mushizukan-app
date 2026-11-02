"""②(反対のシグナルで出る)の、テクニカル指標の出を、「同じ指標の、逆向きのシグナル」に直す。
  python3 tech_exit_patch.py W|D|M   → cache/v2_{tf}_O.pkl のテクニカルの取引を作り直す(元は v2_{tf}_O_before_tech.pkl に退避)、
                                      cache/v2_{tf}_exits.pkl のテクニカルの「出のきっかけ」も更新する
指標のグループ: 移動平均(ゴールデン/デッドクロス・パーフェクトオーダー)、RSI、MACD、ボリンジャー、ストキャスティクス、乖離率、一目、DMI、パラボリックSAR
ほかの種類(ローソク足・チャート・抵抗帯)は、これまでどおり、同じ種類の逆向きのシグナルのどれか。
"""
import os, pickle, shutil, sys
import numpy as np, pandas as pd
import backtest as bt
from build_report import zname
from mtf_run import signals

TF = sys.argv[1] if len(sys.argv) > 1 else "W"
DAYS = {"W": 7.0, "D": 7.0 / 5, "M": 30.4}[TF]; LAST = {"W": 52, "D": 250, "M": 12}[TF]
if TF in ("W", "M"):
    from long_run import load
    P, _ = load(TF)
else:
    P = bt.load_panel()
O = P["O"].astype(float); C = P["C"].astype(float); Cr = P["Craw"].astype(float); T, N = O.shape
meta = pickle.load(open(f"cache/v2_{TF}_meta.pkl", "rb")); names = [s[0] for s in meta["sigs"]]; fam = np.array([s[1] for s in meta["sigs"]]); dd = np.array([s[2] for s in meta["sigs"]])
liq = bt.liquidity_bucket(P); size = np.select([liq == 0, liq <= 2, liq == 3], [-1, 0, 1], 2).astype(np.int8)
ro = np.full((T, N), np.nan); ro[:-1] = O[1:] / O[:-1] - 1
mkt = np.nanmean(np.where(np.abs(ro) < 1, ro, np.nan), axis=1); cum = np.concatenate([[0], np.cumsum(np.log1p(np.nan_to_num(mkt)))])
f_adj = np.where(C > 0, Cr / C, np.nan)
def cost(d, bars):
    return (20.0 if d > 0 else 20.0 + 0.031 * DAYS * np.maximum(bars, 1) / 365 * 1e4) / 1e4
def group(name):
    if name.startswith(("ゴールデンクロス", "デッドクロス")) or "パーフェクトオーダー" in name: return "移動平均"
    for k in ("RSI", "MACD", "ボリンジャー", "ストキャスティクス", "乖離率", "一目", "DMI", "パラボリックSAR"):
        if name.startswith(k): return k
    return None
s = None
if TF == "D" and os.path.exists("cache/tech_sig.pkl"):
    s = pickle.load(open("cache/tech_sig.pkl", "rb"))
if s is None:
    s = signals(P, "tech", TF)
S = {zname(nm): (int(d), np.asarray(v, bool)) for nm, (d, v) in s.items()}
tech_sids = [i for i in range(len(names)) if fam[i] == "tech"]
assert all(names[i] in S for i in tech_sids), [names[i] for i in tech_sids if names[i] not in S]
grp = {i: group(names[i]) for i in tech_sids}; assert all(grp.values()), grp
OPP = {}
for g in set(grp.values()):
    up = [S[names[i]][1] for i in tech_sids if grp[i] == g and dd[i] > 0]; dn = [S[names[i]][1] for i in tech_sids if grp[i] == g and dd[i] < 0]
    OPP[(g, 1)] = np.any(dn, axis=0) if dn else np.zeros((T, N), bool)    # 買いで入った時の、出(同じ指標の売りのどれか)
    OPP[(g, -1)] = np.any(up, axis=0) if up else np.zeros((T, N), bool)
valid = ~P["limit"] & (liq > 0)
rows = []
for sid in tech_sids:
    nm = names[sid]; d = dd[sid]; Sg = S[nm][1]; opp = OPP[(grp[sid], d)]
    t, c = np.nonzero(Sg); ok = valid[t, c] & (t + 1 < T); t, c = t[ok], c[ok]; e = t + 1; ok = O[e, c] > 0; t, c, e = t[ok], c[ok], e[ok]
    for cc in np.unique(c):
        ts = t[c == cc]; op = np.nonzero(opp[:, cc])[0]; last_exit = -1; f = f_adj[:, cc]
        for tt in ts:
            if tt <= last_exit: continue
            ee = tt + 1; k = op[np.searchsorted(op, tt, side="right"):]; xx = k[0] + 1 if len(k) else None
            if xx is not None and xx >= T: xx = None
            if xx is None:
                px = Cr[-1, cc]; pe = O[ee, cc] * f[ee]; r = px / pe - 1; w = T - 1 - ee; m = np.expm1(cum[T - 1] - cum[ee]); xo = -1; last_exit = T
            else:
                if not (O[xx, cc] > 0): continue
                pe, px = O[ee, cc] * f[ee], O[xx, cc] * f[xx]; r = O[xx, cc] / O[ee, cc] - 1; w = xx - ee; m = np.expm1(cum[xx] - cum[ee]); xo = xx; last_exit = xx
            r = float(np.clip(r, -0.9, 3.0)); cst = cost(d, w)
            rows.append((sid, cc, tt, ee, xo, w, pe, px, d * r - cst, d * (r - m) - cst, size[tt, cc]))
NEW = pd.DataFrame(rows, columns=["sid", "c", "t", "e", "x", "w", "pe", "px", "net", "netex", "z"])
NEW = NEW[np.isfinite(NEW.px) & np.isfinite(NEW.net) & np.isfinite(NEW.pe)].copy(); NEW["closed"] = NEW.x >= 0
OO = pd.read_pickle(f"cache/v2_{TF}_O.pkl")
if not os.path.exists(f"cache/v2_{TF}_O_before_tech.pkl"):
    shutil.copy(f"cache/v2_{TF}_O.pkl", f"cache/v2_{TF}_O_before_tech.pkl")
else:
    OO = pd.read_pickle(f"cache/v2_{TF}_O_before_tech.pkl")
keep = OO[~OO.sid.isin(tech_sids)]
OO2 = pd.concat([keep, NEW], ignore_index=True); OO2.to_pickle(f"cache/v2_{TF}_O.pkl")
print("テクニカルの取引", len(OO[OO.sid.isin(tech_sids)]), "→", len(NEW), "(保有の中央値 本数", float(OO[OO.sid.isin(tech_sids) & OO.closed].w.median()), "→", float(NEW[NEW.closed].w.median()), ")", flush=True)
# 出のきっかけ(直近1年)
ex = pickle.load(open(f"cache/v2_{TF}_exits.pkl", "rb"))
ex = {k: v for k, v in ex.items() if fam[k[0]] != "tech"}
rec = NEW[(NEW.x >= T - 1 - LAST) & NEW.closed]
for sid2 in tech_sids:
    for sid in tech_sids:
        if grp[sid] != grp[sid2] or dd[sid] == dd[sid2]: continue
        m = rec[rec.sid == sid]
        if len(m) == 0: continue
        hit = S[names[sid2]][1][(m.x.to_numpy() - 1).clip(0), m.c.to_numpy()]
        for r in m[hit].itertuples():
            ex.setdefault((int(r.sid), int(r.c), int(r.e)), []).append(sid2)
pickle.dump(ex, open(f"cache/v2_{TF}_exits.pkl", "wb"))
print("done")
