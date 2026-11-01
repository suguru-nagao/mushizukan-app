"""全シグナルの売買成績を、2つの見方で計算する(週足: 長期データ 2000〜2026 / 日足: J-Quants 2021〜2026)。
 ① 保有期間を決める: シグナルの次のバーの始値で入り、h本後の始値で出る(毎回のシグナルを独立に数える)。
 ② 反対のシグナルで出る: シグナルの次のバーの始値で入り、同じ種類(ローソク足/チャート/テクニカル/抵抗帯)の逆向きのシグナルのどれかが出た次のバーの始値で出る。保有中に出た同じシグナルは数えない。
損益 = 向き×(出た始値/入った始値−1) − コスト。買いは20bp、売りは20bp+貸株料等(年3.1%×保有日数/365)。勝率 = コスト後の損益がプラスの割合。
出力: cache/v2_{tf}_H.pkl, v2_{tf}_O.pkl, v2_{tf}_meta.pkl
  python3 engine_v2.py W|D
"""
import os, pickle, sys
import numpy as np, pandas as pd
import backtest as bt
from build_report import zname
from mtf_run import signals

TF = sys.argv[1] if len(sys.argv) > 1 else "W"
HOLDS = {"W": [1, 2, 4, 6, 12], "D": [5, 10, 20, 30]}[TF]
DAYS = {"W": 7.0, "D": 7.0 / 5}[TF]       # 1バーあたりの日数
if TF == "W":
    from long_run import load
    P, _ = load("W")
else:
    P = bt.load_panel()
O = P["O"].astype(float); C = P["C"].astype(float); Cr = P["Craw"].astype(float); T, N = O.shape
dates = pd.DatetimeIndex(P["dates"]); codes = [str(c) for c in P["codes"]]
liq = bt.liquidity_bucket(P)
size = np.select([liq == 0, liq <= 2, liq == 3], [-1, 0, 1], 2).astype(np.int8)
valid = ~P["limit"] & (liq > 0)
ro = np.full((T, N), np.nan); ro[:-1] = O[1:] / O[:-1] - 1
mkt = np.nanmean(np.where(np.abs(ro) < 1, ro, np.nan), axis=1); cum = np.concatenate([[0], np.cumsum(np.log1p(np.nan_to_num(mkt)))])


def cost(d, bars):
    return (20.0 if d > 0 else 20.0 + 0.031 * DAYS * np.maximum(bars, 1) / 365 * 1e4) / 1e4


sigs, fam_up, fam_dn = [], {}, {}
for fam in ("candle", "chart", "tech", "zone"):
    s = None
    if TF == "D" and fam in ("chart", "tech") and os.path.exists(f"cache/{fam}_sig.pkl"):
        s = pickle.load(open(f"cache/{fam}_sig.pkl", "rb"))
    if s is None:
        s = signals(P, fam, TF)
    ups = [np.asarray(S, bool) for d, S in s.values() if d > 0]; dns = [np.asarray(S, bool) for d, S in s.values() if d < 0]
    fam_up[fam] = np.any(ups, axis=0) if ups else np.zeros((T, N), bool); fam_dn[fam] = np.any(dns, axis=0) if dns else np.zeros((T, N), bool)
    for nm, (d, S) in s.items():
        sigs.append((zname(nm), fam, int(d), np.asarray(S, bool)))
    print(fam, len(s), flush=True)
NS = len(sigs)
last_t = np.full((NS, N), -1, np.int16)
rowsH, rowsO = [], []
f_adj = np.where(C > 0, Cr / C, np.nan)
for sid, (nm, fam, d, S) in enumerate(sigs):
    t, c = np.nonzero(S)
    if len(t):
        np.maximum.at(last_t[sid], c, t.astype(np.int16))
    ok = valid[t, c] & (t + 1 < T)
    t, c = t[ok], c[ok]
    e = t + 1; ok = O[e, c] > 0; t, c, e = t[ok], c[ok], e[ok]
    H = {"sid": np.full(len(t), sid, np.int16), "c": c.astype(np.int16), "t": t.astype(np.int16), "z": size[t, c]}
    for h in HOLDS:
        x = np.minimum(e + h, T - 1); okx = (e + h < T) & (O[x, c] > 0)
        r = np.where(okx, O[x, c] / O[e, c] - 1, np.nan)
        H[f"n{h}"] = (d * np.clip(r, -0.9, 3.0) - cost(d, h)).astype(np.float32)
        m = np.expm1(cum[x] - cum[e])
        H[f"x{h}"] = (d * (np.clip(r, -0.9, 3.0) - m) - cost(d, h)).astype(np.float32)
    rowsH.append(pd.DataFrame(H))
    # ② 反対のシグナルで出る
    opp = fam_dn[fam] if d > 0 else fam_up[fam]; out = []
    for cc in np.unique(c):
        ts = t[c == cc]; op = np.nonzero(opp[:, cc])[0]; last_exit = -1
        for tt in ts:
            if tt <= last_exit:
                continue
            ee = tt + 1; k = op[np.searchsorted(op, tt, side="right"):]
            xx = k[0] + 1 if len(k) else None
            if xx is not None and xx >= T:
                xx = None
            f = f_adj[:, cc]
            if xx is None:
                px = Cr[-1, cc]; pe = O[ee, cc] * f[ee]; r = px / pe - 1; w = T - 1 - ee; m = np.expm1(cum[T - 1] - cum[ee]); xo = -1; last_exit = T
            else:
                if not (O[xx, cc] > 0):
                    continue
                pe, px = O[ee, cc] * f[ee], O[xx, cc] * f[xx]; r = O[xx, cc] / O[ee, cc] - 1; w = xx - ee; m = np.expm1(cum[xx] - cum[ee]); xo = xx; last_exit = xx
            r = float(np.clip(r, -0.9, 3.0)); cst = cost(d, w)
            out.append((sid, cc, tt, ee, xo, w, pe, px, d * r - cst, d * (r - m) - cst, size[tt, cc]))
    rowsO.append(pd.DataFrame(out, columns=["sid", "c", "t", "e", "x", "w", "pe", "px", "net", "netex", "z"]))
    if sid % 10 == 0:
        print("sid", sid, NS, flush=True)
HH = pd.concat(rowsH, ignore_index=True); OO = pd.concat(rowsO, ignore_index=True)
OO = OO[np.isfinite(OO.px) & np.isfinite(OO.net) & np.isfinite(OO.pe)].copy(); OO["closed"] = OO.x >= 0
HH.to_pickle(f"cache/v2_{TF}_H.pkl"); OO.to_pickle(f"cache/v2_{TF}_O.pkl")
pickle.dump(dict(tf=TF, holds=HOLDS, sigs=[(s[0], s[1], s[2]) for s in sigs], dates=[d.strftime("%Y-%m-%d") for d in dates], codes=codes, size=size[-60:].tolist() if False else None,
                 last_t=last_t, days_per_bar=DAYS), open(f"cache/v2_{TF}_meta.pkl", "wb"))
print("done H", len(HH), "O", len(OO), "保有中", (~OO.closed).sum())
