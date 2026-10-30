"""抵抗帯・支持帯のリバウンドの強さを説明する特徴量つきの、イベント表を作る(cache/zone_char3.pkl)。zone_char.py の拡張版。
特徴量は、すべてシグナル日(t)の引けの時点で分かる値。
"""
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

import backtest as bt
import chart_matched as cm
from zones import SR_PARAMS, TF_PARAMS, detect_zones

P = bt.load_panel(); U = bt.load_margin(P)
dates, codes = P["dates"], P["codes"]
T, N = len(dates), len(codes)
frames = []
for f in sorted(glob.glob(os.path.join(bt.CACHE, "bars", "*.json.gz"))):
    rows = json.load(gzip.open(f, "rt", encoding="utf-8"))
    if rows:
        frames.append(pd.DataFrame(rows)[["Date", "Code", "MktCap"]])
mc = pd.concat(frames).pivot(index="Date", columns="Code", values="MktCap").sort_index()
MCAP = pd.to_numeric(mc.stack(), errors="coerce").unstack().reindex(index=[d.strftime("%Y-%m-%d") for d in dates], columns=codes).to_numpy(dtype=float)
O, H, L, C, Va, Vo = (P[k] for k in ("O", "H", "L", "C", "Va", "Vo"))
roll = lambda a, w, f="median", mp=None: getattr(pd.DataFrame(a).rolling(w, min_periods=mp or w // 2), f)().to_numpy()
turn60 = roll(Va / (MCAP * 1e6), 60); turn250 = roll(Va / (MCAP * 1e6), 250, mp=120)
ret = np.full_like(C, np.nan); ret[1:] = C[1:] / C[:-1] - 1
vol60 = roll(ret, 60, "std"); vol20 = roll(ret, 20, "std")
mret = np.nanmean(np.clip(ret, -0.5, 0.5), axis=1); midx = np.cumprod(1 + np.nan_to_num(mret))
feat = {
    "turn60": turn60, "turn250": turn250, "turn_ratio": turn60 / turn250,
    "mcap": MCAP, "va": roll(Va, 60), "vol60": vol60, "vol_ratio": vol20 / vol60, "price": P["Craw"],
    "ret20": C / pd.DataFrame(C).shift(20).to_numpy() - 1, "ret60": C / pd.DataFrame(C).shift(60).to_numpy() - 1, "ret120": C / pd.DataFrame(C).shift(120).to_numpy() - 1,
    "dist_hi": C / pd.DataFrame(H).rolling(250, min_periods=120).max().to_numpy() - 1, "dist_lo": C / pd.DataFrame(L).rolling(250, min_periods=120).min().to_numpy() - 1,
    "vol_surge": roll(Vo, 5, "mean") / roll(Vo, 60, "mean"), "gap": O / np.vstack([np.full((1, N), np.nan), C[:-1]]) - 1, "day_ret": ret, "day_range": (H - L) / C,
    "mkt_ret20": np.repeat((midx / np.roll(midx, 20) - 1)[:, None], N, 1), "mkt_vol20": np.repeat(pd.Series(mret).rolling(20).std().to_numpy()[:, None], N, 1),
}
# 上場経過年数(長期データの最初の足から。2000年より前の上場は、2000年を起点とする)
lz = np.load(os.path.join(bt.CACHE, "long", "W.npz"), allow_pickle=True); lc = list(lz["codes"]); ldates = pd.DatetimeIndex(lz["dates"]); Cw = lz["C"]
first = {c: ldates[np.argmax(~np.isnan(Cw[:, j]))] for j, c in enumerate(lc) if (~np.isnan(Cw[:, j])).any()}
firstd = np.array([first.get(c, pd.Timestamp("2026-10-01")) for c in codes], dtype="datetime64[ns]")
age = ((dates.to_numpy()[:, None] - firstd[None, :]) / np.timedelta64(1, "D") / 365.25)
feat["age"] = age

meta = {k: np.full((T, N), np.nan) for k in ("str_r", "age_r", "str_s", "age_s")}
sigs = detect_zones(H, L, C, P["Vo"], *TF_PARAMS["D"], sr=SR_PARAMS["D"], meta=meta)
names = {"抵抗帯に到達(跳ね返される)": -1, "支持帯に到達(跳ね返す)": +1}
liq = bt.liquidity_bucket(P); ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool); nolim = ~P["limit"]
prev = np.vstack([np.full((1, N), np.nan), C[:-1]]); r0 = C / prev - 1
adj = {}
for h in (1, 5, 10):
    fr = bt.forward_returns(P, [h])[h]
    r = np.where(ok, fr, np.nan); x = r - bt.baseline(r, liq); xm = np.full_like(x, np.nan)
    for size, lo, hi in cm.SIZES:
        grp = (liq >= lo) & (liq <= hi) & nolim & ~np.isnan(x) & ~np.isnan(r0)
        edges = np.unique(np.percentile(r0[grp], cm.QS)); bins = np.digitize(r0, edges[1:-1])
        base = np.full(len(edges) + 1, np.nan)
        for b in np.unique(bins[grp]):
            base[b] = x[grp & (bins == b)].mean()
        xm = np.where(grp, x - base[bins], xm)
    adj[h] = xm
out = []
for nm, d in names.items():
    t, c = np.nonzero(sigs[nm][1] & ~np.isnan(adj[5]))
    z = dict(sig=nm, d=d, t=t, c=c, margin=U[t, c], mkt=P["mkt"][c], liq=liq[t, c])
    for h in (1, 5, 10):
        z[f"adj{h}"] = d * adj[h][t, c]
    for k, a in feat.items():
        z[k] = a[t, c]
    key = ("str_r", "age_r") if d < 0 else ("str_s", "age_s")
    z["zstr"] = meta[key[0]][t, c]; z["zage"] = meta[key[1]][t, c]
    out.append(pd.DataFrame(z))
df = pd.concat(out, ignore_index=True)
master = P["master"].loc[codes]
df["scale"] = master["ScaleCat"].map({"TOPIX Core30": 5, "TOPIX Large70": 4, "TOPIX Mid400": 3, "TOPIX Small 1": 2, "TOPIX Small 2": 1}).fillna(0).to_numpy()[df.c.to_numpy()]
df["sector"] = master["S33Nm"].to_numpy()[df.c.to_numpy()]
df["code"] = np.array(codes)[df.c.to_numpy()]
df.to_pickle(os.path.join(bt.CACHE, "zone_char3.pkl"))
print(len(df), df.groupby("sig")[["adj1", "adj5", "adj10"]].mean().round(5))
print(df[["turn60", "turn250", "zstr", "zage", "age", "dist_hi"]].describe().round(3))
