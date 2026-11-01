"""週足の「入りと出」をシグナルで決める売買の成績(銘柄別シグナルタブ・export 用)。長期データ(Yahoo週足 2000〜2026)。
  SAR: 買い転換の次の週の始値で買い、売り転換の次の週の始値で売る(売りは逆)。
  ローソク足: 買いシグナル(例: ハンマー)の次の週の始値で買い、その後に「売りのローソク足(9種のどれか)」が出たら、その次の週の始値で売る(売りは逆)。
  まだ出ていない取引は「保有中」(最新終値で評価)。
勝率 = コスト後の損益がプラスだった取引の割合。損益 = 売買の値幅(円/100株)と、損益率(bp)。コスト: 買い 20bp、売り 20bp + 貸株料等(年3.1%×保有日数/365)。
出力: report_data/rt_{k}.json(銘柄コード → 集計と直近1年の取引)、rt_meta.json
"""
import json, os
import numpy as np, pandas as pd
from long_run import load
from mtf_run import signals
from build_report import zname

OUT = "report_data"; SHARD = 10
P, U = load("W"); O = P["O"].astype(float); C = P["C"].astype(float); Cr = P["Craw"].astype(float); Va = P["Va"].astype(float)
dates = P["dates"]; T, N = O.shape; codes = [str(c) for c in P["codes"]]
liq = pd.DataFrame(Va).rolling(13, min_periods=8).median().to_numpy(); q = np.ceil(pd.DataFrame(liq).rank(axis=1, pct=True).to_numpy() * 5)
size = np.select([q <= 2, q == 3, q >= 4], [0, 1, 2], -1)
valid = ~P["limit"] & ~np.isnan(q)
ro = np.full((T, N), np.nan); ro[:-1] = O[1:] / O[:-1] - 1
mkt = np.nanmean(np.where(np.abs(ro) < 1, ro, np.nan), axis=1); cum = np.concatenate([[0], np.cumsum(np.log1p(np.nan_to_num(mkt)))])
LASTY = T - 52

cs = signals(P, "candle", "W"); tech = signals(P, "tech", "W")
sigs = []   # (名前, 系統, +1/-1, 入りの配列, 反対側の配列)
cb = np.any([S for d, S in cs.values() if d > 0], axis=0); cslr = np.any([S for d, S in cs.values() if d < 0], axis=0)
for nm, (d, S) in cs.items():
    sigs.append((zname(nm), "candle", d, S, cslr if d > 0 else cb))
sar_up = tech["パラボリックSAR 買い転換"][1]; sar_dn = tech["パラボリックSAR 売り転換"][1]
sigs.append((zname("パラボリックSAR 買い転換"), "tech", 1, sar_up, sar_dn)); sigs.append((zname("パラボリックSAR 売り転換"), "tech", -1, sar_dn, sar_up))

rows = []
for sid, (nm, fam, d, S, OPP) in enumerate(sigs):
    for c in range(N):
        ts = np.nonzero(S[:, c] & valid[:, c])[0]
        if len(ts) == 0:
            continue
        op = np.nonzero(OPP[:, c])[0]; last_exit = -1
        for t in ts:
            e = t + 1
            if e >= T or t <= last_exit or not (O[e, c] > 0):      # 保有中の間に出た同じシグナルは、重複して数えない
                continue
            k = op[np.searchsorted(op, t, side="right"):]
            x = k[0] + 1 if len(k) else None
            if x is not None and x >= T:
                x = None
            if x is None:
                px, w, openp = Cr[-1, c], T - 1 - e, True              # 保有中: 最新終値(調整前)
                pe = O[e, c] * Cr[e, c] / C[e, c]
                r = px / pe - 1; m = np.expm1(cum[T - 1] - cum[e]); last_exit = T
            else:
                if not (O[x, c] > 0):
                    continue
                f = Cr[:, c] / C[:, c]; pe, px = O[e, c] * f[e], O[x, c] * f[x]; r = O[x, c] / O[e, c] - 1; w = x - e; m = np.expm1(cum[x] - cum[e]); openp = False; last_exit = x
            r = float(np.clip(r, -0.9, 3.0)); gross = d * r; cost = (20 if d > 0 else 20 + 0.031 * 7 * max(w, 1) / 365 * 1e4) / 1e4
            rows.append((sid, c, t, e, -1 if openp else x, w, pe, px, gross - cost, d * (r - m) - cost, size[t, c]))
D = pd.DataFrame(rows, columns=["sid", "c", "t", "e", "x", "w", "pe", "px", "net", "netex", "z"])
D = D[np.isfinite(D.px) & np.isfinite(D.net) & np.isfinite(D.pe)].copy()   # 上場廃止などで最新終値がない保有中を除く
D["closed"] = D.x >= 0; print("取引", len(D), "うち保有中", (~D.closed).sum(), flush=True)
D.to_pickle("cache/rt_trades.pkl")
cl = D[D.closed]
# 規模別の全銘柄平均(決済済みのみ)
zs = {}
for (sid, z), g in cl.groupby(["sid", "z"]):
    zs[(sid, int(z))] = [len(g), float((g.net > 0).mean() * 100), float(g.net.mean() * 1e4), float(g.netex.mean() * 1e4), float(g.w.median())]
out = {}
for (sid, c), g in D.groupby(["sid", "c"]):
    gc = g[g.closed]
    st = [int(sid), len(gc), None if len(gc) == 0 else round(float((gc.net > 0).mean() * 100), 1), None if len(gc) == 0 else round(float(gc.net.mean() * 1e4), 1),
          None if len(gc) == 0 else round(float(gc.net.median() * 1e4), 1), None if len(gc) == 0 else round(float(gc.netex.mean() * 1e4), 1), None if len(gc) == 0 else round(float(gc.w.median()), 1)]
    z = zs.get((sid, int(g.z.iloc[-1])))
    st += [None] * 4 if z is None else [round(z[1], 1), round(z[2], 1), round(z[3], 1), round(z[4], 1)]
    # 直近1年: 決済が直近52週、または保有中
    rec = g[(g.x >= LASTY) | (~g.closed)]
    tr = [[int(sid), int(r.e), int(r.x), round(float(r.pe), 1), round(float(r.px), 1), int(r.w), round(float(r.net * 1e4), 1)] for r in rec.itertuples()]
    o = out.setdefault(c, {"s": [], "tr": []}); o["s"].append(st); o["tr"] += tr
for k in range((N + 399) // 400):
    json.dump({codes[c]: out[c] for c in range(k * 400, min(N, (k + 1) * 400)) if c in out}, open(f"{OUT}/rt_{k}.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
json.dump(dict(sigs=[[s[0], s[1], s[2]] for s in sigs], dates=[d.strftime("%Y-%m-%d") for d in dates], codes=codes, shard=400,
               cols_s=["sid", "n", "win", "mean_bp", "median_bp", "mean_excess_bp", "median_weeks", "size_win", "size_mean_bp", "size_excess_bp", "size_median_weeks"], cols_tr=["sid", "entry_t", "exit_t(-1=保有中)", "entry_price", "exit_or_last_price", "weeks", "net_bp"]),
          open(f"{OUT}/rt_meta.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("done", len(out))
