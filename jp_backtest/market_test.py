"""市場区分(プライム/スタンダード/グロース)別の成績と、「市場」を基準にした時の当たり方(週足・4週保有・コスト後)。
市場区分は、現在の区分(長期データの銘柄コードに当てはめる。区分の変遷は考慮していない)。
A: 種類×売買×市場の、勝率・平均・市場超過(①4週保有、②反対のシグナルで出る)。シグナルごとの平均が、市場の間で似ているか(相関)。
B: 基準の比較に「市場」「市場×規模」を加える(reference_group_test.py と同じ評価、先読みなし)。
"""
import json, pickle
import numpy as np, pandas as pd
H = 4
meta = pickle.load(open("cache/v2_W_meta.pkl", "rb")); HH = pd.read_pickle("cache/v2_W_H.pkl"); OO = pd.read_pickle("cache/v2_W_O.pkl")
codes = meta["codes"]; NS, NC = len(meta["sigs"]), len(codes); dates = pd.to_datetime(meta["dates"]); T = len(dates); yr = dates.year.to_numpy()
z = np.load("cache/peers_w.npz", allow_pickle=True); kp = z["keep"]; zmode = z["zmode"]
st = {x[4]: x[2] for x in json.load(open("report_data/meta.json"))["stocks"]}
mk = np.array([st.get(c, -1) for c in codes]); MN = ["プライム", "スタンダード", "グロース"]
print("市場の銘柄数", {MN[i]: int((mk == i).sum()) for i in range(3)}, "不明", int((mk < 0).sum()))
fam = np.array([s[1] for s in meta["sigs"]]); dr = np.array([s[2] for s in meta["sigs"]]); FAM = {"candle": "ローソク足", "chart": "チャート", "tech": "テクニカル", "zone": "抵抗帯"}
col = f"n{H}"; xc = f"x{H}"
A = HH[np.isfinite(HH[col]) & (mk[HH.c.to_numpy()] >= 0)].copy(); A["m"] = mk[A.c.to_numpy()]; A["f"] = fam[A.sid.to_numpy()]; A["d"] = dr[A.sid.to_numpy()]
B = OO[OO.closed & (mk[OO.c.to_numpy()] >= 0)].copy(); B["m"] = mk[B.c.to_numpy()]; B["f"] = fam[B.sid.to_numpy()]; B["d"] = dr[B.sid.to_numpy()]
rows = []
for nm, df, c, xcol in (("①4週保有", A, col, xc), ("②反対のシグナルで出る", B, "net", "netex")):
    for (f, d, m), g in df.groupby(["f", "d", "m"]):
        rows.append(dict(見方=nm, 種類=FAM[f], 売買="買い" if d > 0 else "売り", 市場=MN[m], 件数=len(g), 勝率=(g[c] > 0).mean() * 100, 平均bp=g[c].mean() * 1e4, 市場超過bp=g[xcol].mean() * 1e4, 保有週=g.w.median() if "w" in g else H))
    for (d, m), g in df.groupby(["d", "m"]):
        rows.append(dict(見方=nm, 種類="全体", 売買="買い" if d > 0 else "売り", 市場=MN[m], 件数=len(g), 勝率=(g[c] > 0).mean() * 100, 平均bp=g[c].mean() * 1e4, 市場超過bp=g[xcol].mean() * 1e4, 保有週=g.w.median() if "w" in g else H))
R = pd.DataFrame(rows).round(1); R.to_csv("results_market_split.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
for nm in ("①4週保有", "②反対のシグナルで出る"):
    print("\n==", nm); print(R[(R.見方 == nm) & (R.種類 == "全体")][["売買", "市場", "件数", "勝率", "平均bp", "市場超過bp", "保有週"]].to_string(index=False))
# シグナルごとの平均が、市場の間で似ているか
print("\n【シグナルごとの平均(4週・コスト後)の、市場どうしの相関(92シグナル)】")
sm = A.groupby(["sid", "m"])[col].mean().unstack() * 1e4
print(sm.corr().round(3).to_string())
print("【シグナルごとの勝率の相関】")
wm = A.assign(w=A[col] > 0).groupby(["sid", "m"]).w.mean().unstack() * 100
print(wm.corr().round(3).to_string())
# B: 基準の比較
col2 = col; HH2 = HH[np.isfinite(HH[col2])].copy(); HH2["y"] = yr[HH2.t.to_numpy()]
YEARS = list(range(2000, 2027)); yi = {y: i for i, y in enumerate(YEARS)}
Ny = np.zeros((len(YEARS), NS, NC), np.float32); Sy = np.zeros_like(Ny)
g = HH2.groupby(["y", "sid", "c"])[col2].agg(["size", "sum"]).reset_index(); Ny[g.y.map(yi), g.sid, g.c] = g["size"]; Sy[g.y.map(yi), g.sid, g.c] = g["sum"]
def group_pred(N, S, labels, minn=30):
    out = np.full((NS, NC), np.nan)
    for q in np.unique(labels[labels >= 0]):
        m = (labels == q) & kp; n = N[:, m].sum(1); v = S[:, m].sum(1)
        out[:, labels == q] = np.where(n >= minn, v / np.maximum(n, 1), np.nan)[:, None]
    return out
rows = {}
for Y in range(2008, 2027):
    j = yi[Y]; Na, Sa = Ny[j], Sy[j]; ok = (Na >= 3) & kp[None, :] & (mk >= 0)[None, :]
    act = (Sa / np.maximum(Na, 1))[ok] * 1e4; w = Na[ok]
    for pk, pl in {"全期間": None, "5年": 5}.items():
        lo = 0 if pl is None else max(0, j - pl); N = Ny[lo:j].sum(0); S = Sy[lo:j].sum(0)
        preds = {"全銘柄": (S[:, mk >= 0].sum(1) / np.maximum(N[:, mk >= 0].sum(1), 1))[:, None] * np.ones((1, NC)), "同規模": group_pred(N, S, zmode), "市場": group_pred(N, S, mk), "市場×規模": group_pred(N, S, np.where(mk >= 0, mk * 10 + zmode, -1)), "自身": np.where(N >= 5, S / np.maximum(N, 1), np.nan)}
        for bk, P in preds.items():
            p = P[ok] * 1e4; good = np.isfinite(p); rows.setdefault((bk, pk), []).append((p[good], act[good], w[good]))
out = []
for (bk, pk), L in rows.items():
    num = d1 = d2 = 0.0
    for p, a, w in L:
        if len(p) < 50: continue
        pc = p - np.average(p, weights=w); ac = a - np.average(a, weights=w); num += (w * pc * ac).sum(); d1 += (w * pc * pc).sum(); d2 += (w * ac * ac).sum()
    out.append(dict(基準=bk, 見る期間=pk, 相関=num / np.sqrt(d1 * d2)))
RB = pd.DataFrame(out).round(4); RB.to_csv("results_market_reference.csv", index=False, encoding="utf-8-sig")
print("\n【基準の比較(相関)】"); print(RB.pivot(index="基準", columns="見る期間", values="相関").to_string())
