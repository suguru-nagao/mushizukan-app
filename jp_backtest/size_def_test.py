"""「同規模」の定義を比べる: 売買代金の順位 / 時価総額の順位、3区分(40/20/40%) / 5区分、区分の決め方(銘柄ごとに、テスト年の直前までに最も多く属した区分。先読みなし)。
時価総額は、週足の長期データには無いので、「現在の発行済株式数(最新の時価総額÷終値) × その週の調整後終値」で近似する(増資・自社株買いは無視)。
reference_group_test.py と同じ評価(週足・4週保有・コスト後・年ごとに平均を引いた相関)。
"""
import glob, gzip, json, pickle
import numpy as np, pandas as pd
H = 4
meta = pickle.load(open("cache/v2_W_meta.pkl", "rb")); HH = pd.read_pickle("cache/v2_W_H.pkl")
codes = meta["codes"]; NS, NC = len(meta["sigs"]), len(codes); dates = pd.to_datetime(meta["dates"]); T = len(dates); yr = dates.year.to_numpy()
kp = np.load("cache/peers_w.npz", allow_pickle=True)["keep"]
W = np.load("cache/long/W.npz", allow_pickle=True); Va = W["Va"].astype(float); Cadj = W["C"].astype(float)
last = {}
for f in sorted(glob.glob("cache/bars/*.json.gz"))[-3:]:
    for r in json.load(gzip.open(f, "rt", encoding="utf-8")):
        if r.get("MktCap") and r.get("C"):
            last[r["Code"]] = r["MktCap"] * 1e6 / r["C"]
shares = np.array([last.get(c, np.nan) for c in codes])
mcap = Cadj * shares[None, :]
print("時価総額を作れた銘柄", int(np.isfinite(shares).sum()), "/", NC, flush=True)
vmed = pd.DataFrame(Va).rolling(4, min_periods=2).median().to_numpy()
def groups(x, edges):
    pct = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy(); g = np.full(x.shape, -1)
    ok = np.isfinite(pct)
    g[ok] = np.digitize(pct[ok], edges)
    return g
G = {"売買代金3区分": groups(vmed, [0.4, 0.6]), "時価総額3区分": groups(mcap, [0.4, 0.6]), "売買代金5区分": groups(vmed, [.2, .4, .6, .8]), "時価総額5区分": groups(mcap, [.2, .4, .6, .8])}
NG = {k: int(v.max()) + 1 for k, v in G.items()}
cum = {}
for k, g in G.items():
    c = np.zeros((T + 1, NC, NG[k]), np.float32)
    for q in range(NG[k]):
        c[1:, :, q] = np.cumsum(g == q, axis=0)
    cum[k] = c
col = f"n{H}"; HH = HH[np.isfinite(HH[col])]; HH["y"] = yr[HH.t.to_numpy()]
YEARS = list(range(2000, 2027)); yi = {y: i for i, y in enumerate(YEARS)}
Ny = np.zeros((len(YEARS), NS, NC), np.float32); Sy = np.zeros_like(Ny)
g = HH.groupby(["y", "sid", "c"])[col].agg(["size", "sum"]).reset_index()
Ny[g.y.map(yi), g.sid, g.c] = g["size"]; Sy[g.y.map(yi), g.sid, g.c] = g["sum"]
def group_pred(N, S, labels, minn=30):
    out = np.full((NS, NC), np.nan)
    for q in np.unique(labels[labels >= 0]):
        m = (labels == q) & kp; n = N[:, m].sum(1); v = S[:, m].sum(1)
        out[:, labels == q] = np.where(n >= minn, v / np.maximum(n, 1), np.nan)[:, None]
    return out
rows = {}
okm = np.isfinite(shares)
for Y in range(2008, 2027):
    j = yi[Y]; end = int(np.searchsorted(yr, Y))
    Na, Sa = Ny[j], Sy[j]; ok = (Na >= 3) & kp[None, :] & okm[None, :]
    act = (Sa / np.maximum(Na, 1))[ok] * 1e4; w = Na[ok]
    for pk, pl in {"全期間": None, "5年": 5}.items():
        lo = 0 if pl is None else max(0, j - pl); N = Ny[lo:j].sum(0); S = Sy[lo:j].sum(0)
        preds = {"全銘柄": (S.sum(1) / np.maximum(N.sum(1), 1))[:, None] * np.ones((1, NC))}
        for k in G:
            cnt = cum[k][end] - cum[k][max(0, end - 0)] if False else cum[k][end]       # 直前までの、各区分に属した週数
            lab = np.where(cnt.sum(1) >= 52, cnt.argmax(1), -1)
            preds[k + "(直前までの最頻)"] = group_pred(N, S, lab)
        for bk, P in preds.items():
            p = P[ok] * 1e4; good = np.isfinite(p)
            rows.setdefault((bk, pk), []).append((p[good], act[good], w[good]))
    print(Y, flush=True)
out = []
for (bk, pk), L in rows.items():
    num = d1 = d2 = 0.0; ys = []
    for p, a, w in L:
        if len(p) < 50: continue
        pc = p - np.average(p, weights=w); ac = a - np.average(a, weights=w)
        num += (w * pc * ac).sum(); d1 += (w * pc * pc).sum(); d2 += (w * ac * ac).sum(); ys.append((w * pc * ac).sum() / np.sqrt((w * pc * pc).sum() * (w * ac * ac).sum()))
    out.append(dict(基準=bk, 見る期間=pk, 相関=num / np.sqrt(d1 * d2), 正の年=int(np.sum(np.array(ys) > 0)), 年数=len(ys)))
R = pd.DataFrame(out).round(4); R.to_csv("results_size_def.csv", index=False, encoding="utf-8-sig")
print(R.pivot(index="基準", columns="見る期間", values="相関").to_string())
