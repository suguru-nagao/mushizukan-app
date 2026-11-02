"""銘柄別シグナルの「基準」(全銘柄 / 同規模 / 値動きが似た銘柄群)と、「何を見る期間」を、先読みなしで比べる(週足、4週保有、コスト後の損益)。
各テスト年 Y(2008〜2026)について、Y より前のデータだけで、銘柄×シグナルの予測値を作り、Y 年の実際の平均(件数3以上のセル)との相関を見る。
  予測値 = 基準の銘柄群の、同じシグナルの平均。
  基準: 全銘柄 / 同規模 / 値動きが似た同規模の上位200銘柄(相関を測る期間: 全期間・直近104週・52週・26週。Y の直前までで測る)/ 自身
  成績を見る期間: Y の前の 全期間 / 5年 / 2年 / 1年
評価は、年ごとに平均を引いた相関(年全体の水準の違いを除いて、セルの並びを当てる力)を、件数で重み付けして、全テスト年を通して出す。
"""
import numpy as np, pandas as pd, pickle, time
H = 4; K = 200
meta = pickle.load(open("cache/v2_W_meta.pkl", "rb")); HH = pd.read_pickle("cache/v2_W_H.pkl")
NS, NC = len(meta["sigs"]), len(meta["codes"]); dates = pd.to_datetime(meta["dates"]); T = len(dates); yr = dates.year.to_numpy()
z = np.load("cache/peers_w.npz", allow_pickle=True); kp = z["keep"]; zmode = z["zmode"]
W = np.load("cache/long/W.npz", allow_pickle=True); C = W["C"].astype(float)
ret = np.full((T, NC), np.nan); ret[1:] = C[1:] / C[:-1] - 1; ret = np.clip(ret, -0.3, 0.3)
res = np.full_like(ret, np.nan)
for zz in range(3):
    m = (zmode == zz); sub = ret[:, m]; res[:, m] = sub - np.nanmean(sub, axis=1, keepdims=True)
col = f"n{H}"; HH = HH[np.isfinite(HH[col])]
HH["y"] = yr[HH.t.to_numpy()]
YEARS = list(range(2000, 2027)); yi = {y: i for i, y in enumerate(YEARS)}
Ny = np.zeros((len(YEARS), NS, NC), np.float32); Sy = np.zeros_like(Ny)
g = HH.groupby(["y", "sid", "c"])[col].agg(["size", "sum"]).reset_index()
Ny[g.y.map(yi), g.sid, g.c] = g["size"]; Sy[g.y.map(yi), g.sid, g.c] = g["sum"]
print("集計済み", flush=True)

def peers_at(end, L):
    """end(週番号)の直前までの L 週(None=全期間)の残差リターンの相関で、同規模の上位Kを返す。"""
    lo = 1 if L is None else max(1, end - L)
    x = res[lo:end]; ok = ~np.isnan(x); cnt = ok.sum(0); keep = (cnt >= min(100, (end - lo) * 0.6)) & kp
    x0 = np.where(ok, x, 0.0); sd = np.sqrt((x0 ** 2).sum(0) / np.maximum(cnt, 1)); x0 = np.clip(x0 / np.where(sd > 0, sd, 1), -4, 4)
    num = x0.T @ x0; ov = ok.T.astype(np.float32) @ ok.astype(np.float32); c = num / np.maximum(ov, 1)
    c[ov < min(60, (end - lo) * 0.5)] = -9; c[~keep, :] = -9; c[:, ~keep] = -9; c[zmode[:, None] != zmode[None, :]] = -9; np.fill_diagonal(c, -9)
    nn = np.argsort(-c, axis=1)[:, :K]; vd = np.take_along_axis(c, nn, 1) > -9
    return nn, vd

def pool(N, S, nn, vd):
    PN = np.zeros((NS, NC), np.float32); PS = np.zeros_like(PN)
    for i in range(NS):
        PN[i] = (N[i][nn] * vd).sum(1); PS[i] = (S[i][nn] * vd).sum(1)
    return PN, PS

def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    d = np.sqrt((w * a * a).sum() * (w * b * b).sum()); return (w * a * b).sum() / d if d > 0 else np.nan

PERF = {"全期間": None, "5年": 5, "2年": 2, "1年": 1}
CORR = {"全期間": None, "104週": 104, "52週": 52, "26週": 26}
rows = {}   # (基準, 見る期間) -> list of (pred, actual, w, year, n)
t0 = time.time()
for Y in range(2008, 2027):
    j = yi[Y]; end = int(np.searchsorted(yr, Y))        # Y 年の最初の週
    Na, Sa = Ny[j], Sy[j]; ok = (Na >= 3) & kp[None, :]
    act = (Sa / np.maximum(Na, 1))[ok] * 1e4; w = Na[ok]
    PE = {k: peers_at(end, L) for k, L in CORR.items()}
    for pk, pl in PERF.items():
        lo = 0 if pl is None else max(0, j - pl); N = Ny[lo:j].sum(0); S = Sy[lo:j].sum(0)
        gm = S.sum() / max(N.sum(), 1)
        sigm = (S.sum(1) / np.maximum(N.sum(1), 1))[:, None] * np.ones((1, NC))
        sizem = np.zeros((NS, NC))
        for zz in range(3):
            m = (zmode == zz) & kp; sizem[:, zmode == zz] = ((S[:, m].sum(1)) / np.maximum(N[:, m].sum(1), 1))[:, None]
        preds = {"全銘柄": sigm, "同規模": sizem, "自身": np.where(N >= 5, S / np.maximum(N, 1), np.nan)}
        for ck, (nn, vd) in PE.items():
            PN, PS = pool(N, S, nn, vd); preds["仲間(相関" + ck + ")"] = np.where(PN >= 30, PS / np.maximum(PN, 1), np.nan)
        for bk, P in preds.items():
            p = P[ok] * 1e4; good = np.isfinite(p)
            rows.setdefault((bk, pk), []).append((p[good], act[good], w[good], Y))
    print(Y, f"{time.time() - t0:.0f}s", flush=True)
out = []
for (bk, pk), L in rows.items():
    num = den1 = den2 = 0.0; ycs = []
    for p, a, w, Y in L:
        if len(p) < 50: continue
        pc = p - np.average(p, weights=w); ac = a - np.average(a, weights=w)
        num += (w * pc * ac).sum(); den1 += (w * pc * pc).sum(); den2 += (w * ac * ac).sum(); ycs.append(wcorr(p, a, w))
    out.append(dict(基準=bk, 見る期間=pk, 相関=num / np.sqrt(den1 * den2), 年ごとの相関の平均=np.nanmean(ycs), 年ごとの相関が正の年数=int(np.sum(np.array(ycs) > 0)), 年数=len(ycs)))
R = pd.DataFrame(out).round(3); R.to_csv("results_reference_group.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200); print(R.pivot(index="基準", columns="見る期間", values="相関")[["全期間", "5年", "2年", "1年"]].to_string()); print(R.pivot(index="基準", columns="見る期間", values="年ごとの相関が正の年数")[["全期間", "5年", "2年", "1年"]].to_string())
