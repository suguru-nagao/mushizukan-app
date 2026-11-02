"""仮説「値動きの相関が高い銘柄は、同じようなシグナルが出て、勝率・損益も似る」を確かめる。週足・4週保有・コスト後。
前半(〜2013)の値動きの相関で、銘柄ごとに、相関が高い順の上位K銘柄(同規模)を決め、
 (1) 同じシグナルが、同じ週に出やすいか(同時発生)
 (2) 同じシグナルの成績(シグナル平均と銘柄の上がりやすさを引いた「銘柄×シグナルの癖」)が、後半に、仲間の癖と似るか
 を、Kを変えて見る(K = 5, 20, 50, 200, 800、比較: 同規模からランダムに200)。
"""
import numpy as np, pandas as pd, pickle
H = 4
meta = pickle.load(open("cache/v2_W_meta.pkl", "rb")); HH = pd.read_pickle("cache/v2_W_H.pkl")
NS, NC, T = len(meta["sigs"]), len(meta["codes"]), len(meta["dates"]); half = T // 2
z = np.load("cache/peers_w.npz", allow_pickle=True); kp = z["keep"]; zmode = z["zmode"]
W = np.load("cache/long/W.npz", allow_pickle=True); C = W["C"].astype(float)
ret = np.full((T, NC), np.nan); ret[1:] = C[1:] / C[:-1] - 1; ret = np.clip(ret, -0.3, 0.3)
res = np.full_like(ret, np.nan)
for zz in range(3):
    m = zmode == zz; res[:, m] = ret[:, m] - np.nanmean(ret[:, m], axis=1, keepdims=True)
x = res[1:half]; ok = ~np.isnan(x); cnt = ok.sum(0); keep = (cnt >= 150) & kp
x0 = np.where(ok, x, 0.0); sd = np.sqrt((x0 ** 2).sum(0) / np.maximum(cnt, 1)); x0 = np.clip(x0 / np.where(sd > 0, sd, 1), -4, 4)
corr = (x0.T @ x0) / np.maximum(ok.T.astype(np.float32) @ ok.astype(np.float32), 1)
corr[~keep, :] = -9; corr[:, ~keep] = -9; corr[zmode[:, None] != zmode[None, :]] = -9; np.fill_diagonal(corr, -9)
order = np.argsort(-corr, axis=1)
print("前半の相関(上位5/20/50/200/800の平均)", {k: round(float(np.mean(np.take_along_axis(corr, order[:, :k], 1)[keep][:, :]) if False else np.mean([corr[i, order[i, :k]].mean() for i in np.nonzero(keep)[0][::20]])), 3) for k in (5, 20, 50, 200, 800)})
col = f"n{H}"; HH = HH[np.isfinite(HH[col])]
def mats(df):
    g = df.groupby(["sid", "c"])[col]; n = np.zeros((NS, NC)); s = np.zeros((NS, NC)); st = pd.DataFrame(dict(n=g.size(), s=g.sum())); i = st.index.to_frame(index=False)
    n[i.sid, i.c] = st.n; s[i.sid, i.c] = st.s; return n, s
n1, s1 = mats(HH[HH.t < half]); n2, s2 = mats(HH[HH.t >= half])
# (1) 同時発生: 銘柄Aにシグナルが出た週に、Bにも同じシグナルが出る確率(相関上位 vs ランダム同規模)
TT = len(meta["dates"])
sig_week = HH[(HH.t >= half)][["sid", "c", "t"]].drop_duplicates()
co = {}
rng = np.random.default_rng(0)
cand = np.nonzero(keep)[0]; samp = rng.choice(cand, 150, replace=False)
sets = {}
for sid in range(0, NS, 6):
    d = sig_week[sig_week.sid == sid]; M = np.zeros((TT, NC), bool); M[d.t, d.c] = True; sets[sid] = M
def cooc(k):
    num = den = 0
    for sid, M in sets.items():
        for a in samp:
            ts = np.nonzero(M[:, a])[0]; ts = ts[ts >= half]
            if len(ts) < 5: continue
            pr = order[a, :k] if k > 0 else rng.choice(np.nonzero((zmode == zmode[a]) & keep)[0], 50, replace=False)
            num += M[ts][:, pr].sum(); den += len(ts) * len(pr)
    return num / max(den, 1)
print("\n(1) 同じ週に、同じシグナルが出る確率(銘柄Aに出た週に、仲間Bにも出る)")
for k in (5, 20, 50, 200):
    print(f"  相関上位{k}: {cooc(k) * 100:.2f}%")
print(f"  同規模のランダム50: {cooc(0) * 100:.2f}%")
# (2) 癖の相関
ok2 = (n1 >= 10) & (n2 >= 10) & keep[None, :]
def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w); return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))
w = np.minimum(n1, n2)[ok2]
own2 = (s2 / np.maximum(n2, 1))[ok2] * 1e4; sig2 = np.repeat((s2.sum(1) / np.maximum(n2.sum(1), 1))[:, None], NC, 1)[ok2] * 1e4; stk2 = np.repeat((s2.sum(0) / np.maximum(n2.sum(0), 1))[None, :], NS, 0)[ok2] * 1e4
res2 = own2 - sig2 - stk2 + s2.sum() / n2.sum() * 1e4
sig1 = np.repeat((s1.sum(1) / np.maximum(n1.sum(1), 1))[:, None], NC, 1); stk1 = np.repeat((s1.sum(0) / np.maximum(n1.sum(0), 1))[None, :], NS, 0); g1 = s1.sum() / n1.sum()
print("\n(2) 仲間の「銘柄×シグナルの癖」(前半) と、自身の癖(後半)の相関  /  仲間の前半の平均 → 自身の後半の平均 の相関")
def peer_stats(k):
    if k > 0:
        idx = order[:, :k]; vd = np.take_along_axis(corr, idx, 1) > -9
    else:
        idx = np.zeros((NC, 200), int); vd = np.ones((NC, 200), bool)
        for a in range(NC):
            pool = np.nonzero((zmode == zmode[a]) & keep)[0]
            idx[a] = rng.choice(pool, 200, replace=len(pool) < 200)
    PN = np.zeros((NS, NC)); PS = np.zeros((NS, NC))
    for i in range(NS):
        PN[i] = (n1[i][idx] * vd).sum(1); PS[i] = (s1[i][idx] * vd).sum(1)
    return PN, PS
for k in (5, 20, 50, 200, 800, 0):
    PN, PS = peer_stats(k); m = ok2 & (PN >= 30)
    ww = np.minimum(n1, n2)[m]; pm = (PS / np.maximum(PN, 1))[m] * 1e4
    o2 = (s2 / np.maximum(n2, 1))[m] * 1e4
    sg2 = np.repeat((s2.sum(1) / np.maximum(n2.sum(1), 1))[:, None], NC, 1)[m] * 1e4; sk2 = np.repeat((s2.sum(0) / np.maximum(n2.sum(0), 1))[None, :], NS, 0)[m] * 1e4
    r2 = o2 - sg2 - sk2 + s2.sum() / n2.sum() * 1e4
    pr1 = pm - (np.repeat((s1.sum(1) / np.maximum(n1.sum(1), 1))[:, None], NC, 1)[m] * 1e4) - (np.repeat((s1.sum(0) / np.maximum(n1.sum(0), 1))[None, :], NS, 0)[m] * 1e4) + g1 * 1e4
    print(f"  {'相関上位' + str(k) if k else '同規模ランダム200':<14} セル{m.sum():>7}  癖→癖 {wcorr(pr1, r2, ww):+.3f}   平均→平均 {wcorr(pm, o2, ww):+.3f}")
