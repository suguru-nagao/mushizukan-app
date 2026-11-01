"""「銘柄ごとに、効きやすいシグナル・出やすいシグナルがあるか」を、前半→後半で確かめる(週足、4週保有、コスト後の損益)。
銘柄×シグナルごとの平均を、前半と後半で出し、次の3つの補正を引いて残った部分(その銘柄だけの癖)が、後半にも残るかを、相関で見る。
  生の平均 → (シグナルの全銘柄平均を引く) → (さらに、その銘柄の全シグナル平均=銘柄の上がりやすさ、を引く)
後半の自身の平均を当てる力を、自身の前半の平均 / 仲間の前半の平均 / 同規模(全銘柄)の前半の平均 で比べる。
"""
import numpy as np, pandas as pd, pickle
H = 4
meta = pickle.load(open("cache/v2_W_meta.pkl", "rb")); HH = pd.read_pickle("cache/v2_W_H.pkl")
NS, NC, T = len(meta["sigs"]), len(meta["codes"]), len(meta["dates"]); half = T // 2
z = np.load("cache/peers_w.npz", allow_pickle=True); nn, vd, kp, zmode = z["nn"].astype(int), z["valid"], z["keep"], z["zmode"]
col = f"n{H}"; HH = HH[np.isfinite(HH[col])]
def mats(df):
    g = df.groupby(["sid", "c"])[col]
    n = np.zeros((NS, NC)); s = np.zeros((NS, NC)); w = np.zeros((NS, NC))
    st = pd.DataFrame(dict(n=g.size(), s=g.sum(), w=g.apply(lambda x: (x > 0).sum()))); i = st.index.to_frame(index=False)
    n[i.sid, i.c] = st.n; s[i.sid, i.c] = st.s; w[i.sid, i.c] = st.w
    return n, s, w
n1, s1, w1 = mats(HH[HH.t < half]); n2, s2, w2 = mats(HH[HH.t >= half])
print("前半", n1.sum().astype(int), "後半", n2.sum().astype(int))
def peerpool(M):
    out = np.zeros_like(M); msk = vd & kp[nn]
    for i in range(NS):
        out[i] = (M[i][nn] * msk).sum(1)
    return out
pn1, ps1, pw1 = peerpool(n1), peerpool(s1), peerpool(w1)
MIN = 10
ok = (n1 >= MIN) & (n2 >= MIN) & (pn1 >= 100) & kp[None, :]
def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))
w = np.minimum(n1, n2)[ok]
own1 = (s1 / np.maximum(n1, 1))[ok] * 1e4; own2 = (s2 / np.maximum(n2, 1))[ok] * 1e4
peer1 = (ps1 / np.maximum(pn1, 1))[ok] * 1e4
# シグナルの全銘柄平均(前半) と、銘柄の全シグナル平均(前半)
sig1 = np.repeat((s1.sum(1) / np.maximum(n1.sum(1), 1))[:, None], NC, 1)[ok] * 1e4
stk1 = np.repeat((s1.sum(0) / np.maximum(n1.sum(0), 1))[None, :], NS, 0)[ok] * 1e4
stk2 = np.repeat((s2.sum(0) / np.maximum(n2.sum(0), 1))[None, :], NS, 0)[ok] * 1e4
sig2 = np.repeat((s2.sum(1) / np.maximum(n2.sum(1), 1))[:, None], NC, 1)[ok] * 1e4
gm1, gm2 = s1.sum() / n1.sum() * 1e4, s2.sum() / n2.sum() * 1e4
print(f"対象のセル(銘柄×シグナル) {ok.sum()}  銘柄 {len(np.unique(np.nonzero(ok)[1]))}  前半の全体平均 {gm1:.1f}bp 後半 {gm2:.1f}bp")
res1 = own1 - sig1 - stk1 + gm1; res2 = own2 - sig2 - stk2 + gm2          # 銘柄×シグナルの癖(交互作用)
pr1 = peer1 - sig1 - stk1 + gm1                                          # 仲間の癖
print("\n【効きやすさ】前半 → 後半 の相関(件数で重み付け)")
print(f"  生の平均どうし                         {wcorr(own1, own2, w):+.3f}")
print(f"  シグナルの平均を引いたもの                 {wcorr(own1 - sig1, own2 - sig2, w):+.3f}")
print(f"  さらに銘柄の上がりやすさも引いた(=銘柄×シグナルの癖)  {wcorr(res1, res2, w):+.3f}")
print(f"  仲間の癖(前半) → 自身の癖(後半)          {wcorr(pr1, res2, w):+.3f}")
print("【後半の自身の平均を当てる力(相関)】")
for lab, x in (("自身の前半の平均", own1), ("仲間の前半の平均", peer1), ("シグナルの全銘柄平均(前半)", sig1), ("銘柄の上がりやすさ(前半)", stk1), ("シグナル平均+銘柄の上がりやすさ", sig1 + stk1 - gm1)):
    print(f"  {lab:<28}{wcorr(x, own2, w):+.3f}")
# 勝率版
wr1 = (w1 / np.maximum(n1, 1))[ok] * 100; wr2 = (w2 / np.maximum(n2, 1))[ok] * 100; pwr1 = (pw1 / np.maximum(pn1, 1))[ok] * 100
sg1 = np.repeat((w1.sum(1) / np.maximum(n1.sum(1), 1))[:, None], NC, 1)[ok] * 100; sg2 = np.repeat((w2.sum(1) / np.maximum(n2.sum(1), 1))[:, None], NC, 1)[ok] * 100
sk1 = np.repeat((w1.sum(0) / np.maximum(n1.sum(0), 1))[None, :], NS, 0)[ok] * 100; sk2 = np.repeat((w2.sum(0) / np.maximum(n2.sum(0), 1))[None, :], NS, 0)[ok] * 100
g1, g2 = w1.sum() / n1.sum() * 100, w2.sum() / n2.sum() * 100
print("\n【勝率】前半 → 後半 の相関")
print(f"  生の勝率どうし {wcorr(wr1, wr2, w):+.3f} / シグナル平均を引いた {wcorr(wr1 - sg1, wr2 - sg2, w):+.3f} / さらに銘柄の勝ちやすさを引いた {wcorr(wr1 - sg1 - sk1 + g1, wr2 - sg2 - sk2 + g2, w):+.3f}")
print(f"  後半の自身の勝率を当てる力: 自身の前半 {wcorr(wr1, wr2, w):+.3f} / 仲間の前半 {wcorr(pwr1, wr2, w):+.3f} / シグナル平均 {wcorr(sg1, wr2, w):+.3f} / 銘柄の勝ちやすさ {wcorr(sk1, wr2, w):+.3f}")
# 出やすさ: 銘柄×シグナルの発生率(1000バーあたり)
bars1 = np.array([(half - 0)] * NC); bars2 = np.array([(T - half)] * NC)
r1 = np.log((n1 + 0.5) / half * 1000); r2 = np.log((n2 + 0.5) / (T - half) * 1000)
okf = kp[None, :] & np.ones((NS, 1), bool); okf &= (n1 + n2) >= 6
wf = np.sqrt(n1 + n2)[okf]
m1 = r1[okf]; m2 = r2[okf]
sf1 = np.repeat(r1.mean(1)[:, None], NC, 1)[okf]; sf2 = np.repeat(r2.mean(1)[:, None], NC, 1)[okf]
kf1 = np.repeat(r1.mean(0)[None, :], NS, 0)[okf]; kf2 = np.repeat(r2.mean(0)[None, :], NS, 0)[okf]
print("\n【出やすさ】(銘柄×シグナルの発生率の、前半→後半の相関)")
print(f"  生 {wcorr(m1, m2, wf):+.3f} / シグナル平均を引いた {wcorr(m1 - sf1, m2 - sf2, wf):+.3f} / さらに銘柄の出やすさ(全シグナルの平均)も引いた {wcorr(m1 - sf1 - kf1, m2 - sf2 - kf2, wf):+.3f}")

# 追加: 同規模(シグナル×規模)の前半の平均、件数が多い場合の自身の予測力
sz = np.tile(zmode[None, :], (NS, 1))
def grp(M, N_):
    out = np.zeros((NS, NC))
    for zz in range(3):
        m = (sz == zz) & kp[None, :]
        out = np.where(sz == zz, (M * m).sum(1, keepdims=True) / np.maximum((N_ * m).sum(1, keepdims=True), 1), out)
    return out
sg_size1 = grp(s1, n1)[ok] * 1e4
print("\n【追加】同規模(シグナル×規模)の前半の平均 → 後半の自身の平均:", f"{wcorr(sg_size1, own2, w):+.3f}", "(シグナルの全銘柄平均は", f"{wcorr(sig1, own2, w):+.3f})")
for lo in (10, 30, 60):
    k = (n1[ok] >= lo)
    if k.sum() > 200:
        print(f"  自身の前半の件数 {lo} 以上({k.sum()}セル): 自身 {wcorr(own1[k], own2[k], w[k]):+.3f} / 仲間 {wcorr(peer1[k], own2[k], w[k]):+.3f} / シグナル平均 {wcorr(sig1[k], own2[k], w[k]):+.3f} / 同規模 {wcorr(sg_size1[k], own2[k], w[k]):+.3f} / 自身の癖(交互作用) {wcorr(res1[k], res2[k], w[k]):+.3f}")
