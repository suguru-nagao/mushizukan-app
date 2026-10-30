"""週足のパラボリックSARの転換でエントリー・イグジットした時の勝率(長期データ 2000〜2026、Yahoo週足)。
買い: 買い転換の次の週の始値で入り、次の売り転換の次の週の始値で出る。売り(ショート): その逆。
勝率 = 1回の取引(エントリーからイグジットまで)が、コスト前/後でプラスだった割合。市場(全銘柄の等ウェイト)の同期間リターンとの超過も見る。
コスト: 買い=往復20bp(現物)、売り=20bp+貸株料・配当調整金(年3.1%×保有日数/365)
"""
import numpy as np, pandas as pd
from tech_signals import sar_flips
from backtest import cluster_t

z = np.load("cache/long/W.npz", allow_pickle=True)
O, H, L, C, Va = (z[k].astype(float) for k in ("O", "H", "L", "C", "Va")); dates = pd.to_datetime(z["dates"]); T, N = O.shape
up, dn = sar_flips(H, L, C)
liq = pd.DataFrame(Va).rolling(13, min_periods=8).median().to_numpy()
# 規模(同週の売買代金の分位 → 小型1-2 / 中型3 / 大型4-5)
rk = pd.DataFrame(liq).rank(axis=1, pct=True).to_numpy(); q = np.ceil(rk * 5)
ro = np.full((T, N), np.nan); ro[:-1] = O[1:] / O[:-1] - 1                 # 始値→次の始値の週リターン
mkt = np.nanmean(np.where(np.abs(ro) < 1, ro, np.nan), axis=1)             # 市場(等ウェイト)の週リターン
cum = np.concatenate([[0], np.cumsum(np.log1p(np.nan_to_num(mkt)))])      # 市場の累積対数リターン(区間 [a,b) の始値→始値)
rows = []
for c in range(N):
    iu = np.nonzero(up[:, c])[0]; idn = np.nonzero(dn[:, c])[0]
    for side, a, b in ((1, iu, idn), (-1, idn, iu)):
        for e0 in a:
            nxt = b[b > e0]
            if len(nxt) == 0:
                continue
            e, x = e0 + 1, nxt[0] + 1
            if x >= T or e >= T or not (O[e, c] > 0 and O[x, c] > 0):
                continue
            r = O[x, c] / O[e, c] - 1
            rows.append((c, side, e, x, x - e, r, np.expm1(cum[x] - cum[e]), q[e0, c]))
D = pd.DataFrame(rows, columns=["c", "side", "e", "x", "w", "r", "m", "q"])
D = D[np.isfinite(D.r)]
D["gross"] = D.side * D.r; D["ex"] = D.side * (D.r - D.m)
D["cost"] = np.where(D.side == 1, 20, 20 + 0.031 * 7 * D.w / 365 * 1e4) / 1e4
D["net"] = D.gross - D.cost; D["netex"] = D.ex - D.cost
D["size"] = pd.cut(D.q, [0, 2, 3, 5], labels=["小型", "中型", "大型"]); D["year"] = dates[D.e.clip(upper=T - 1)].year
D.to_pickle("cache/sar_trades.pkl")


def summ(g, lab):
    return dict(区分=lab, 取引数=len(g), 保有週_中央値=g.w.median(), 勝率_コスト前=(g.gross > 0).mean() * 100, 勝率_コスト後=(g.net > 0).mean() * 100,
                勝率_市場超過=(g.ex > 0).mean() * 100, 平均_bp=g.gross.clip(-1, 3).mean() * 1e4, 中央値_bp=g.gross.median() * 1e4,
                平均コスト後_bp=g.net.clip(-1, 3).mean() * 1e4, 平均超過コスト後_bp=g.netex.clip(-1, 3).mean() * 1e4,
                t_超過コスト後=cluster_t(g.netex.clip(-1, 3).to_numpy(), g.e.to_numpy(), 5))
out = []
for s, nm in ((1, "買い(買い転換→売り転換)"), (-1, "売り(売り転換→買い転換)")):
    g = D[D.side == s]; out.append(summ(g, nm + " 全体"))
    for sz in ("小型", "中型", "大型"):
        out.append(summ(g[g["size"] == sz], nm + " " + sz))
    for a, b in ((2000, 2007), (2008, 2014), (2015, 2020), (2021, 2026)):
        out.append(summ(g[(g.year >= a) & (g.year <= b)], f"{nm} {a}-{b}"))
R = pd.DataFrame(out).round(1); R.to_csv("results_sar_weekly.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R.to_string(index=False))
