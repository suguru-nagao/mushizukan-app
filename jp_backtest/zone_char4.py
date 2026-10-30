"""リバウンドの強さを説明する特徴の探索(価格・出来高・帯の特徴)。
1) 特徴ごとの五分位別の平均(5日保有、売り・買い合わせ)  2) 勾配ブースティングで、時間で分けた検証(前半60%で学習、後半40%で評価)と、順列重要度
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import backtest as bt

warnings.filterwarnings("ignore")
df = pd.read_pickle(bt.CACHE + "/zone_char3.pkl")
df["margin"] = df.margin.astype(float)
df["lturn60"] = np.log(df.turn60.clip(lower=1e-6)); df["lturn250"] = np.log(df.turn250.clip(lower=1e-6)); df["lmcap"] = np.log(df.mcap); df["lva"] = np.log(df.va)
df["lprice"] = np.log(df.price); df["lvol60"] = np.log(df.vol60.clip(lower=1e-4))
FEATS = {"lturn60": "回転率(60日)", "lturn250": "回転率(250日)", "turn_ratio": "回転率の変化(60日/250日)", "lmcap": "時価総額", "lva": "売買代金", "lvol60": "変動率(60日)", "vol_ratio": "変動率の変化(20日/60日)",
         "lprice": "株価", "ret20": "直近20日の騰落率", "ret60": "直近60日の騰落率", "ret120": "直近120日の騰落率", "dist_hi": "250日高値からの距離", "dist_lo": "250日安値からの距離",
         "vol_surge": "出来高の急増(5日/60日)", "gap": "当日の窓", "day_ret": "当日の騰落率", "day_range": "当日の値幅", "mkt_ret20": "市場の直近20日", "mkt_vol20": "市場の変動(20日)",
         "age": "上場経過年数", "scale": "TOPIX規模区分", "margin": "貸借銘柄", "zstr": "帯の強さ(平均の何倍か)", "zage": "帯の出来高の古さ(平均何日前か)"}
df = df.dropna(subset=["adj5"] + list(FEATS))
df["adj5c"] = df.adj5.clip(-0.15, 0.15)       # 外れ値の影響を抑える(±15%)
split = df.t.quantile(0.6)
rows = []
for nm, g in (("全体", df), ("抵抗帯(売り)", df[df.d < 0]), ("支持帯(買い)", df[df.d > 0])):
    tr, te = g[g.t < split], g[g.t >= split]
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=300, l2_regularization=5.0, random_state=0)
    X = list(FEATS)
    m.fit(tr[X], tr.adj5c)
    pr = m.predict(te[X])
    corr = np.corrcoef(pr, te.adj5c)[0, 1]
    q = pd.qcut(pr, 5, labels=False, duplicates="drop")
    top = te[q == q.max()]; bot = te[q == 0]
    sp = (top.adj5.mean() - bot.adj5.mean()) * 1e4
    t_sp = bt.cluster_t(np.concatenate([top.adj5.to_numpy() - te.adj5.mean(), te.adj5.mean() - bot.adj5.to_numpy()]), np.concatenate([top.t.to_numpy(), bot.t.to_numpy()]), 5)
    print(f"{nm}: 検証 n={len(te)} 予測と実績の相関={corr:.4f} 上位20%={top.adj5.mean() * 1e4:+.1f}bp 下位20%={bot.adj5.mean() * 1e4:+.1f}bp 差={sp:+.1f}bp t={t_sp:.2f}", flush=True)
    sub = te.sample(min(len(te), 20000), random_state=0)
    imp = permutation_importance(m, sub[X], sub.adj5c, n_repeats=3, random_state=0, scoring="neg_mean_squared_error")
    for f, v in sorted(zip(X, imp.importances_mean), key=lambda z: -z[1])[:8]:
        rows.append(dict(対象=nm, 特徴=FEATS[f], 重要度=v * 1e8))
    print("  重要度の上位:", ", ".join(f"{FEATS[f]}({v * 1e8:.1f})" for f, v in sorted(zip(X, imp.importances_mean), key=lambda z: -z[1])[:6]), flush=True)
pd.DataFrame(rows).to_csv("results_zone_char_ml.csv", index=False, encoding="utf-8-sig")
# 五分位別
q_rows = []
for f, lab in FEATS.items():
    if f in ("margin", "scale"):
        continue
    for nm, g in (("全体", df), ("抵抗帯(売り)", df[df.d < 0]), ("支持帯(買い)", df[df.d > 0])):
        qq = pd.qcut(g[f], 5, labels=False, duplicates="drop")
        for k in range(qq.max() + 1):
            m = qq == k
            q_rows.append(dict(特徴=lab, 対象=nm, 五分位=k + 1, n=int(m.sum()), 代表値=g[f][m].median(), adj5=g.adj5[m].mean() * 1e4, t=bt.cluster_t(g.adj5[m].clip(-0.15, 0.15).to_numpy(), g.t[m].to_numpy(), 5)))
Q = pd.DataFrame(q_rows); Q.to_csv("results_zone_char_q.csv", index=False, encoding="utf-8-sig")
S = Q[Q.対象 == "全体"].pivot(index="特徴", columns="五分位", values="adj5").round(1)
S["Q1-Q5"] = (S[1] - S[5]).round(1)
print(S.sort_values("Q1-Q5").to_string())
