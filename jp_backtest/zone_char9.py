"""信用残(株探・直近5週平均)の多い銘柄ほど、価格帯別出来高の最多付近で跳ね返るか(現在値による近似)。
信用残は銘柄ごとに安定していると仮定し、直近5週の平均を全期間のイベントに当てはめる(点時点ではない)。
"""
import glob, json, os
import numpy as np
import pandas as pd
import backtest as bt

df = pd.read_pickle(bt.CACHE + "/zone_char3.pkl")
rows = []
for p in glob.glob(os.path.join(bt.CACHE, "kabutan", "*.json")):
    r = json.load(open(p))
    if len(r) >= 3:
        a = np.array([[x[1], x[2]] for x in r], float)
        rows.append((os.path.basename(p)[:4], a[:, 0].mean() * 1e3, a[:, 1].mean() * 1e3))
M = pd.DataFrame(rows, columns=["c4", "short", "long"])
df["c4"] = df.code.astype(str).str[:4]
df = df.merge(M, on="c4")
df["shares"] = df.mcap * 1e6 / df.price
df["long_r"] = df["long"] / df.shares; df["short_r"] = df["short"] / df.shares
df["ratio"] = df["long"] / df["short"].clip(lower=1)
dvol = df.va / df.price                                   # 日出来高(株)の概算
df["long_days"] = df["long"] / dvol; df["tot_days"] = (df["long"] + df["short"]) / dvol
for f in ("long_r", "short_r", "ratio", "long_days", "tot_days"):
    df["l_" + f] = np.log(df[f].clip(lower=1e-6))
df["y"] = df.adj5.clip(-0.15, 0.15) * 1e4
print("対象銘柄数", df.c4.nunique(), "イベント数", len(df))
out = []
for nm, g in (("抵抗帯(売り)", df[df.d < 0]), ("支持帯(買い)", df[df.d > 0])):
    for f, lab in (("l_long_r", "買い残/発行済株式"), ("l_short_r", "売り残/発行済株式"), ("l_ratio", "信用倍率"), ("l_long_days", "買い残の出来高日数"), ("l_tot_days", "信用残合計の出来高日数")):
        gg = g.dropna(subset=[f]); k = pd.qcut(gg[f], 5, labels=False, duplicates="drop")
        for q in (0, k.max()):
            s = gg[k == q]; y = s.y.to_numpy()   # adj5 は向き(d)調整済み: 正=跳ね返った
            out.append(dict(対象=nm, 特徴=lab, 分位="Q1(低)" if q == 0 else "Q5(高)", n=len(s), 平均bp=y.mean(), 中央値bp=np.median(y), 勝率=(y > 0).mean() * 100,
                            t=bt.cluster_t(y / 1e4, s.t.to_numpy(), 5)))
R = pd.DataFrame(out).round(1); R.to_csv("results_zone_char_margin.csv", index=False, encoding="utf-8-sig")
print(R.to_string(index=False))
