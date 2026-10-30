"""決算データ(株主・財務の代用指標)と、リバウンドの強さ。取得済みの銘柄だけで分析する。
代用指標: 配当利回り(長期保有の株主が多いほど高い傾向)、自己株式比率(流通する株が少ない)、PBR、PER、ROE、自己資本比率、直近の決算からの日数。
"""
import glob
import gzip
import json
import os
import warnings

import numpy as np
import pandas as pd

import backtest as bt

warnings.filterwarnings("ignore")
df = pd.read_pickle(bt.CACHE + "/zone_char3.pkl")
P = bt.load_panel(); dates = P["dates"]
df["date"] = dates[df.t.to_numpy()]
have = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(bt.CACHE, "fins", "*.json.gz"))}
df = df[df.code.isin(have)].copy()
num = lambda s: pd.to_numeric(s.replace("", np.nan), errors="coerce")
parts = []
for code, g in df.groupby("code"):
    rows = json.load(gzip.open(os.path.join(bt.CACHE, "fins", f"{code}.json.gz"), "rt", encoding="utf-8"))
    if not rows:
        continue
    f = pd.DataFrame(rows)
    f["DiscDate"] = pd.to_datetime(f.DiscDate)
    for c in ("Eq", "ShOutFY", "TrShFY", "FEPS", "FDivAnn", "DivAnn", "EqAR", "FNP", "FSales", "FOP", "Sales", "NP", "EPS", "BPS"):
        if c in f:
            f[c] = num(f[c])
    f = f.sort_values("DiscDate")
    # 項目ごとに、直近の値で埋める(前の決算の値を引き継ぐ)
    ff = f[["DiscDate", "Eq", "ShOutFY", "TrShFY", "FEPS", "FDivAnn", "DivAnn", "EqAR", "FNP", "FSales", "FOP"]].copy()
    for c in ff.columns[1:]:
        ff[c] = ff[c].ffill()
    ff["disc"] = ff.DiscDate
    g = g.sort_values("date")
    m = pd.merge_asof(g, ff, left_on="date", right_on="DiscDate", direction="backward", allow_exact_matches=False)
    parts.append(m)
D = pd.concat(parts, ignore_index=True)
sh = D.ShOutFY - D.TrShFY.fillna(0)
D["div_yield"] = (D.FDivAnn.fillna(D.DivAnn)) / D.price
D["treasury"] = D.TrShFY / D.ShOutFY
D["pbr"] = D.price / (D.Eq / sh)
D["per"] = D.price / D.FEPS
D["roe"] = D.FNP / D.Eq
D["eqar"] = D.EqAR
D["op_margin"] = D.FOP / D.FSales
D["since_disc"] = (D.date - D.disc).dt.days
D["y"] = D.adj5.clip(-0.15, 0.15) * 1e4
D.loc[(D.per <= 0) | (D.per > 200), "per"] = np.nan
D.loc[(D.pbr <= 0) | (D.pbr > 30), "pbr"] = np.nan
FE = {"div_yield": "配当利回り(予想)", "treasury": "自己株式比率", "pbr": "PBR", "per": "PER(予想)", "roe": "ROE(予想)", "eqar": "自己資本比率", "op_margin": "営業利益率(予想)", "since_disc": "直近の決算からの日数"}
print("銘柄数", D.code.nunique(), "イベント", len(D))
rows = []
for f, lab in FE.items():
    for nm, g in (("全体", D), ("抵抗帯(売り)", D[D.d < 0]), ("支持帯(買い)", D[D.d > 0])):
        g = g.dropna(subset=[f])
        if len(g) < 5000:
            continue
        q = pd.qcut(g[f], 5, labels=False, duplicates="drop")
        for k in range(q.max() + 1):
            mk = q == k
            rows.append(dict(特徴=lab, 対象=nm, 五分位=k + 1, n=int(mk.sum()), 代表値=g[f][mk].median(), adj5=g.y[mk].mean(), t=bt.cluster_t(g.y[mk].to_numpy() / 1e4, g.t[mk].to_numpy(), 5)))
Q = pd.DataFrame(rows); Q.to_csv("results_zone_char_fund.csv", index=False, encoding="utf-8-sig")
S = Q[Q.対象 == "全体"].pivot(index="特徴", columns="五分位", values="adj5").round(1); S["Q1-Q5"] = (S[1] - S[5]).round(1)
V = Q[Q.対象 == "全体"].pivot(index="特徴", columns="五分位", values="代表値").round(3)
print(S.to_string()); print(V.to_string())
for nm in ("抵抗帯(売り)", "支持帯(買い)"):
    S2 = Q[Q.対象 == nm].pivot(index="特徴", columns="五分位", values="adj5").round(1); S2["Q1-Q5"] = (S2[1] - S2[5]).round(1); print(nm); print(S2.to_string())
