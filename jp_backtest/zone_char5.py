import warnings
import numpy as np
import pandas as pd
import backtest as bt
warnings.filterwarnings("ignore")
df = pd.read_pickle(bt.CACHE + "/zone_char3.pkl")
df["margin"] = df.margin.astype(float)
for c, f in (("turn60", "lturn60"), ("turn250", "lturn250"), ("mcap", "lmcap"), ("va", "lva"), ("price", "lprice"), ("vol60", "lvol60")):
    df[f] = np.log(df[c].clip(lower=1e-6))
df = df.dropna(subset=["adj5", "lturn250", "lmcap", "lvol60", "zage", "zstr", "dist_hi", "ret120", "gap", "day_ret", "age"])
df["y"] = df.adj5.clip(-0.15, 0.15) * 1e4
def tab(g, key, lab):
    r = g.groupby(key).agg(n=("y", "size"), adj=("y", "mean"), turn=("turn250", "median"))
    r["t"] = [bt.cluster_t(g.y[g[key] == k].to_numpy() / 1e4, g.t[g[key] == k].to_numpy(), 5) for k in r.index]
    print(f"== {lab}"); print(r.round(2).to_string())
tab(df.assign(mk=df.mkt.map({"0111": "プライム", "0112": "スタンダード", "0113": "グロース"})), "mk", "市場区分")
tab(df.assign(sc=df.scale.map({5: "Core30", 4: "Large70", 3: "Mid400", 2: "Small1", 1: "Small2", 0: "その他(小型)"})), "sc", "TOPIX規模区分")
df["zage_q"] = pd.qcut(df.zage, 5, labels=False); tab(df, "zage_q", "帯の出来高の古さの五分位(0=新しい)")
df["zstr_q"] = pd.qcut(df.zstr, 5, labels=False); tab(df, "zstr_q", "帯の強さの五分位(0=弱い)")
df["age_q"] = pd.qcut(df.age, 5, labels=False); tab(df, "age_q", "上場経過年数の五分位(0=新しい)")
# 業種
sec = df.groupby("sector").agg(n=("y", "size"), adj=("y", "mean"))
sec = sec[sec.n >= 2500].sort_values("adj")
print("== 業種(件数2,500以上)"); print(pd.concat([sec.head(5), sec.tail(5)]).round(1).to_string())
# 多変量(標準化。クラスター頑健)
X = ["lturn250", "lmcap", "lvol60", "ret120", "dist_hi", "zage", "zstr", "gap", "day_ret", "margin", "age", "scale"]
Z = df[X].astype(float); Z = (Z - Z.mean()) / Z.std(); Z.insert(0, "const", 1.0)
y = df.y.to_numpy(); b = np.linalg.lstsq(Z.to_numpy(), y, rcond=None)[0]; e = y - Z.to_numpy() @ b
cl = (df.t.to_numpy() // 5).astype(int); XtX = np.linalg.inv(Z.to_numpy().T @ Z.to_numpy()); meat = np.zeros((Z.shape[1],) * 2)
for g in np.unique(cl):
    m = cl == g; s = Z.to_numpy()[m].T @ e[m]; meat += np.outer(s, s)
se = np.sqrt(np.diag(XtX @ meat @ XtX))
names = dict(lturn250="回転率(250日)", lmcap="時価総額", lvol60="変動率(60日)", ret120="直近120日の騰落率", dist_hi="250日高値からの距離", zage="帯の出来高の古さ", zstr="帯の強さ", gap="当日の窓", day_ret="当日の騰落率", margin="貸借銘柄", age="上場経過年数", scale="TOPIX規模区分", const="定数")
R = pd.DataFrame(dict(特徴=[names[c] for c in Z.columns], 係数bp=b, t=b / se)); R.to_csv("results_zone_char_ols2.csv", index=False, encoding="utf-8-sig")
print("== 多変量(1標準偏差あたりのbp)"); print(R.round(2).to_string(index=False))
