import warnings
import numpy as np
import pandas as pd
import backtest as bt
warnings.filterwarnings("ignore")
df = pd.read_pickle(bt.CACHE + "/zone_char3.pkl")
df["margin"] = df.margin.astype(float)
for c, f in (("turn60", "lturn60"), ("turn250", "lturn250"), ("mcap", "lmcap"), ("va", "lva"), ("price", "lprice"), ("vol60", "lvol60")):
    df[f] = np.log(df[c].clip(lower=1e-6))
df = df.dropna(subset=["adj5", "lturn250", "lmcap", "lvol60", "zage", "zstr", "dist_hi", "ret120", "gap", "day_ret", "age", "ret20"])
df["y"] = df.adj5.clip(-0.15, 0.15) * 1e4
names = dict(lturn250="回転率(250日)", lmcap="時価総額", lvol60="変動率(60日)", ret20="直近20日の騰落率", ret120="直近120日の騰落率", dist_hi="250日高値からの距離", zage="帯の出来高の古さ", zstr="帯の強さ", gap="当日の窓", day_ret="当日の騰落率", margin="貸借銘柄", age="上場経過年数", const="定数")
X = ["lturn250", "lmcap", "lvol60", "ret20", "ret120", "dist_hi", "zage", "zstr", "gap", "day_ret", "margin", "age"]
out = {}
for nm, g in (("抵抗帯(売り)", df[df.d < 0]), ("支持帯(買い)", df[df.d > 0])):
    Z = g[X].astype(float); Z = (Z - Z.mean()) / Z.std(); Z.insert(0, "const", 1.0)
    y = g.y.to_numpy(); b = np.linalg.lstsq(Z.to_numpy(), y, rcond=None)[0]; e = y - Z.to_numpy() @ b
    cl = (g.t.to_numpy() // 5).astype(int); XtX = np.linalg.inv(Z.to_numpy().T @ Z.to_numpy()); meat = np.zeros((Z.shape[1],) * 2)
    for k in np.unique(cl):
        m = cl == k; s = Z.to_numpy()[m].T @ e[m]; meat += np.outer(s, s)
    se = np.sqrt(np.diag(XtX @ meat @ XtX))
    out[nm] = pd.DataFrame(dict(b=b, t=b / se), index=[names[c] for c in Z.columns])
R = pd.concat(out, axis=1); R.columns = [f"{a}_{b}" for a, b in R.columns]; R.to_csv("results_zone_char_ols3.csv", encoding="utf-8-sig")
print(R.round(2).to_string())
