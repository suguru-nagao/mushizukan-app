import numpy as np
import pandas as pd

import backtest as bt

df = pd.read_pickle(bt.CACHE + "/zone_char.pkl")
df["lturn"] = np.log(df.turn.clip(lower=1e-6)); df["hold"] = 1 / df.turn
df["lmcap"] = np.log(df.mcap); df["lva"] = np.log(df.va); df["lprice"] = np.log(df.price); df["lvol"] = np.log(df.vol.clip(lower=1e-4))
df = df.dropna(subset=["turn", "margin", "mcap", "vol", "adj"])
pd.set_option("display.width", 200)


def ct(x, t, h):
    return bt.cluster_t(x / 1e4 * 1e4, t, max(h, 5)) if len(x) > 50 else np.nan


rows = []
for h in (1, 5):
    d = df[df.h == h]
    for nm, g in (("全体(売り・買い合わせ)", d), ("抵抗帯(売り)", d[d.d < 0]), ("支持帯(買い)", d[d.d > 0])):
        # 回転率の五分位(その日の銘柄間ではなく、全イベントで)
        q = pd.qcut(g.turn, 5, labels=False, duplicates="drop")
        for k in range(5):
            m = q == k
            rows.append(dict(h=h, 対象=nm, 特徴="回転率", 区分=f"Q{k + 1}" + ("(遅い)" if k == 0 else "(速い)" if k == 4 else ""), n=int(m.sum()), 値=g.turn[m].median() * 100, adj=g.adj[m].mean() * 1e4, t=ct(g.adj[m].to_numpy() * 1e-4 * 1e4, g.t[m].to_numpy(), h) if False else bt.cluster_t(g.adj[m].to_numpy(), g.t[m].to_numpy(), max(h, 5))))
        for lab, m in (("貸借銘柄", g.margin), ("貸借以外", ~g.margin)):
            rows.append(dict(h=h, 対象=nm, 特徴="信用(貸借)", 区分=lab, n=int(m.sum()), 値=np.nan, adj=g.adj[m].mean() * 1e4, t=bt.cluster_t(g.adj[m].to_numpy(), g.t[m].to_numpy(), max(h, 5))))
        for feat, col in (("時価総額", "mcap"), ("売買代金", "va"), ("変動率(60日)", "vol"), ("株価", "price")):
            q = pd.qcut(g[col], 5, labels=False, duplicates="drop")
            for k in (0, 2, 4):
                m = q == k
                rows.append(dict(h=h, 対象=nm, 特徴=feat, 区分=f"Q{k + 1}", n=int(m.sum()), 値=g[col][m].median(), adj=g.adj[m].mean() * 1e4, t=bt.cluster_t(g.adj[m].to_numpy(), g.t[m].to_numpy(), max(h, 5))))
R = pd.DataFrame(rows); R.to_csv("results_zone_char.csv", index=False, encoding="utf-8-sig")
print(R[(R.h == 5) & (R.対象 == "全体(売り・買い合わせ)")][["特徴", "区分", "n", "値", "adj", "t"]].round(2).to_string(index=False))
print(R[(R.h == 1) & (R.対象 == "全体(売り・買い合わせ)")][["特徴", "区分", "n", "adj", "t"]].round(2).to_string(index=False))

# 回帰(クラスター頑健)
def ols(d, cols, h):
    X = d[cols].astype(float).copy()
    X = (X - X.mean()) / X.std()
    X.insert(0, "const", 1.0)
    y = d.adj.to_numpy() * 1e4
    b = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    e = y - X.to_numpy() @ b
    cl = (d.t.to_numpy() // max(h, 5)).astype(int)
    XtX = np.linalg.inv(X.to_numpy().T @ X.to_numpy())
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(cl):
        m = cl == g; s = X.to_numpy()[m].T @ e[m]; meat += np.outer(s, s)
    se = np.sqrt(np.diag(XtX @ meat @ XtX))
    return pd.DataFrame(dict(係数bp=b, t=b / se), index=X.columns)

df["margin_f"] = df.margin.astype(float)
for h in (1, 5):
    for nm, mask in (("全体", df.d != 0), ("抵抗帯(売り)", df.d < 0), ("支持帯(買い)", df.d > 0)):
        d = df[(df.h == h) & mask]
        r = ols(d, ["lturn", "margin_f", "lmcap", "lvol", "lprice"], h)
        print(f"== 回帰 h={h} {nm} (標準化。係数=1標準偏差あたりのbp)"); print(r.round(2).to_string())
        r.to_csv(f"results_zone_char_ols_{h}_{nm}.csv", encoding="utf-8-sig")
