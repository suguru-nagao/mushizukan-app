"""銘柄ごとの「いつもの値動き」(その銘柄の全日の平均超過リターン)を引いても、特徴の効果が残るか。
adj5 は、同日・同規模の平均を引いただけなので、割高株が下がり続ける、などの、銘柄の持続的な傾向(ドリフト)が混ざる。
"""
import warnings
import numpy as np
import pandas as pd
import backtest as bt
warnings.filterwarnings("ignore")
df = pd.read_pickle(bt.CACHE + "/zone_char3.pkl")
P = bt.load_panel(); liq = bt.liquidity_bucket(P)
ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
r = np.where(ok, bt.forward_returns(P, [5])[5], np.nan); x = r - bt.baseline(r, liq)
mc = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)          # 銘柄ごとの全日の平均超過リターン(5日)
df["drift"] = mc[df.c.to_numpy()]
df["adj5fe"] = df.adj5 - df.d * df.drift
print("銘柄ごとの平均超過リターン(5日)の分布 bp: ", np.nanpercentile(mc * 1e4, [5, 25, 50, 75, 95]).round(1))
print("イベントの平均 adj5=%.1fbp → ドリフト除去後 %.1fbp" % (df.adj5.mean() * 1e4, df.adj5fe.mean() * 1e4))
df["y"] = df.adj5fe.clip(-0.15, 0.15) * 1e4; df["y0"] = df.adj5.clip(-0.15, 0.15) * 1e4
df["lturn250"] = np.log(df.turn250.clip(lower=1e-6)); df["lmcap"] = np.log(df.mcap)
rows = []
def q(g, f, lab, nm):
    g = g.dropna(subset=[f]); k = pd.qcut(g[f], 5, labels=False, duplicates="drop")
    a = g[k == 0]; b = g[k == k.max()]
    rows.append(dict(特徴=lab, 対象=nm, Q1=a.y.mean(), Q5=b.y.mean(), Q1_ドリフト除去前=a.y0.mean(), Q5_ドリフト除去前=b.y0.mean(), t_Q1=bt.cluster_t(a.y.to_numpy() / 1e4, a.t.to_numpy(), 5), t_Q5=bt.cluster_t(b.y.to_numpy() / 1e4, b.t.to_numpy(), 5)))
for nm, g in (("全体", df), ("抵抗帯(売り)", df[df.d < 0]), ("支持帯(買い)", df[df.d > 0])):
    for f, lab in (("lturn250", "回転率(250日)"), ("lmcap", "時価総額"), ("zstr", "帯の強さ"), ("zage", "帯の出来高の古さ"), ("vol60", "変動率(60日)"), ("dist_hi", "250日高値からの距離"), ("ret20", "直近20日の騰落率")):
        q(g, f, lab, nm)
R = pd.DataFrame(rows).round(1); R.to_csv("results_zone_char_fe.csv", index=False, encoding="utf-8-sig")
print(R.to_string(index=False))
# 市場区分
for mk, lab in (("0111", "プライム"), ("0112", "スタンダード"), ("0113", "グロース")):
    g = df[df.mkt == mk]; print(lab, round(g.y0.mean(), 1), "→", round(g.y.mean(), 1))
