"""価格帯別出来高の最多付近(抵抗帯・支持帯)でのリバウンドの強さは、銘柄の特徴で違うか。
対象: 日足(J-Quants、2021年10月〜)の「抵抗帯に到達(跳ね返される)」(売り)と「支持帯に到達(跳ね返す)」(買い)。出来高の帯のみ(従来の抵抗帯)。
結果の指標: シグナルの向き(売り=下がる方が正)の超過リターン(翌日始値で入り h 日後の始値で出る。同日・同規模の全銘柄平均を引き、同じ騰落率の日の平均も引く)。
銘柄の特徴(シグナル日時点で分かる値):
  turn  : 直近60日の売買代金の中央値 / 時価総額 = 1日の回転率。1/turn = 平均的な保有日数の目安(回転が遅い=長く持たれている)
  margin: 貸借銘柄(制度信用の売買が可能で、貸借取引がある銘柄。各時点の指定)。※信用残のデータは、契約プランに含まれないため、使えない
  mcap, va, vol, price, market: 時価総額・売買代金・変動率(60日)・株価・市場区分
"""
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

import backtest as bt
import chart_matched as cm
from zones import SR_PARAMS, TF_PARAMS, detect_zones

HERE = os.path.dirname(os.path.abspath(__file__))
P = bt.load_panel(); U = bt.load_margin(P)
dates, codes = P["dates"], P["codes"]
T, N = len(dates), len(codes)
# 時価総額(百万円)
frames = []
for f in sorted(glob.glob(os.path.join(bt.CACHE, "bars", "*.json.gz"))):
    rows = json.load(gzip.open(f, "rt", encoding="utf-8"))
    if rows:
        frames.append(pd.DataFrame(rows)[["Date", "Code", "MktCap"]])
mc = pd.concat(frames).pivot(index="Date", columns="Code", values="MktCap").sort_index()
MCAP = pd.to_numeric(mc.stack(), errors="coerce").unstack().reindex(index=[d.strftime("%Y-%m-%d") for d in dates], columns=codes).to_numpy(dtype=float)
Va = P["Va"]
turn = pd.DataFrame(Va / (MCAP * 1e6)).rolling(60, min_periods=30).median().to_numpy()
ret = np.full_like(P["C"], np.nan); ret[1:] = P["C"][1:] / P["C"][:-1] - 1
vol = pd.DataFrame(ret).rolling(60, min_periods=30).std().to_numpy()
print("turn median", np.nanmedian(turn), "mcap median(百万円)", np.nanmedian(MCAP), flush=True)

sigs = detect_zones(P["H"], P["L"], P["C"], P["Vo"], *TF_PARAMS["D"], sr=SR_PARAMS["D"])
names = {"抵抗帯に到達(跳ね返される)": -1, "支持帯に到達(跳ね返す)": +1}
liq = bt.liquidity_bucket(P)
ok = ~bt.ahead(P["limit"].astype(float), 1).astype(bool)
nolim = ~P["limit"]
prev = np.vstack([np.full((1, N), np.nan), P["C"][:-1]]); r0 = P["C"] / prev - 1
out = []
for h in (1, 5):
    fr = bt.forward_returns(P, [h])[h]
    r = np.where(ok, fr, np.nan); x = r - bt.baseline(r, liq)
    xm = np.full_like(x, np.nan)
    for size, lo, hi in cm.SIZES:
        grp = (liq >= lo) & (liq <= hi) & nolim & ~np.isnan(x) & ~np.isnan(r0)
        edges = np.unique(np.percentile(r0[grp], cm.QS)); bins = np.digitize(r0, edges[1:-1])
        base = np.full(len(edges) + 1, np.nan)
        for b in np.unique(bins[grp]):
            base[b] = x[grp & (bins == b)].mean()
        xm = np.where(grp, x - base[bins], xm)
    for nm, d in names.items():
        _, S = sigs[nm]
        t, c = np.nonzero(S & ~np.isnan(xm))
        out.append(pd.DataFrame(dict(h=h, sig=nm, d=d, t=t, c=c, adj=d * xm[t, c], raw=d * x[t, c], turn=turn[t, c], margin=U[t, c], mcap=MCAP[t, c], va=Va[t, c], vol=vol[t, c], price=P["Craw"][t, c], mkt=P["mkt"][c], liq=liq[t, c])))
df = pd.concat(out, ignore_index=True)
df.to_parquet(os.path.join(HERE, "cache", "zone_char.parquet")) if False else df.to_pickle(os.path.join(bt.CACHE, "zone_char.pkl"))
print(df.groupby(["h", "sig"]).agg(n=("adj", "size"), adj=("adj", "mean")).round(5))
