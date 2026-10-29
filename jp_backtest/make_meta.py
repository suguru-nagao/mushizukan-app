"""シグナル一覧の絞り込み用: 銘柄リスト(P["codes"]と同じ順)・業種・保有期間を report_data/meta.json に書く。"""
import json
import os

import backtest as bt
from hold_run import HOLDS

P = bt.load_panel()
m = P["master"].loc[P["codes"]]
secs = sorted(m["S33Nm"].dropna().unique())
stocks = [[c[:4], r.CoName.replace("　", " "), {"0111": 0, "0112": 1, "0113": 2}.get(r.Mkt, -1), r.S33Nm, c] for c, r in zip(m.index, m.itertuples())]
os.makedirs("report_data", exist_ok=True)
json.dump({"stocks": stocks, "secs": secs, "holds": {k: v for k, v in HOLDS.items() if k in ("D", "W")}}, open("report_data/meta.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(len(stocks), len(secs))
