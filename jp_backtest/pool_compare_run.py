"""仲間の作り方・期間の比較を、5設定で実行して CSV に保存する(stock_pool2)。"""
import pandas as pd

import stock_pool2 as p2
import stock_pool3 as p3

rows = []
for h, f in p3.SET:
    for m, r in p2.evaluate(h, f).items():
        rows.append(dict(method=m, h=h, frac=f, pairs=r["pairs"], corr=r["corr"], within=r["within"]))
pd.DataFrame(rows).to_csv("results_pool_compare.csv", index=False, encoding="utf-8-sig")
rows = []
for w in (None, 520, 260, 156, 104):
    for h, f in p3.SET:
        o = p2.evaluate(h, f, window=w)
        for m in ("corr60", "corr200", "corrz200", "secc15"):
            rows.append(dict(window=w or 0, h=h, frac=f, method=m, corr=o[m]["corr"], within=o[m]["within"], pairs=o[m]["pairs"]))
pd.DataFrame(rows).to_csv("results_pool_window.csv", index=False, encoding="utf-8-sig")
print("done")
