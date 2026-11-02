"""チャート用の価格(調整前に戻した始値・高値・安値・終値)を、銘柄ごとに report_data/v2/{tf}_px_{k}.json に書く。
  python3 px_build.py W|D|M    表示できる期間: 日足 約1.7年(420本)・週足 約5年(260本)・月足 10年(120本)
"""
import json, pickle, sys
import numpy as np
import backtest as bt
TF = sys.argv[1] if len(sys.argv) > 1 else "W"
WIN = {"W": 260, "D": 420, "M": 120}[TF]; SH = 400
if TF in ("W", "M"):
    from long_run import load
    P, _ = load(TF)
else:
    P = bt.load_panel()
O, H, L, C, Cr = (P[k].astype(float) for k in ("O", "H", "L", "C", "Craw")); T, N = O.shape
codes = pickle.load(open(f"cache/v2_{TF}_meta.pkl", "rb"))["codes"]
f = np.where(C > 0, Cr / C, np.nan); lo = max(0, T - WIN)
for k in range((N + SH - 1) // SH):
    d = {}
    for c in range(k * SH, min(N, (k + 1) * SH)):
        cl = Cr[lo:, c]
        if np.isfinite(cl).sum() < 3:
            continue
        r = lambda a: [None if not np.isfinite(v) else (int(round(v)) if abs(v) >= 1000 else round(float(v), 1)) for v in a]   # 1000円以上は整数にして、容量を減らす
        d[codes[c]] = {"o": r(O[lo:, c] * f[lo:, c]), "h": r(H[lo:, c] * f[lo:, c]), "l": r(L[lo:, c] * f[lo:, c]), "c": r(cl)}
    json.dump({"s": lo, "d": d}, open(f"report_data/v2/{TF}_px_{k}.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
print("done", TF, lo, T)
