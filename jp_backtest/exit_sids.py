"""②の売買(直近1年)の、出のきっかけになったシグナル(反対向きのシグナルで、出の1本前に点灯していたもの)を調べる。
  python3 exit_sids.py W|D|M   → cache/v2_{tf}_exits.pkl  {(入りのsid, 銘柄番号, 入りのバー): [出のきっかけのsid, ...]}
あわせて、チャートの描画用に、銘柄ごとの直近の価格(調整前の始値・高値・安値・終値)を report_data/v2/{tf}_px_{k}.json に書く。
"""
import json, os, pickle, sys
import numpy as np, pandas as pd
import backtest as bt
from build_report import zname
from mtf_run import signals

TF = sys.argv[1] if len(sys.argv) > 1 else "W"
WIN = {"W": 156, "D": 330, "M": 48}[TF]; LAST = {"W": 52, "D": 250, "M": 12}[TF]; SH = 400
if TF in ("W", "M"):
    from long_run import load
    P, _ = load(TF)
else:
    P = bt.load_panel()
O, H, L, C, Cr = (P[k].astype(float) for k in ("O", "H", "L", "C", "Craw")); T, N = O.shape
meta = pickle.load(open(f"cache/v2_{TF}_meta.pkl", "rb")); codes = meta["codes"]; names = [s[0] for s in meta["sigs"]]
OO = pd.read_pickle(f"cache/v2_{TF}_O.pkl")
rec = OO[(OO.x >= T - 1 - LAST) & OO.closed].copy()
print("対象の取引", len(rec), flush=True)
# 価格(調整前に戻す)
f = np.where(C > 0, Cr / C, np.nan)
lo = T - WIN
out = {}
for k in range((N + SH - 1) // SH):
    d = {}
    for c in range(k * SH, min(N, (k + 1) * SH)):
        o = O[lo:, c] * f[lo:, c]; h = H[lo:, c] * f[lo:, c]; l = L[lo:, c] * f[lo:, c]; cl = Cr[lo:, c]
        if np.isfinite(cl).sum() < 3:
            continue
        r = lambda a: [None if not np.isfinite(v) else round(float(v), 1) for v in a]
        d[codes[c]] = {"o": r(o), "h": r(h), "l": r(l), "c": r(cl)}
    json.dump({"s": lo, "d": d}, open(f"report_data/v2/{TF}_px_{k}.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
print("価格を書き出し", flush=True)
# 出のきっかけ
sigs = []
for fam in ("candle", "chart", "tech", "zone"):
    s = None
    if TF == "D" and fam in ("chart", "tech") and os.path.exists(f"cache/{fam}_sig.pkl"):
        s = pickle.load(open(f"cache/{fam}_sig.pkl", "rb"))
    if s is None:
        s = signals(P, fam, TF)
    for nm, (d, S) in s.items():
        sigs.append((zname(nm), fam, int(d), np.asarray(S, bool)))
byname = {s[0]: s for s in sigs}
assert set(byname) == set(names), (len(byname), len(names))
sigs = [byname[n] for n in names]     # 検出の順番は、実行ごとに変わることがあるので、集計時の並び(sid)に合わせる
res = {}
fam_of = np.array([s[1] for s in sigs]); d_of = np.array([s[2] for s in sigs])
for sid2, (nm, fam, d, S) in enumerate(sigs):
    m = rec[(fam_of[rec.sid.to_numpy()] == fam) & (d_of[rec.sid.to_numpy()] == -d)]    # 入りの向きが逆(=この signal が反対側)
    if len(m) == 0:
        continue
    hit = S[(m.x.to_numpy() - 1).clip(0), m.c.to_numpy()]
    for r in m[hit].itertuples():
        res.setdefault((int(r.sid), int(r.c), int(r.e)), []).append(sid2)
pickle.dump(res, open(f"cache/v2_{TF}_exits.pkl", "wb"))
print("done", len(res))
