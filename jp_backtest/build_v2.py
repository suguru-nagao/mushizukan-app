"""engine_v2.py の結果を、ダッシュボード用のJSON(report_data/v2/)にまとめる。
  python3 build_v2.py W|D|M
出力(tf = W または D):
  v2/{tf}_meta.json            シグナル一覧([名前, 種類, 向き])、日付、コード、保有期間の候補、日付1本あたりの日数
  v2/{tf}_{m}_{k}.json         銘柄別・シグナル別の成績。m = h{保有本数}(①) または o(②)。k = 銘柄番号÷400
                               {コード: [[sid, 件数, 勝率%, 平均bp, 中央値bp], ...]}(その銘柄自身の成績。基準は全銘柄=_base.json)
  v2/{tf}_lit.json             いま点灯中(最新から2本以内)のシグナル {コード: [[sid, 直近の発生のバー番号], ...]}
  v2/{tf}_tr_{k}.json          ②の、直近1年の取引(同じ種類・同じ向きは1ポジション) {コード: [[[sid,...], 入りバー, 出バー(-1=保有中), 入り値, 出値, 本数, 損益bp], ...]}
  v2/{tf}_base.json            全銘柄の、シグナル別・規模別の成績 {m: {sid: {z: [件数, 勝率, 平均bp, 中央値bp, 市場超過bp, 前半の平均, 後半の平均, 保有本数の中央値]}}}  z: 0小型 1中型 2大型 3全体
"""
import json, os, pickle, sys
import numpy as np, pandas as pd

TF = sys.argv[1] if len(sys.argv) > 1 else "W"
OUT = "report_data/v2"; os.makedirs(OUT, exist_ok=True)
meta = pickle.load(open(f"cache/v2_{TF}_meta.pkl", "rb")); HH = pd.read_pickle(f"cache/v2_{TF}_H.pkl"); OO = pd.read_pickle(f"cache/v2_{TF}_O.pkl")
sigs, codes, holds, T = meta["sigs"], meta["codes"], meta["holds"], len(meta["dates"]); NS, NC = len(sigs), len(codes)
LAST = {"W": 52, "D": 250, "M": 12}[TF]; LIT = 2; SH = 400
def dump(name, obj):
    json.dump(obj, open(f"{OUT}/{name}", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"), allow_nan=False)

last_t = meta["last_t"]
lit = {}
for sid, c in zip(*np.nonzero(last_t >= T - 1 - LIT)):
    lit.setdefault(codes[c], []).append([int(sid), int(last_t[sid, c])])
dump(f"{TF}_lit.json", lit)
dump(f"{TF}_meta.json", dict(sigs=[list(s) for s in sigs], dates=meta["dates"], codes=codes, holds=holds, days_per_bar=meta["days_per_bar"], lit_bars=LIT, shard=SH, last_bars=LAST))

def stats(df, col):
    g = df.groupby(["sid", "c"])[col]
    return pd.DataFrame(dict(n=g.size(), w=g.apply(lambda s: (s > 0).sum()), m=g.mean(), md=g.median()))

base = {}
modes = [(f"h{h}", HH, f"n{h}", f"x{h}") for h in holds] + [("o", OO[OO.closed], "net", "netex")]
for m, df, col, xcol in modes:
    df = df[np.isfinite(df[col])]
    st = stats(df, col)
    litset = {(s, c) for c, v in ((codes.index(k), v) for k, v in lit.items()) for s, _ in v}
    rows = {}
    for (sid, c), r in st.iterrows():
        if r.n < 3 and (sid, c) not in litset:
            continue
        rows.setdefault(c, []).append([int(sid), int(r.n), round(float(r.w / r.n * 100), 1), round(float(r.m * 1e4), 1), round(float(r.md * 1e4), 1)])
    for k in range((NC + SH - 1) // SH):
        dump(f"{TF}_{m}_{k}.json", {codes[c]: rows[c] for c in range(k * SH, min(NC, (k + 1) * SH)) if c in rows})
    # 全銘柄の規模別
    half = T // 2; key = "t" if m != "o" else "e"
    b = {}
    for sid, g0 in df.groupby("sid"):
        b[int(sid)] = {}
        for z in (0, 1, 2, 3):
            g = g0 if z == 3 else g0[g0.z == z]
            if len(g) == 0:
                continue
            first, second = g[g[key] < half][col], g[g[key] >= half][col]
            b[int(sid)][z] = [len(g), round(float((g[col] > 0).mean() * 100), 1), round(float(g[col].mean() * 1e4), 1), round(float(g[col].median() * 1e4), 1), round(float(g[xcol].mean() * 1e4), 1),
                              round(float(first.mean() * 1e4), 1) if len(first) else None, round(float(second.mean() * 1e4), 1) if len(second) else None,
                              None if m != "o" else float(g.w.median())]
    base[m] = b
    print(m, len(rows), flush=True)
dump(f"{TF}_base.json", base)
# ②の直近1年の取引: 同じ種類・同じ向きは同時に1ポジション。同じバーで入る複数のシグナルは1つにまとめる
fam = [s[1] for s in sigs]; d_ = [s[2] for s in sigs]
rec = OO[(OO.x >= T - 1 - LAST) | (~OO.closed)].sort_values(["c", "e", "sid"])
out = {}
cur = {}
for r in rec.itertuples():
    key = (r.c, fam[r.sid], d_[r.sid]); c0 = cur.get(key)
    if c0 is not None and (c0[2] < 0 or r.e < c0[2]):
        if r.e == c0[1]:
            c0[0].append(int(r.sid))
        continue
    t = [[int(r.sid)], int(r.e), int(r.x), round(float(r.pe), 1), round(float(r.px), 1), int(r.w), round(float(r.net * 1e4), 1)]
    cur[key] = t; out.setdefault(r.c, []).append(t)
for k in range((NC + SH - 1) // SH):
    dump(f"{TF}_tr_{k}.json", {codes[c]: out[c] for c in range(k * SH, min(NC, (k + 1) * SH)) if c in out})
print("done")
