"""帯に過去に跳ね返された回数が多いほど、次も跳ね返されやすいか(点時点)。
イベント(日足の抵抗帯・支持帯への到達)ごとに、前日までの窓120本から帯を作り(make_zones)、前日までの250本で、その帯に触れた回数・跳ね返された回数を数える
(触れ終わって5本後の終値が判定できる回だけ。イベント日より後は使わない)。
"""
import numpy as np, pandas as pd
import backtest as bt
from zones import TF_PARAMS, _clean, make_zones
from zone_levels import MERGE, AFTER, LOOKBACK

P = bt.load_panel(); W = TF_PARAMS["D"][0]
df = pd.read_pickle(bt.CACHE + "/zone_char3.pkl")
res_cols = {k: np.full(len(df), np.nan) for k in ("n_touch", "n_rej", "n_brk")}
for c, g in df.groupby("c"):
    cl = _clean(P["H"][:, c], P["L"][:, c], P["C"][:, c], P["Vo"][:, c])
    if cl is None:
        continue
    first, h, l, cc, v = cl; tp = (h + l + cc) / 3
    for i, t, d in zip(g.index, g.t.to_numpy(), g.d.to_numpy()):
        u = t - 1 - first                     # 前日(帯を作る日)のバー位置
        if u < W + 5 or u >= len(cc) - 1:
            continue
        z = make_zones(tp[u - W + 1:u + 1], v[u - W + 1:u + 1], cc[u])
        lo, hi = (z[0], z[1]) if d < 0 else (z[2], z[3])
        if np.isnan(lo):
            continue
        # イベント日の高値(安値)がその帯に届いていること
        tt = t - first
        if (d < 0 and not (h[tt] >= lo and cc[tt] <= hi)) or (d > 0 and not (l[tt] <= hi and cc[tt] >= lo)):
            continue
        s0 = max(1, u - LOOKBACK + 1); touch = (h[s0:u + 1] >= lo) & (l[s0:u + 1] <= hi); idx = np.nonzero(touch)[0] + s0
        if len(idx) == 0:
            res_cols["n_touch"][df.index.get_loc(i)] = 0; res_cols["n_rej"][df.index.get_loc(i)] = 0; res_cols["n_brk"][df.index.get_loc(i)] = 0; continue
        grp = [[idx[0]]]
        for j in idx[1:]:
            (grp[-1].append(j) if j - grp[-1][-1] <= MERGE else grp.append([j]))
        nt = nr = nb = 0
        for gg in grp:
            s, e = gg[0], gg[-1]; pre = cc[s - 1]
            side = "below" if pre < lo else "above" if pre > hi else None
            if side is None or (d < 0) != (side == "below"):
                continue                                   # 抵抗帯は下から来た触れ、支持帯は上から来た触れだけを数える
            nt += 1; k = e + AFTER
            if k <= u:
                ok = cc[k] < lo if side == "below" else cc[k] > hi
                bk = cc[k] > hi if side == "below" else cc[k] < lo
                nr += ok; nb += bk
        j = df.index.get_loc(i); res_cols["n_touch"][j], res_cols["n_rej"][j], res_cols["n_brk"][j] = nt, nr, nb
for k, a in res_cols.items():
    df[k] = a
df.to_pickle(bt.CACHE + "/zone_char10.pkl")
d = df.dropna(subset=["n_touch"]).copy(); d["y"] = d.adj5.clip(-.15, .15) * 1e4
print("対象イベント", len(d), "/", len(df))
rows = []
for nm, g in (("抵抗帯(売り)", d[d.d < 0]), ("支持帯(買い)", d[d.d > 0])):
    for lab, f in (("跳ね返された回数", g.n_rej), ("突破・割れの回数", g.n_brk), ("触れた回数", g.n_touch)):
        b = pd.cut(f, [-1, 0, 1, 2, 99], labels=["0回", "1回", "2回", "3回以上"])
        for q in ("0回", "1回", "2回", "3回以上"):
            s = g[b == q]
            if len(s) > 100:
                rows.append(dict(対象=nm, 指標=lab, 区分=q, n=len(s), 平均bp=s.y.mean(), 中央値bp=s.y.median(), 勝率=(s.y > 0).mean() * 100, t=bt.cluster_t(s.y.to_numpy() / 1e4, s.t.to_numpy(), 5)))
    g2 = g[g.n_touch >= 2].copy(); g2["rate"] = g2.n_rej / g2.n_touch
    for lab, s in (("触れ2回以上で、跳ね返し率50%以上", g2[g2.rate >= .5]), ("触れ2回以上で、跳ね返し率50%未満", g2[g2.rate < .5])):
        rows.append(dict(対象=nm, 指標="跳ね返し率", 区分=lab, n=len(s), 平均bp=s.y.mean(), 中央値bp=s.y.median(), 勝率=(s.y > 0).mean() * 100, t=bt.cluster_t(s.y.to_numpy() / 1e4, s.t.to_numpy(), 5)))
R = pd.DataFrame(rows).round(1); R.to_csv("results_zone_char_bounce.csv", index=False, encoding="utf-8-sig"); print(R.to_string(index=False))
