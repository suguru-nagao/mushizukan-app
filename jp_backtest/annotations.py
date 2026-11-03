"""チャートに「どこに注目したシグナルか」を描くための、位置情報(抵抗帯・支持帯・ライン、チャートパターンの山谷とライン)を作る。
  python3 annotations.py W|D|M   → report_data/v2/{tf}_an_{k}.json  {コード: [[sid, 信号のバー, 位置情報], ...]}
対象: ②の直近1年の取引の、入りと出のシグナル(チャート・抵抗帯の種類のみ。ローソク足・テクニカルは、画面側で価格から描く)と、いま点灯中のシグナル。
位置情報(価格は、調整前に戻した値):
  抵抗帯・支持帯・ライン: {"z": [[下端, 上端, "v"|"l"], ...]}  v=価格帯別出来高の帯、l=折り返し点を結んだ線の帯
  チャートパターン: {"pts": [[バー, 価格, ラベル], ...], "lines": [[バー1, 価格1, バー2, 価格2, ラベル], ...]}
"""
import json, os, pickle, sys
import numpy as np, pandas as pd
import backtest as bt
import chart_patterns as cp
from zones import SR_PARAMS, TF_PARAMS, SR_MIN, SR_TOL, SR_HALF, _clean as zclean, make_zones, sr_lines, zone_sets, pivot_idx

TF = sys.argv[1] if len(sys.argv) > 1 else "W"
LAST = {"W": 52, "D": 250, "M": 12}[TF]; LIT = {"W": 2, "D": 2, "M": 1}[TF]; SH = 400
if TF in ("W", "M"):
    from long_run import load
    P, _ = load(TF)
else:
    P = bt.load_panel()
H, L, C, Cr, Vo = (P[k].astype(float) for k in ("H", "L", "C", "Craw", "Vo")); T, N = H.shape
meta = pickle.load(open(f"cache/v2_{TF}_meta.pkl", "rb")); codes = meta["codes"]; sigs = meta["sigs"]; last_t = meta["last_t"]
OO = pd.read_pickle(f"cache/v2_{TF}_O.pkl"); exits = pickle.load(open(f"cache/v2_{TF}_exits.pkl", "rb"))
fam = [s[1] for s in sigs]; names = [s[0] for s in sigs]
reqs = {}
def add(sid, c, t):
    if fam[sid] in ("chart", "zone") and t >= 1:
        reqs.setdefault(int(c), set()).add((int(sid), int(t)))
rec = OO[(OO.x >= T - 1 - LAST) | (~OO.closed)]
for r in rec.itertuples():
    add(r.sid, r.c, r.e - 1)
    for s2 in exits.get((int(r.sid), int(r.c), int(r.e)), []):
        add(s2, r.c, r.x - 1)
for sid, c in zip(*np.nonzero(last_t >= T - 1 - LIT)):
    add(sid, c, int(last_t[sid, c]))
print("銘柄", len(reqs), "シグナル位置", sum(len(v) for v in reqs.values()), flush=True)
W_, R_, hi_long, hi_short = TF_PARAMS[TF]; SRP = SR_PARAMS[TF]
f_adj = np.where(C > 0, Cr / C, np.nan)

ZSET = {"": "出来高", "線: ": "線", "重なり: ": "重なり", "出来高のみ: ": "出来高のみ", "線のみ: ": "線のみ"}
def zone_info(name):
    for p, st in ZSET.items():
        if p and name.startswith(p):
            return st, name[len(p):]
    return "出来高", name
def side_of(key):    # 抵抗側なら0、支持側なら2
    return 0 if "抵抗" in key else 2

def zone_geom(cl, t_rel, setname, key):
    first, h, l, cc, v = cl; n = len(cc); tp = (h + l + cc) / 3
    Ks, Wsr = SRP; hp, lp = pivot_idx(h, Ks), pivot_idx(-l, Ks)
    tau = t_rel - 1 if R_ == 1 else W_ - 1 + R_ * ((t_rel - 1 - (W_ - 1)) // R_)
    q = side_of(key)
    for _ in range(14):
        if tau < W_ - 1 or tau >= n - 1:
            return None, []
        vz = make_zones(tp[tau - W_ + 1:tau + 1], v[tau - W_ + 1:tau + 1], cc[tau])
        a0, a1 = np.searchsorted(hp, tau - Wsr), np.searchsorted(hp, tau - Ks, side="right")
        b0, b1 = np.searchsorted(lp, tau - Wsr), np.searchsorted(lp, tau - Ks, side="right")
        zs = zone_sets(vz, sr_lines(np.concatenate([h[hp[a0:a1]], l[lp[b0:b1]]])), cc[tau])
        out = []; pvs = []
        vband = (vz[q], vz[q + 1]); lband = (zs["線"][q], zs["線"][q + 1])
        if setname in ("出来高", "出来高のみ", "重なり") and not np.isnan(vband[0]): out.append([float(vband[0]), float(vband[1]), "v"])
        if setname in ("線", "線のみ", "重なり") and not np.isnan(lband[0]):
            out.append([float(lband[0]), float(lband[1]), "l"])
            # この線(帯)を作った折り返し点を探す: 窓内の折り返し点を、価格順に、近いものどうしでまとめ直し、帯の中心が一致するまとまりを選ぶ
            lv = np.concatenate([h[hp[a0:a1]], l[lp[b0:b1]]]); bb = np.concatenate([hp[a0:a1], lp[b0:b1]]); kd = np.array(["h"] * (a1 - a0) + ["l"] * (b1 - b0))
            od = np.argsort(lv); lv, bb, kd = lv[od], bb[od], kd[od]; st = 0; best = None
            for ii in range(1, len(lv) + 1):
                if ii == len(lv) or lv[ii] > lv[st] * (1 + SR_TOL):
                    if ii - st >= SR_MIN:
                        m = float(lv[st:ii].mean()); dist = abs(m * (1 - SR_HALF) - lband[0])
                        if best is None or dist < best[0]: best = (dist, st, ii)
                    st = ii
            if best is not None and best[0] < 1e-6 * max(1.0, lband[0]) * 100:
                pvs = [(int(bb[j]), float(lv[j]), str(kd[j])) for j in range(best[1], best[2])]
        if setname in ("出来高のみ",) : pass
        if setname == "出来高のみ":
            pass
        if out:
            return out, pvs
        tau -= R_
    return None, []

def chart_geoms(cl, wanted):
    """wanted: {(キー, 信号のバー(相対))...} → {(名前, バー相対): 位置情報}"""
    first, h, l, cc = cl; res = {}
    hi = cp.pivot_idx(h); lo = cp.pivot_idx(-l)
    rt = {}; cp.tops(h, l, cc, hi, rt)
    rb = {}; cp.tops(1e4 / l, 1e4 / h, 1e4 / cc, lo, rb)
    rc = {}; cp.channels(h, l, cc, hi, lo, rc)
    return rt, rb, rc

out = {}
cnt = 0
for c, rq in sorted(reqs.items()):
    zc = zclean(H[:, c], L[:, c], C[:, c], Vo[:, c])
    if zc is None:
        continue
    first = zc[0]; res = []
    cch = None
    for sid, t in rq:
        nm = names[sid]; t_rel = t - first; g = None
        if t_rel < 5:
            continue
        fa = f_adj[t, c] if np.isfinite(f_adj[t, c]) else 1.0
        if fam[sid] == "zone" and not nm.startswith("新"):
            st, key = zone_info(nm)
            z, pv = zone_geom(zc, t_rel, st, key)
            if z:
                g = {"z": [[round(a * fa, 1), round(b * fa, 1), k] for a, b, k in z]}
                if pv:
                    g["pv"] = [[int(b_ + first), round(p_ * fa, 1), kd_] for b_, p_, kd_ in sorted(pv)]
                    g["win"] = [int(max(0, t_rel - SRP[1]) + first), int(t_rel - SRP[0] + first)]   # 折り返し点を探した範囲(バー)
        elif fam[sid] == "chart":
            if cch is None:
                cc_ = cp._clean(H[:, c], L[:, c], C[:, c])
                if cc_ is None or len(cc_[1]) < 120:
                    cch = ("none",)
                else:
                    cch = cp.detect_charts.__globals__ and (cc_, *chart_geoms(cc_, None))
            if cch != ("none",):
                (cf, h2, l2, c2), rt, rb, rc = cch; rel = t - cf
                m = None; inv = False
                for key, nmap in (("double", "ダブルトップ"), ("triple", "トリプルトップ"), ("hs", "三尊")):
                    if nm == nmap: m = rt.get((key, rel))
                for key, nmap in (("double", "ダブルボトム"), ("triple", "トリプルボトム"), ("hs", "逆三尊")):
                    if nm == nmap: m = rb.get((key, rel)); inv = m is not None
                if m is None and "・" in nm:
                    base, dr = nm.split("・"); m = rc.get((rel, base, 1 if dr == "上抜け" else -1))
                if m is not None:
                    cv = (lambda v: 1e4 / v) if inv else (lambda v: v)
                    lab = {"double": ["山1", "山2"], "triple": ["山1", "山2", "山3"], "hs": ["左肩", "頭", "右肩", "谷1", "谷2"]}
                    pts = []
                    for j, (b, p) in enumerate(m["pts"]):
                        labs = (["谷1", "谷2"] if inv and nm in ("ダブルボトム",) else ["谷1", "谷2", "谷3"] if inv and nm == "トリプルボトム" else ["左肩", "頭", "右肩", "山1", "山2"] if inv and nm == "逆三尊" else lab.get(next((k for k, n2 in (("double", "ダブルトップ"), ("triple", "トリプルトップ"), ("hs", "三尊")) if nm == n2), ""), ["高値1", "高値2", "安値1", "安値2"]))
                        pts.append([int(b + cf), round(cv(p) * fa, 1), labs[j] if j < len(labs) else ""])
                    lines = [[int(b1 + cf), round(cv(p1) * fa, 1), int(b2 + cf), round(cv(p2) * fa, 1), lb] for b1, p1, b2, p2, lb in m["lines"]]
                    g = {"pts": pts, "lines": lines}
        if g:
            res.append([sid, t, g]); cnt += 1
    if res:
        out[c] = res
print("位置情報", cnt, flush=True)
for k in range((N + SH - 1) // SH):
    json.dump({codes[c]: out[c] for c in range(k * SH, min(N, (k + 1) * SH)) if c in out}, open(f"report_data/v2/{TF}_an_{k}.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
print("done")
