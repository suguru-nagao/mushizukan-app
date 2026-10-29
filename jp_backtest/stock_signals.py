"""銘柄別に、買いシグナルの過去実績を推定してランキングする(補正あり)+ 過去データでの検証。

使い方: python3 stock_signals.py 9759 [保有日数=10]  /  python3 stock_signals.py --validate
考え方:
- 1銘柄だけでは件数が少なく、当たりが偶然になりやすい。そこで、銘柄の平均を、全銘柄の同じシグナルの平均に寄せる(縮小推定)。
  寄せ方は、シグナルごとに「銘柄間の差が本物か」を前半データで推定して決める(差が小さければ、全銘柄の平均にほぼ一致)。
- --validate は、前半で選んだシグナルが、後半に実際に良かったかを全銘柄で調べる(選び方が役に立つかの検証)。
"""
import os
import sys

import numpy as np

import backtest as bt

COST = 20.0
Z = np.load(os.path.join(bt.CACHE, "stock_events.npz"), allow_pickle=True)
SID, T, C, SZ = Z["sid"].astype(np.int32), Z["t"].astype(np.int32), Z["c"].astype(np.int32), Z["z"]
NAMES, FAMS, DATES, CODES = Z["names"], Z["fams"], Z["dates"], Z["codes"]
NS, NC = len(NAMES), len(CODES)
SPLIT = len(DATES) // 2


def thin(h):
    """同じ銘柄・同じシグナルで、h 日以内に重なる発生を間引く(h 日ごとのブロックの先頭だけ残す)。"""
    key = (SID.astype(np.int64) * NC + C) * 4000 + T // h
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def prep(h):
    idx = thin(h)
    return dict(sid=SID[idx], c=C[idx], t=T[idx], z=SZ[idx], x=Z[f"x{h}"][idx].astype(np.float64) * 1e4)


def fit(E, mask, zsel=None):
    """mask の期間のデータで、(シグナル, 銘柄) ごとの推定値を作る。返値: dict[(sid,c)] -> (n, 平均, 縮小推定, 全銘柄平均)。"""
    if zsel is not None:
        mask = mask & (E["z"] == zsel)   # 規模(0=小型 1=中型 2=大型)で分けて推定する
    sid, c, z, x = E["sid"][mask], E["c"][mask], E["z"][mask], E["x"][mask]
    ok = ~np.isnan(x); sid, c, z, x = sid[ok], c[ok], z[ok], x[ok]
    # 全銘柄の平均(シグナル×規模)
    g = sid * 3 + z
    mu = np.bincount(g, x, NS * 3) / np.maximum(np.bincount(g, minlength=NS * 3), 1)
    # 銘柄ごと
    k = sid.astype(np.int64) * NC + c
    uk, inv = np.unique(k, return_inverse=True)
    n = np.bincount(inv).astype(float)
    m = np.bincount(inv, x) / n
    pooled = np.bincount(inv, mu[g]) / n          # 同じ規模のイベントに対する全銘柄平均の、銘柄ごとの平均
    s_sid = (uk // NC).astype(int)
    # シグナルごとに、銘柄間の本当のばらつき tau2 を推定(モーメント法)。誤差分散は、銘柄内のばらつき / 件数
    dev = x - mu[g]
    within = np.bincount(inv, (dev - (m - pooled)[inv]) ** 2) / np.maximum(n - 1, 1)
    tau2 = np.zeros(NS); s2 = np.zeros(NS)
    for i in range(NS):
        sel = (s_sid == i) & (n >= 3)
        if sel.sum() < 10:
            continue
        d = (m - pooled)[sel]
        samp = (within[sel] / n[sel])
        tau2[i] = max(0.0, d.var() - samp.mean()); s2[i] = within[sel].mean()
    k_shrink = np.where(tau2[s_sid] > 0, s2[s_sid] / np.maximum(tau2[s_sid], 1e-9), 1e9)
    w = n / (n + k_shrink)
    est = pooled + w * (m - pooled)
    return dict(keys=uk, sid=s_sid, c=(uk % NC).astype(int), n=n, mean=m, pooled=pooled, est=est, w=w, tau2=tau2)


def validate(h, zsel=None):
    E = prep(h)
    tr = E["t"] < SPLIT
    F = fit(E, tr, zsel)
    # 後半の実績
    te = (~tr) & ~np.isnan(E["x"])
    if zsel is not None:
        te = te & (E["z"] == zsel)
    k_te = E["sid"][te].astype(np.int64) * NC + E["c"][te]
    pos = np.searchsorted(F["keys"], k_te)
    pos = np.clip(pos, 0, len(F["keys"]) - 1)
    hit = F["keys"][pos] == k_te
    xt = E["x"][te][hit]; p = pos[hit]
    # (シグナル, 銘柄)単位の後半平均
    nte = np.bincount(p, minlength=len(F["keys"])); mte = np.bincount(p, xt, len(F["keys"])) / np.maximum(nte, 1)
    use = (F["n"] >= 5) & (nte >= 3)
    out = {"対象(シグナル×銘柄)": int(use.sum()), "規模": zsel}
    for nm, v in (("銘柄の素の平均", F["mean"]), ("縮小推定", F["est"]), ("全銘柄平均のみ", F["pooled"])):
        out["相関:" + nm] = float(np.corrcoef(v[use], mte[use])[0, 1])
    # 銘柄ごとに、推定上位3シグナルを選んだときの後半実績
    res = {}
    stk = F["c"]
    for nm, v in (("銘柄の素の平均", F["mean"]), ("縮小推定", F["est"]), ("全銘柄平均のみ", F["pooled"]), ("無作為", np.random.default_rng(0).random(len(F["keys"])))):
        tot = cnt = 0.0
        order = np.lexsort((-v, stk))
        rank = np.zeros(len(order), int)
        s_sorted = stk[order]
        first = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
        rank[order] = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
        sel = use & (rank < 3)
        res[nm] = (float((mte[sel] * nte[sel]).sum() / nte[sel].sum()), int(nte[sel].sum()), int(sel.sum()))
    allm = float((mte[use] * nte[use]).sum() / nte[use].sum())
    out["上位3選択の後半平均bp"] = res; out["使えるペア全体の後半平均bp"] = allm
    return out


def stock_size(ci, E):
    """銘柄の規模: 直近2年のシグナル発生時点の規模の最頻値(なければ全期間)。"""
    m = (E["c"] == ci) & (E["t"] >= len(DATES) - 500)
    if m.sum() == 0:
        m = E["c"] == ci
    return int(np.bincount(E["z"][m], minlength=3).argmax())


def pooled_t(E, zsel, h):
    """同じ規模の全銘柄での、シグナルごとの平均のt値(同じ日・保有期間の重なりをブロックでまとめた値)。"""
    out = np.full(NS, np.nan)
    for i in range(NS):
        m = (E["sid"] == i) & (E["z"] == zsel) & ~np.isnan(E["x"])
        if m.sum() > 50:
            out[i] = bt.cluster_t(E["x"][m] / 1e4, E["t"][m], max(h, 5))
    return out


def report(code, h):
    code5 = code if len(code) == 5 else code + "0"
    ci = list(CODES).index(code5)
    E = prep(h)
    zs = stock_size(ci, E)
    F = fit(E, np.ones(len(E["t"]), bool), zs)
    pt = pooled_t(E, zs, h)
    sel = F["c"] == ci
    rows = []
    for j in np.flatnonzero(sel):
        i = F["sid"][j]
        m = (E["sid"] == i) & (E["c"] == ci) & (E["z"] == zs) & ~np.isnan(E["x"])
        x = E["x"][m]; t = E["t"][m]
        a, b = x[t < SPLIT], x[t >= SPLIT]
        last = int(E["t"][(E["sid"] == i) & (E["c"] == ci)].max())
        rows.append(dict(sig=NAMES[i], fam=FAMS[i], n=len(x), mean=x.mean(), est=F["est"][j], pooled=F["pooled"][j], pt=pt[i], w=F["w"][j],
                         win=(x > 0).mean(), t=x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan,
                         f=a.mean() if len(a) else np.nan, l=b.mean() if len(b) else np.nan, na=len(a), nb=len(b),
                         recent=DATES[last], tau=F["tau2"][i] ** 0.5, size=["小型", "中型", "大型"][zs]))
    return rows


def score(r):
    """信頼度(0〜5)。条件: 件数20以上 / 平均が有意(t≥2) / 前半・後半とも正 / 同じ規模の全銘柄でも有意に正(t≥2) / 補正後がコスト20bpを超える。"""
    c = [r["n"] >= 20, r["t"] >= 2, r["f"] > 0 and r["l"] > 0, r["pt"] >= 2, r["est"] > COST]
    return sum(bool(x) for x in c), c


if __name__ == "__main__":
    if sys.argv[1] == "--validate":
        for h in (5, 10, 20):
            for z in (None, 0, 1, 2):
                print(h, validate(h, z))
    else:
        h = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        rows = report(sys.argv[1], h)
        for r in rows:
            r["score"], r["c"] = score(r)
        big = sorted([r for r in rows if r["n"] >= 20], key=lambda r: (-r["score"], -r["est"]))
        print(f"規模: {rows[0]['size']}。" + f"{sys.argv[1]} 買いシグナル(翌日始値で買い、{h}日保有、同日・同規模の全銘柄平均との差bp)。件数20未満の{len(rows) - len(big)}シグナルは、件数不足で除外")
        print(f"{'シグナル':<30}{'信頼度':>6}{'件数':>5}{'平均':>8}{'補正後':>8}{'同規模全銘柄':>10}{'勝率':>6}{'前半':>8}{'後半':>8}  直近の発生")
        for r in big:
            print(f"{r['sig']:<30}{'★' * r['score'] + '☆' * (5 - r['score']):>8}{r['n']:>5}{r['mean']:>8.1f}{r['est']:>8.1f}{r['pooled']:>8.1f}(t={r['pt']:.1f}){r['win']:>6.0%}{r['f']:>8.1f}{r['l']:>8.1f}  {r['recent']}")
