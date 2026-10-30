"""結果 CSV から、単一ファイルの HTML レポート(report.html)を生成する。

使い方: python3 jp_backtest/build_report.py
J-Quants の元データは含めない(集計済みの統計量のみ)。
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def csv(name):
    return pd.read_csv(os.path.join(HERE, name))


def clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def rows(df, cols):
    return [{k: clean(r[c]) for k, c in cols.items()} for _, r in df.iterrows()]


def build_data():
    key = ["パターン", "方向", "グループ", "保有日数"]
    main = csv("results_main.csv")
    liq = main[main.区分 == "流動性分位"]
    a = liq[liq.期間 == "全体"].set_index(key)
    f = liq[liq.期間 == "前半"].set_index(key)["mean_bp"].rename("f")
    l = liq[liq.期間 == "後半"].set_index(key)["mean_bp"].rename("l")
    m = a.join(f).join(l).reset_index()
    meta = json.load(open(os.path.join(HERE, "report_data", "meta.json"), encoding="utf-8"))
    D = {"stocks": meta["stocks"], "pool": build_pool(), "dtop": build_dtop(), "secs": meta["secs"], "holds": meta["holds"], "main": rows(m, dict(p="パターン", d="方向", q="グループ", h="保有日数", n="n", m="mean_bp", md="median_bp",
                              net="net_bp", w="win", t="t", f="f", l="l"))}

    h3 = csv("results_h3.csv")
    D["h3"] = rows(h3, dict(pool="束ね", h="保有日数", dir="方向", n="銘柄数", r="順位相関", t="t", r2="流動性除去後の相関",
                            lo="探索期間の下位20pct銘柄の検証期間bp", hi="探索期間の上位20pct銘柄の検証期間bp"))
    h4 = csv("results_h4.csv")
    D["h4"] = {"tests": int(len(h4)), "sig": int((h4["差のt"].abs() >= 3.8).sum())}
    cells = csv("results_h4_cells.csv")
    cells = cells[cells.パターン.isin(["逆ハンマー", "首吊り線"]) & (cells.規模 == "小型(分位1-2)") & cells.保有日数.isin([1, 5])]
    D["h4cells"] = rows(cells, dict(dim="局面", v="値", p="パターン", h="保有日数", n="n", m="mean_bp", t="t"))

    mv = csv("results_micro_variants.csv")
    D["micro"] = rows(mv, dict(p="パターン", z="規模", v="変種", n="n", m="mean_bp", t="t"))
    D["price"] = rows(csv("results_micro_price.csv"), dict(p="パターン", b="株価帯", n="n", m="mean_bp", t="t"))
    D["sens"] = rows(csv("results_sensitivity.csv"), dict(c="設定", p="パターン", h="保有日数", n="n", m="mean_bp", t="t"))

    ex = csv("results_extra.csv")
    k2 = ["シグナル", "規模", "保有日数"]
    ea = ex[ex.期間 == "全体"].set_index(k2)
    ef = ex[ex.期間 == "前半"].set_index(k2)["mean_bp"].rename("f")
    el = ex[ex.期間 == "後半"].set_index(k2)["mean_bp"].rename("l")
    em = ea.join(ef).join(el).reset_index()
    D["extra"] = rows(em, dict(s="シグナル", z="規模", h="保有日数", n="n", m="mean_bp", net="net_bp", t="t", f="f", l="l"))

    D["robust"] = rows(csv("results_robust.csv"), dict(s="シグナル", z="規模", v="変種", c="条件", n="n", m="平均bp", md="中央値bp", t="t"))
    D["robustYear"] = rows(csv("results_robust_year.csv"), dict(s="シグナル", z="規模", y="年", n="n", m="平均bp"))

    mm = csv("results_margin_main.csv")
    mm = mm[(mm.区分 == "流動性分位") & (mm.期間 == "全体")]
    D["mmain"] = rows(mm, dict(p="パターン", d="方向", q="グループ", h="保有日数", n="n", m="mean_bp", t="t"))
    ms = csv("results_margin_short.csv")
    ms = ms[ms.シグナル.isin(MARGIN_SIGNALS) & ms.変種.isin(["A", "C", "D", "G"])]
    D["mshort"] = rows(ms, dict(u="対象", s="シグナル", z="規模", v="変種", n="n", m="平均bp", t="t", f="前半bp", l="後半bp"))
    D["myear"] = rows(csv("results_margin_year.csv"), dict(s="シグナル", y="年", n="n", m="平均bp"))
    for key, fname in [("cmain", "results_chart_main.csv"), ("cmm", "results_chart_margin_main.csv")]:
        cdf = csv(fname)
        cdf = cdf[(cdf.区分 == "流動性分位") & (cdf.期間 == "全体")]
        D[key] = rows(cdf, dict(p="パターン", d="方向", q="グループ", h="保有日数", n="n", m="mean_bp", md="median_bp", net="net_bp", w="win", t="t"))
    D["cmatch"] = rows(csv("results_chart_matched.csv"), dict(u="対象", s="シグナル", d="方向", z="規模", v="変種", n="n", r0="シグナル日の平均騰落率pct",
                                                              raw="素の超過bp", rt="素のt", adj="同じ騰落率との差bp", at="差のt"))
    D["cvar"] = rows(csv("results_chart_variants.csv"), dict(u="対象", s="シグナル", d="方向", z="規模", v="変種", n="n", m="平均bp", t="t", f="前半bp", l="後半bp"))
    for key, fname in [("tmain", "results_tech_main.csv"), ("tmm", "results_tech_margin_main.csv")]:
        tdf = csv(fname)
        tdf = tdf[(tdf.区分 == "流動性分位") & (tdf.期間 == "全体")]
        D[key] = rows(tdf, dict(p="パターン", d="方向", q="グループ", h="保有日数", n="n", m="mean_bp", md="median_bp", net="net_bp", w="win", t="t"))
    D["tmatch"] = rows(csv("results_tech_matched.csv"), dict(u="対象", k="基準日数", s="シグナル", d="方向", z="規模", v="変種", n="n", r0="シグナル日の平均騰落率pct",
                                                             raw="素の超過bp", rt="素のt", adj="同じ騰落率との差bp", at="差のt"))
    D["tcost"] = rows(csv("results_tech_cost.csv"), {"s": "シグナル", "d": "サイン方向", "z": "規模", "h": "保有", "n": "n", "g": "粗利bp", "tk": "呼値比率bp",
                                                      "c20": "コスト20bp固定_bp", "c1": "呼値1枚+貸株料_bp", "t1": "呼値1枚+貸株料_t", "c2": "呼値2枚+貸株料_bp"})
    D["mtf"] = build_mtf()
    D["wk"] = rows(csv("results_weekly_check.csv"), {"u": "対象", "s": "シグナル", "d": "方向", "z": "規模", "n": "n", "g": "粗利bp", "t": "t", "x3": "上位3週を除く平均bp",
                                                      "c20": "コスト20bp固定_bp", "c1": "呼値1枚+貸株料_bp", "t1": "呼値1枚+貸株料_t", "c2": "呼値2枚+貸株料_bp", "tk": "呼値比率bp", "fee": "貸株料bp"})
    D["wky"] = rows(csv("results_weekly_check_year.csv"), {"u": "対象", "s": "シグナル", "z": "規模", "y": "年", "n": "n", "m": "平均bp"})
    D["wsens"] = rows(csv("results_weekly_sens.csv"), {"u": "対象", "c": "設定", "n": "n", "m": "平均bp", "t": "t", "f": "前半bp", "l": "後半bp"})
    D["cvarW"] = rows(csv("results_chart_W_variants.csv"), {"u": "対象", "s": "シグナル", "d": "方向", "z": "規模", "v": "変種", "n": "n", "m": "平均bp", "t": "t", "f": "前半bp", "l": "後半bp"})
    D["zcost"] = rows(csv("results_zone_cost.csv"), {"s": "シグナル", "d": "サイン方向", "z": "規模", "h": "保有", "n": "n", "g": "粗利bp", "tk": "呼値比率bp", "c20": "コスト20bp固定_bp", "c1": "呼値1枚+貸株料_bp", "t1": "呼値1枚+貸株料_t", "c2": "呼値2枚+貸株料_bp"})
    D["conf"] = rows(csv("results_confluence.csv"), {"f": "分析", "tf": "上位足", "z": "規模", "h": "保有日数", "na": "一致n", "a": "一致bp", "nb": "逆行n", "b": "逆行bp", "d": "差bp", "t": "差のt"})
    D["confb"] = rows(csv("results_confluence_base.csv"), {"tf": "上位足", "z": "規模", "h": "保有日数", "u": "上向きbp", "d": "下向きbp", "diff": "差bp", "t": "差のt"})
    cm = csv("results_cost_model.csv")
    D["cost"] = rows(cm, {"s": "シグナル", "z": "規模", "h": "保有", "n": "n", "g": "粗利bp", "tk": "呼値比率bp", "fee": "貸株料bp",
                          "c20": "コスト20bp固定_bp", "c1": "呼値1枚+貸株料_bp", "t1": "呼値1枚+貸株料_t",
                          "c2": "呼値2枚+貸株料_bp", "t2": "呼値2枚+貸株料_t"})
    return D


MARGIN_SIGNALS = ["窓下げ2%超", "[探索]終値が高値圏(レンジ3%超)", "大陽線|出来高急増", "赤三兵|出来高急増", "首吊り線", "流れ星"]


import re

ZN = re.compile(r"\((長期|短期)・\d+本\)")


def zname(n):
    """新高値/安値のシグナル名から本数を外す(時間軸で本数が違うため)。"""
    return ZN.sub(r"(\1)", n)


def main_rows(df, zone=False):
    """流動性分位別の全体/前半/後半を、1 行にまとめた行のリスト。"""
    df = df[df.区分 == "流動性分位"].copy()
    df["パターン"] = df["パターン"].map(zname)
    key = ["パターン", "方向", "グループ", "保有日数"]
    a = df[df.期間 == "全体"].set_index(key)
    f = df[df.期間 == "前半"].set_index(key)["mean_bp"].rename("f")
    l = df[df.期間 == "後半"].set_index(key)["mean_bp"].rename("l")
    m = a.join(f).join(l).reset_index()
    return rows(m, dict(p="パターン", d="方向", q="グループ", h="保有日数", n="n", m="mean_bp", md="median_bp", net="net_bp", w="win", t="t", f="f", l="l"))


def match_rows(df):
    df = df.copy()
    if "基準日数" not in df.columns:
        df["基準日数"] = 1
    df["シグナル"] = df["シグナル"].map(zname)
    return rows(df, dict(u="対象", k="基準日数", s="シグナル", d="方向", z="規模", v="変種", n="n", r0="シグナル日の平均騰落率pct",
                         raw="素の超過bp", rt="素のt", adj="同じ騰落率との差bp", at="差のt"))


def hold_rows(df):
    df = df.copy()
    df["シグナル"] = df["シグナル"].map(zname)
    return rows(df, dict(u="対象", k="基準日数", s="シグナル", d="方向", z="規模", h="保有", n="n", raw="素の超過bp", rt="素のt", adj="同じ騰落率との差bp", at="差のt"))


ZPREF = [("zline", "線: "), ("zboth", "重なり: "), ("zvonly", "出来高のみ: "), ("zlonly", "線のみ: ")]


def split_zone(d):
    """抵抗帯の結果(main / mm / match)を、考え方ごとの分析(zone, zline, zboth, zvonly, zlonly)に、名前の接頭辞で分ける。"""
    out = {}
    for fam, pre in ZPREF:
        out[fam] = {"main": [r for r in d["main"] if r["p"].startswith(pre)], "mm": [r for r in d["mm"] if r["p"].startswith(pre)],
                    "match": [r for r in d["match"] if r["s"].startswith(pre)], "hold": [r for r in d.get("hold", []) if r["s"].startswith(pre)]}
    pres = tuple(x for _, x in ZPREF)
    out["zone"] = {"main": [r for r in d["main"] if not r["p"].startswith(pres)], "mm": [r for r in d["mm"] if not r["p"].startswith(pres)],
                   "match": [r for r in d["match"] if not r["s"].startswith(pres)], "hold": [r for r in d.get("hold", []) if not r["s"].startswith(pres)]}
    return out


def build_dtop():
    """週足ダブルトップ(売り)の、長期データでの実コストと相場局面別(results_dtop_regime.csv)。行は [対象, 保有週, 規模, 区分, n, 粗利, 粗利t, 呼値, 貸株料, 純1tick, 純1tick_t, 純2tick, 勝率]。"""
    f = os.path.join(HERE, "results_dtop_regime.csv")
    if not os.path.exists(f):
        return []
    d = csv("results_dtop_regime.csv")
    return [[r.対象, int(r.保有週), r.規模, r.区分, int(r.n), round(r.粗利bp, 1), round(r.粗利t, 1), round(r.呼値bp, 1), round(r.貸株料bp, 1), round(r.純1tick_bp, 1), round(r.純1tick_t, 1), round(r.純2tick_bp, 1), round(r.勝率, 2)] for r in d.itertuples()]


def build_pool():
    """銘柄別シグナルタブの、仲間の選び方の検証表(results_pool_*.csv)。各行は [名前, 平均, 最小, 最大, 追加の値...]。"""
    def agg(df, key, val="corr"):
        g = df.groupby(key)[val].agg(["mean", "min", "max"])
        return g
    out = {}
    L1 = [("own", "自身の平均だけ"), ("size1", "規模で分けない全銘柄の平均"), ("size2", "規模2分割(大型とそれ以外)の全銘柄の平均"), ("size3", "規模3分割(小型・中型・大型)の全銘柄の平均"),
          ("sec", "同じ業種(33業種)の銘柄"), ("secz", "同じ業種 × 同じ規模"), ("corr15", "相関上位15銘柄(規模問わず)"), ("corr60", "相関上位60銘柄(規模問わず)"),
          ("corr200", "相関上位200銘柄(規模問わず)"), ("corrz15", "相関上位15銘柄(同じ規模)"), ("corrz60", "相関上位60銘柄(同じ規模)"), ("corrz100", "相関上位100銘柄(同じ規模)"),
          ("corrz200", "相関上位200銘柄(同じ規模)"), ("secc15", "同じ業種の中で、相関上位15銘柄")]
    if os.path.exists(os.path.join(HERE, "results_pool_compare.csv")):
        g = agg(csv("results_pool_compare.csv"), "method")
        out["compare"] = [[lab, float(g.loc[k, "mean"]), float(g.loc[k, "min"]), float(g.loc[k, "max"])] for k, lab in L1 if k in g.index]
    if os.path.exists(os.path.join(HERE, "results_pool_window.csv")):
        w = csv("results_pool_window.csv"); g = w.groupby(["window", "method"])["corr"].mean().unstack()
        out["window"] = [[lab, float(g.loc[k, "corr200"]), float(g.loc[k, "corrz200"])] for k, lab in ((0, "学習期間の全体"), (520, "直近10年(520週)"), (260, "直近5年(260週)"), (156, "直近3年(156週)"), (104, "直近2年(104週)")) if k in g.index]
    if os.path.exists(os.path.join(HERE, "results_pool_corrtype.csv")):
        c = csv("results_pool_corrtype.csv")
        out["corrtype"] = [[r.variant, float(r.avg), float(r.mn), float(r.mx)] for r in c.itertuples()]
    if os.path.exists(os.path.join(HERE, "results_pool_redo.csv")):
        r = csv("results_pool_redo.csv"); g = r.groupby(["method", "set"])["corr"].agg(["mean", "min", "max"]).unstack()
        rows_ = [[m, float(g.loc[m, ("mean", "選択")]), float(g.loc[m, ("mean", "未使用")]), float(g.loc[m, ("min", "未使用")]), float(g.loc[m, ("max", "未使用")])] for m in g.index]
        out["redo"] = sorted(rows_, key=lambda x: -x[2])
    if os.path.exists(os.path.join(HERE, "results_pool_resp_z.csv")):
        r = csv("results_pool_resp_z.csv"); g = r.groupby(["method", "set"])["corr"].agg(["mean", "min", "max"]).unstack()
        out["resp"] = [[k, float(g.loc[k, ("mean", "選択")]), float(g.loc[k, ("mean", "未使用")]), float(g.loc[k, ("min", "未使用")]), float(g.loc[k, ("max", "未使用")])] for k in ("corr", "resp", "both")]
    for side in ("buy", "sell"):
        f = f"results_pool_{side}.csv"
        if os.path.exists(os.path.join(HERE, f)):
            r = csv(f); g = r.groupby("method").agg(corr=("corr", "mean"), mn=("corr", "min"), mx=("corr", "max"), top3=("top3", "mean"), allmean=("allmean", "mean"))
            out[side] = [[k, float(g.loc[k, "corr"]), float(g.loc[k, "mn"]), float(g.loc[k, "mx"]), float(g.loc[k, "top3"]), float(g.loc[k, "allmean"])] for k in g.index]
    return out


def build_mtf():
    """{分析: {時間軸: {main, mm, match}}}。日足の既存結果と、週足・月足(mtf_run.py)、長期(WL/ML, long_run.py)の結果をそろえる。"""
    files = {
        ("candle", "D"): ("results_main.csv", "results_margin_main.csv", "results_candle_D_matched.csv"),
        ("chart", "D"): ("results_chart_main.csv", "results_chart_margin_main.csv", "results_chart_matched.csv"),
        ("tech", "D"): ("results_tech_main.csv", "results_tech_margin_main.csv", "results_tech_matched.csv"),
    }
    for fam in ("candle", "chart", "tech", "zone"):
        for tf in ("D", "W", "M", "WL", "ML"):
            files.setdefault((fam, tf), (f"results_{fam}_{tf}_main.csv", f"results_{fam}_{tf}_margin_main.csv", f"results_{fam}_{tf}_matched.csv"))
    out = {}
    for (fam, tf), (fm, fmm, fx) in files.items():
        if not all(os.path.exists(os.path.join(HERE, x)) for x in (fm, fmm, fx)):
            continue
        d = {"main": main_rows(csv(fm)), "mm": main_rows(csv(fmm)), "match": match_rows(csv(fx))}
        fh = os.path.join(HERE, f"results_{fam}_{tf}_hold.csv")
        if os.path.exists(fh):
            d["hold"] = hold_rows(csv(os.path.basename(fh)))
        if fam == "zone":
            for f2, dd in split_zone(d).items():
                out.setdefault(f2, {})[tf] = dd
        else:
            out.setdefault(fam, {})[tf] = d
    return out


if __name__ == "__main__":
    D = build_data()
    tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/{}", json.dumps(D, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    out = os.path.join(HERE, "report.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
