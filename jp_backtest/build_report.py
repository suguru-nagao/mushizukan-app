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
    D = {"main": rows(m, dict(p="パターン", d="方向", q="グループ", h="保有日数", n="n", m="mean_bp", md="median_bp",
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
    cm = csv("results_cost_model.csv")
    D["cost"] = rows(cm, {"s": "シグナル", "z": "規模", "h": "保有", "n": "n", "g": "粗利bp", "tk": "呼値比率bp", "fee": "貸株料bp",
                          "c20": "コスト20bp固定_bp", "c1": "呼値1枚+貸株料_bp", "t1": "呼値1枚+貸株料_t",
                          "c2": "呼値2枚+貸株料_bp", "t2": "呼値2枚+貸株料_t"})
    return D


MARGIN_SIGNALS = ["窓下げ2%超", "[探索]終値が高値圏(レンジ3%超)", "大陽線|出来高急増", "赤三兵|出来高急増", "首吊り線", "流れ星"]


if __name__ == "__main__":
    D = build_data()
    tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/{}", json.dumps(D, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    out = os.path.join(HERE, "report.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
