"""別のウェブアプリに渡すため、検証結果をJSONで export/ に書き出す。
  python3 export_for_app.py            # 銘柄別シグナル・到達銘柄(キャッシュ)・調査結果・要約を書き出す
  python3 export_for_app.py --zones    # 到達銘柄を日足から再計算してから書き出す(数分かかる)
出力(export/):
  manifest.json       生成時刻、最新の確定週、ファイル一覧、スキーマの説明
  stock_signals.json  銘柄別シグナル。{code4: {name,sector,size,short_history, long:{hold:{grade,k,cost_bp,lit:[...]}}, short:{...}}}
  zone_touch.json     直近で抵抗帯・支持帯に到達した銘柄(貸借・信用倍率つき)
  research.json       リバウンドする銘柄の違いなど、報告書作成中の調査結果(表)
  summary.json        検証結果の要約(結論・コスト・注意点)
  zone_levels.json    銘柄ごとの、出来高最多の抵抗帯・支持帯(支持線・抵抗線のように)と、跳ね返された回数(zone_levels.py で作る)
"""
import datetime, glob, gzip, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "report_data"); OUT = os.path.join(HERE, "export"); os.makedirs(OUT, exist_ok=True)
HOLDS = [1, 2, 4, 6, 12]


def dump(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"), default=lambda o: None if o != o else float(o))
    if name == "stock_signals.json":      # 大きいので gzip も出す(git にはこちらだけ入れる)
        with gzip.open(p + ".gz", "wt", encoding="utf-8", compresslevel=9) as g:
            json.dump(obj, g, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(p)


def cost_bp(side, h):          # report_template.html の COST と同じ
    return 20.0 if side == "long" else 20 + (0.0110 + 0.020) * 7 * h / 365 * 1e4


def load_pref(pfx, h, k):
    return json.load(open(os.path.join(RD, f"{pfx}_{h}_{k}.json"), encoding="utf-8"))


def stock_signals():
    meta = json.load(open(os.path.join(RD, "meta.json"), encoding="utf-8"))
    mL = json.load(open(os.path.join(RD, "wrank_meta.json"), encoding="utf-8"))
    mS = json.load(open(os.path.join(RD, "wrank_s_meta.json"), encoding="utf-8"))
    info = {s[4]: s for s in meta["stocks"]}
    codes = mL["codes"]; last = len(mL["dates"]) - 1
    out = {}; cidx = {c: i for i, c in enumerate(codes)}
    for side, pfx, mt in (("long", "wrank", mL), ("short", "wrank_s", mS)):
        for h in HOLDS:
            cost = cost_bp(side, h)
            for k in range(10):
                for code, rows in load_pref(pfx, h, k).items():
                    c4 = code[:4]; st = info.get(code)
                    e = out.setdefault(c4, dict(code=c4, name=st[1] if st else None, sector=st[3] if st else None,
                                                size=["小型", "中型", "大型"][mL["size"][cidx[code]]] if st else None, long={}, short={}))
                    fb = bool(mL["fb"][cidx[code]])
                    e["short_history"] = fb
                    lit = []
                    for r in rows:
                        # r = [sig, n, own, est, size_mean, peers_n, peers_t, win%, first_half, second_half, last_t, score, own_t]
                        if r[10] < 0 or last - r[10] > 2 or r[1] < (5 if fb else 10):
                            continue
                        ev = r[3]
                        lit.append(dict(signal=mt["sigs"][r[0]][0], family=mt["sigs"][r[0]][1], week=mt["dates"][r[10]], n=r[1],
                                        estimate_bp=ev, own_mean_bp=r[2], win_pct=r[7], over_cost=bool(ev > cost)))
                    kk = sum(x["over_cost"] for x in lit)
                    grade = None if fb else ("強" if kk >= 3 else "中" if kk == 2 else "弱(参考)" if kk == 1 else "なし")
                    e[side][str(h)] = dict(grade=grade, k=kk, cost_bp=round(cost, 1), n_lit=len(lit), lit=sorted(lit, key=lambda x: -x["estimate_bp"])[:8])
    return out, mL["dates"][last]


def zone_touch():
    f = os.path.join(HERE, "cache", "zone_touch_latest.pkl")
    if "--zones" in sys.argv or not os.path.exists(f):
        import backtest as bt
        from zones import SR_PARAMS, TF_PARAMS, detect_zones
        P = bt.load_panel(); U = bt.load_margin(P)
        sigs = detect_zones(P["H"], P["L"], P["C"], P["Vo"], *TF_PARAMS["D"], sr=SR_PARAMS["D"])
        T = len(P["dates"]); rows = []
        for nm in ("抵抗帯に到達(跳ね返される)", "支持帯に到達(跳ね返す)"):
            for back in (0, 1):
                for c in np.nonzero(sigs[nm][1][T - 1 - back])[0]:
                    rows.append((nm[:3], P["dates"][T - 1 - back].strftime("%m-%d"), P["codes"][c], bool(U[T - 1 - back, c]), P["mkt"][c], float(P["Craw"][T - 1 - back, c]), float(P["Va"][T - 1 - back, c])))
        pd.DataFrame(rows, columns=["帯", "日", "code", "貸借", "mkt", "price", "va"]).to_pickle(f)
    d = pd.read_pickle(f)
    meta = json.load(open(os.path.join(RD, "meta.json"), encoding="utf-8")); info = {s[4]: s for s in meta["stocks"]}
    mg = os.path.join(HERE, "cache", "zone_touch_margin.csv")
    mgd = pd.read_csv(mg, dtype={"code": str}).drop_duplicates("code").set_index("code") if os.path.exists(mg) else pd.DataFrame()
    recs = []
    for _, r in d.iterrows():
        st = info.get(r.code); m = mgd.loc[r.code] if r.code in mgd.index else None
        recs.append(dict(code=r.code[:4], name=st[1] if st else None, sector=st[3] if st else None, zone="抵抗帯" if r["帯"] == "抵抗帯" else "支持帯",
                         direction_hint="売り(下がる方向)" if r["帯"] == "抵抗帯" else "買い(上がる方向)", date=r["日"], margin_stock=bool(r["貸借"]), price=r.price,
                         turnover_jpy=r.va, margin_long_k=None if m is None else float(m["買残千株"]), margin_short_k=None if m is None else float(m["売残千株"]),
                         margin_ratio=None if m is None else float(m["倍率"]), margin_asof=None if m is None else m["基準日"]))
    return dict(note="日足で直近2営業日に点灯した価格帯別出来高の抵抗帯・支持帯への到達。検証では優位性があるのは抵抗帯(売り)のみで、支持帯(買い)は優位性なし。信用倍率は株探の直近週(抵抗帯・貸借銘柄のみ取得)。", items=recs)


def research():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results_zone_char*.csv"))):
        k = os.path.basename(f)[8:-4]
        try:
            out[k] = pd.read_csv(f).round(3).to_dict("records")
        except Exception:
            pass
    return dict(note="価格帯別出来高の最多付近で跳ね返る銘柄の違いの調査(報告書の抵抗帯タブ、REPORT.md 追加検証26・27)。表の列名は各CSVのまま。", tables=out)


def summary():
    disp = {}
    for side in ("buy", "sell"):
        for h in HOLDS:
            f = os.path.join(HERE, f"results_display2_{side}_{h}.csv")
            if os.path.exists(f):
                disp[f"{side}_{h}w"] = pd.read_csv(f).round(2).to_dict("records")
    return dict(
        generated_from="jp_backtest/REPORT.md と各 results_*.csv",
        method=["シグナルの引け後に判定し、翌営業日の始値で入り、保有期間後の始値で出る", "同日・同流動性の全銘柄平均と、同じ騰落率の日の平均を引いた超過リターン", "規模別(小型・中型・大型)に分析", "長期データ(Yahoo 2000〜2026)は生存者バイアスあり。外れ値は除去・クリップ済み"],
        cost_bp={"買い(現物・楽天ゼロコース)": 20, "売り(制度信用6か月)": {f"{h}週": round(cost_bp("short", h), 1) for h in HOLDS}, "含まない": ["逆日歩", "貸株枠の制約"]},
        findings=[
            "銘柄別推定は、相関の高い同規模の上位200銘柄(仲間)の平均が最良。業種のみ・自身の平均・規模の平均より当たる",
            "点灯シグナルのうち推定がコストを超える個数で判定: 0=なし 1=弱 2=中 3以上=強",
            "ロングは保有が長いほど(4〜12週)効き、ショートは短いほど(1〜2週)効く",
            "コスト前の実績(後半検証の4設定の最小〜最大): 買い4週 3個以上 +36〜+50bp、6週 +57〜+94、12週 +89〜+157。売り1週 3個以上 +53〜+60、2週 +27〜+50、4週 +24〜+58。売りは貸株料・配当調整金を引くとほぼ消える",
            "週足ダブルトップ(売り)は8週で粗利+79bp(t 4.1)、1ティックと貸株料を引くと+42bp(t 2.2)。26年中7年はマイナス",
            "価格帯別出来高の最多付近: 抵抗帯(売り)は5日後で勝率約55%、平均+20bp前後。支持帯(買い)は勝率約46%で優位性なし",
            "回転率(売買の回転の遅さ)との関係は、外れ値を抑えると消える。極端に遅い銘柄で抵抗帯の勝率が数ポイント高い程度",
            "信用倍率が高い銘柄ほど抵抗帯で跳ね返る傾向(+38bp, t 3.0)は、現在値による近似で、銘柄単位では弱い。点時点の信用残が必要",
            "履歴が短く仲間で推定できない銘柄は、規模平均では予測力がなく、既定では判定を出さない"],
        cautions=["過去の統計であり、将来を保証しない", "勝率は50〜58%程度で、1銘柄ごとのぶれは大きい", "売りは取引コスト(貸株料・配当調整金・逆日歩)で優位性がほぼ消える"],
        display_logic_validation=disp)


def main():
    ss, last = stock_signals()
    sizes = {"stock_signals.json": dump("stock_signals.json", ss), "zone_touch.json": dump("zone_touch.json", zone_touch()),
             "research.json": dump("research.json", research()), "summary.json": dump("summary.json", summary()),
             "zone_levels.json": dump("zone_levels.json", json.load(open(os.path.join(HERE, "cache", "zone_levels.json"), encoding="utf-8")))}
    dump("manifest.json", dict(generated_at=datetime.datetime.now().isoformat(timespec="seconds"), last_confirmed_week=last, holds_weeks=HOLDS, files=sizes,
                               schema=dict(stock_signals="{code4: {name, sector, size, short_history, long|short: {hold_weeks: {grade, k, cost_bp, lit:[{signal,family,week,n,estimate_bp,own_mean_bp,win_pct,over_cost}]}}}}。grade は short_history=true の銘柄では null(判定なし)",
                                           zone_touch="items[{code,name,zone,direction_hint,date,margin_stock,price,turnover_jpy,margin_long_k,margin_short_k,margin_ratio,margin_asof}]",
                                           zone_levels="stocks{code4: {close, resistance|support: {low, high, mid, dist_pct, touches, rejected_from_below, broken_up, held_from_above, broken_down, pending, last_bounce} | null}}。低・高は帯の下端・上端(円)、dist_pctは現在値からの距離(%)",
                                           units="estimate_bp / own_mean_bp / cost_bp は bp(1bp=0.01%)")))
    print("書き出し:", sizes, "最新の確定週", last)


if __name__ == "__main__":
    main()
