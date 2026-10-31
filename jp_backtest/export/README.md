# 別のウェブアプリに渡す検証結果(JSON)

`python3 jp_backtest/export_for_app.py` で再生成します(到達銘柄の再計算は `--zones`、数分かかります)。
中身と形式は `manifest.json` の `schema` を参照してください。

| ファイル | 内容 |
|---|---|
| `manifest.json` | 生成時刻、最新の確定週、スキーマ |
| `stock_signals.json.gz` | 銘柄別シグナル(約3,700銘柄×買い/売り×保有1・2・4・6・12週)。展開: `gunzip -k stock_signals.json.gz` |
| `zone_touch.json` | 直近で抵抗帯・支持帯に到達した銘柄(信用倍率つき) |
| `research.json` | 報告書作成中の調査結果の表 |
| `zone_levels.json` | 銘柄ごとの、出来高最多の抵抗帯・支持帯(支持線・抵抗線のように)と、過去250日に触れた回数・跳ね返された回数。`zone_levels.py` で作る |
| `summary.json` | 検証結果の要約(結論・コスト・注意点) |

読み込み例(JavaScript / Node):

```js
const zlib = require("zlib"), fs = require("fs");
const ss = JSON.parse(zlib.gunzipSync(fs.readFileSync("export/stock_signals.json.gz")));
const toyota = ss["7203"];               // 銘柄コード4桁
console.log(toyota.long["4"].grade, toyota.short["2"].lit);
```

注意: `grade` が null の銘柄は、履歴が短く判定を出せない銘柄です。`estimate_bp` はコスト前、`cost_bp` を引いた値が見込みです。過去の統計であり、将来を保証しません。
