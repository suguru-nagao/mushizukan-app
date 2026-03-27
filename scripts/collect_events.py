#!/usr/bin/env python3
"""
kids_event_collector
────────────────────
電車・生き物・ポケモン・車 テーマの子ども向けイベントを
複数サイトからスクレイピングし、SQLite に保存後 JSON を出力する。

GitHub Actions で毎朝 06:00 JST に実行することを想定。
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── タイムゾーン・定数 ───────────────────────────────────────────
JST = timezone(timedelta(hours=9))
TODAY = datetime.now(JST).date()
FUTURE_LIMIT = TODAY + timedelta(days=30)

DB_PATH = os.environ.get("DB_PATH", "events.db")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "public/events.json")
GEMINI_API_KEY = os.environ.get("EXPO_PUBLIC_GEMINI_API_KEY", "")
EVENTBRITE_API_KEY = os.environ.get("EVENTBRITE_API_KEY", "")
MAX_LLM_CALLS = int(os.environ.get("MAX_LLM_CALLS", "20"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── テーマ定義 ───────────────────────────────────────────────────
THEME_KEYWORDS: dict[str, list[str]] = {
    "train": [
        "鉄道", "電車", "新幹線", "車両基地", "列車", "SL", "トロッコ",
        "ミニ電車", "JR", "東急", "小田急", "相鉄", "京急", "西武", "東武",
        "モノレール", "路面電車", "乗車体験", "鉄道博物館",
    ],
    "creature": [
        "昆虫", "生き物", "自然観察", "動物", "植物", "水族館", "動物園",
        "博物館", "カブトムシ", "チョウ", "クワガタ", "化石", "恐竜",
        "魚", "鳥", "虫", "標本", "生物", "ビオトープ", "自然",
    ],
    "pokemon": [
        "ポケモン", "Pokemon", "Pokémon", "ポケモンカード",
        "ポケモンセンター", "ポケカ", "ポケモンGO",
    ],
    "car": [
        "自動車", "ミニカー", "モーターショー", "キッズカー",
        "トヨタ", "ホンダ", "日産", "マツダ", "スバル", "EV体験",
        "ドライブ体験", "カーレース",
    ],
}

# 子ども向けスコア
KID_SCORE_MAP: dict[str, int] = {
    "親子": 2, "キッズ": 2, "子ども": 2, "子供": 2, "ファミリー": 2,
    "こども": 2, "幼児": 2, "小学生": 1, "未就学": 2,
    "体験": 1, "ワークショップ": 1, "見学": 1, "乗車": 1,
    "見学会": 2, "体験会": 2, "乗車体験": 2, "撮影会": 1,
    "遊び": 1, "工作": 1, "実験": 1, "教室": 1,
    "フェスタ": 1, "フェア": 1, "まつり": 1, "祭": 1,
    "展示": 1, "特別展": 1, "企画展": 1, "春休み": 1,
    "ツアー": 1, "探検": 1, "スタンプラリー": 2, "乗り放題": 1,
    "開放": 1, "公開": 1, "お披露目": 1, "記念": 1,
}
KID_PENALTY_MAP: dict[str, int] = {
    "セミナー": -2, "ビジネス": -2, "株主": -2, "採用": -2,
    "投資": -2, "転職": -2, "就活": -2, "企業向け": -2,
    "ワイン": -3, "アルコール": -3, "酒": -2, "Beer": -2, "ビール": -2,
    "夜間": -2, "深夜": -3, "成人": -2, "大人限定": -3,
}
EXCLUDE_KEYWORDS = [
    "予算", "国会", "議会", "法案", "規制", "人事異動", "辞任", "就任",
    "株価", "決算", "経済指標", "金利", "為替", "訃報", "死亡",
    "逮捕", "起訴", "裁判", "地震", "台風", "災害", "警報",
    "補助金", "助成金", "税制", "料金改定",
]

# 都道府県・市区町村パターン
PREF_RE = re.compile(r"(東京都|北海道|(?:京都|大阪)府|.{2,3}県)")
ADDR_RE = re.compile(
    r"((?:東京都|北海道|(?:京都|大阪)府|.{2,3}県)\s*[^\s。、\n]{1,8}(?:市|区|町|村))"
)
DATE_RE = re.compile(r"(\d{4})[年/](\d{1,2})[月/](\d{1,2})日?")
DATE_SHORT_RE = re.compile(r"(\d{1,2})月(\d{1,2})日")


# ── DB 操作 ─────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id               TEXT PRIMARY KEY,
            title            TEXT NOT NULL,
            description      TEXT,
            location_raw     TEXT,
            prefecture       TEXT,
            city             TEXT,
            date_start       TEXT NOT NULL,
            date_end         TEXT,
            source           TEXT,
            source_url       TEXT,
            fetched_at       TEXT,
            is_kid_friendly  BOOLEAN DEFAULT 1,
            is_train_related BOOLEAN DEFAULT 0,
            theme_id         TEXT DEFAULT 'train'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON events(date_start)")
    conn.commit()


def load_existing_json(conn: sqlite3.Connection, path: str) -> None:
    """既存の events.json を DB に読み込んで差分取得に利用する"""
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for e in data.get("events", []):
            conn.execute(
                """INSERT OR IGNORE INTO events
                   (id,title,description,location_raw,prefecture,city,
                    date_start,date_end,source,source_url,fetched_at,
                    is_kid_friendly,is_train_related,theme_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    e.get("id", ""), e.get("title", ""), e.get("description", ""),
                    e.get("location_raw", ""), e.get("prefecture", ""), e.get("city", ""),
                    e.get("date", ""), e.get("date_end"), e.get("source", ""),
                    e.get("url", ""), e.get("fetched_at", ""),
                    True, e.get("theme_id", "train") == "train",
                    e.get("theme_id", "train"),
                ),
            )
        conn.commit()
        log.info("既存JSON読み込み完了")
    except Exception as ex:
        log.warning(f"既存JSON読み込み失敗: {ex}")


def event_exists(conn: sqlite3.Connection, event_id: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone())


def insert_event(conn: sqlite3.Connection, ev: dict) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO events
           (id,title,description,location_raw,prefecture,city,
            date_start,date_end,source,source_url,fetched_at,
            is_kid_friendly,is_train_related,theme_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ev["id"], ev["title"], ev.get("description", ""),
            ev.get("location_raw", ""), ev.get("prefecture"), ev.get("city"),
            ev["date_start"], ev.get("date_end"),
            ev.get("source", ""), ev.get("source_url", ""),
            datetime.now(JST).isoformat(),
            ev.get("is_kid_friendly", True), ev.get("is_train_related", False),
            ev.get("theme_id", "train"),
        ),
    )


def cleanup_old(conn: sqlite3.Connection) -> None:
    cutoff = (TODAY - timedelta(days=1)).isoformat()
    r = conn.execute("DELETE FROM events WHERE date_start < ?", (cutoff,))
    conn.commit()
    if r.rowcount:
        log.info(f"期限切れイベント削除: {r.rowcount}件")


# ── ユーティリティ ───────────────────────────────────────────────
def make_id(title: str, date_start: str, location_raw: str) -> str:
    raw = f"{title}|{date_start}|{location_raw}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def parse_date(text: str) -> Optional[str]:
    """文字列から YYYY-MM-DD を抽出する"""
    if not text:
        return None
    t = text.strip()
    m = DATE_RE.search(t)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    m2 = DATE_SHORT_RE.search(t)
    if m2:
        return f"{TODAY.year}-{m2.group(1).zfill(2)}-{m2.group(2).zfill(2)}"
    # ISO形式
    m3 = re.search(r"(\d{4}-\d{2}-\d{2})", t)
    if m3:
        return m3.group(1)
    return None


def is_future(date_str: str) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return TODAY <= d <= FUTURE_LIMIT
    except Exception:
        return False


def extract_location(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """(location_raw, prefecture, city)"""
    location_raw = text.strip()[:60]
    pref: Optional[str] = None
    city: Optional[str] = None

    m = ADDR_RE.search(text)
    if m:
        addr = m.group(1).replace(" ", "")
        pm = PREF_RE.search(addr)
        if pm:
            pref = pm.group(1)
            rest = addr[pm.end():]
            cm = re.search(r"([^\s]{2,8}(?:市|区|町|村))", rest)
            if cm:
                city = cm.group(1)
    elif PREF_RE.search(text):
        pref = PREF_RE.search(text).group(1)  # type: ignore

    return location_raw, pref, city


def matches_theme(theme_id: str, title: str, desc: str) -> bool:
    text = f"{title} {desc}"
    return any(kw in text for kw in THEME_KEYWORDS.get(theme_id, []))


def kid_score(title: str, desc: str) -> int:
    text = f"{title} {desc}"
    if any(kw in text for kw in EXCLUDE_KEYWORDS):
        return -99
    score = 0
    for kw, pts in KID_SCORE_MAP.items():
        if kw in text:
            score += pts
    for kw, pts in KID_PENALTY_MAP.items():
        if kw in text:
            score += pts
    return score


def fetch_html(url: str, retries: int = 2) -> Optional[BeautifulSoup]:
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            if attempt == retries:
                log.warning(f"取得失敗 {url}: {e}")
            else:
                time.sleep(2 ** attempt)
    return None


# ── LLM 補完（Gemini） ───────────────────────────────────────────
_llm_calls = 0


def call_gemini_judge(title: str, desc: str) -> dict:
    """子ども向け判定 + 説明文生成（曖昧なケースのみ使用）"""
    global _llm_calls
    if not GEMINI_API_KEY or _llm_calls >= MAX_LLM_CALLS:
        return {"is_kid_friendly": False, "description": desc}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
    )
    prompt = (
        f"以下のイベントが「10歳未満の子どもと親が楽しめるイベント」かどうか判定してください。\n"
        f"タイトル: {title}\n説明: {desc or '(なし)'}\n\n"
        "JSONで回答（コードブロック不要）:\n"
        '{"is_kid_friendly": true/false, "description_ja": "子ども向け説明（40文字以内）"}'
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 150},
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        result = json.loads(text)
        _llm_calls += 1
        return {
            "is_kid_friendly": result.get("is_kid_friendly", False),
            "description": result.get("description_ja", desc),
        }
    except Exception as e:
        log.warning(f"Gemini判定失敗: {e}")
        return {"is_kid_friendly": False, "description": desc}


# ── スクレイパー ─────────────────────────────────────────────────

def scrape_ikoyo(themes: list[str]) -> list[dict]:
    """
    いこーよ — 子ども向けイベント検索
    セレクタ（2026年3月時点確認済み）:
      カード: div.card__link--index
      タイトル: a.card__header--index h2.heading--h3
      日付: a.card__content--index div.card__info.p-index-rating
      場所: a.card__header--index div.card__info span.ellipsis
      URL: a.card__header--index[href]
    """
    base = "https://iko-yo.net"
    kw_map: dict[str, list[str]] = {
        "train":    ["鉄道", "電車"],
        "creature": ["昆虫", "自然観察", "動物"],
        "pokemon":  ["ポケモン"],
        "car":      ["自動車", "乗り物"],
    }
    keywords: list[str] = []
    for t in themes:
        keywords.extend(kw_map.get(t, []))
    keywords = list(dict.fromkeys(keywords))  # 重複除去

    results: list[dict] = []
    seen_urls: set[str] = set()

    for kw in keywords:
        for page in range(1, 4):
            url = f"{base}/events?keyword={kw}&page={page}"
            soup = fetch_html(url)
            if not soup:
                break

            cards = soup.select("div.card__link--index")
            if not cards:
                log.debug(f"いこーよ: カード0件 kw={kw} page={page}")
                break

            found_on_page = 0
            for card in cards:
                try:
                    header_a = card.select_one("a.card__header--index")
                    content_a = card.select_one("a.card__content--index")
                    if not header_a or not content_a:
                        continue

                    # タイトル（ヘッダー内 h2 を優先、なければ content 内）
                    title_el = header_a.select_one("h2.heading--h3")
                    if not title_el:
                        title_el = content_a.select_one("div.card__title")
                    title = title_el.get_text(strip=True) if title_el else ""
                    if not title:
                        continue

                    # URL
                    href = header_a.get("href", "")
                    src_url = (base + href) if href.startswith("/") else href
                    if src_url in seen_urls:
                        continue
                    seen_urls.add(src_url)

                    # 日付（div.card__info.p-index-rating にカレンダーアイコン+日付）
                    date_el = content_a.select_one("div.card__info.p-index-rating")
                    date_start = parse_date(date_el.get_text() if date_el else "")
                    if not date_start:
                        continue

                    # 場所（ヘッダー内 div.card__info > span.ellipsis）
                    place_el = header_a.select_one("div.card__info span.ellipsis")
                    location_raw = place_el.get_text(strip=True) if place_el else ""
                    # 「都道府県市区 / カテゴリ」の形式なので / 以前だけ使う
                    if "/" in location_raw:
                        location_raw = location_raw.split("/")[0].strip()
                    _, pref, city = extract_location(location_raw)

                    results.append({
                        "title": title,
                        "description": "",
                        "location_raw": location_raw,
                        "prefecture": pref,
                        "city": city,
                        "date_start": date_start,
                        "source": "いこーよ",
                        "source_url": src_url,
                    })
                    found_on_page += 1
                except Exception as e:
                    log.debug(f"いこーよ カード解析エラー: {e}")

            log.debug(f"いこーよ kw={kw} page={page}: {found_on_page}件")
            if found_on_page == 0:
                break
            time.sleep(1)

    log.info(f"いこーよ: {len(results)}件取得")
    return results


def _fetch_tetsudo_event(url: str, fallback_title: str) -> Optional[dict]:
    """鉄道コム: 個別イベントページから情報を取得"""
    soup = fetch_html(url)
    if not soup:
        return None

    # ── 日付 ─────────────────────────────────────────────────────
    # ① ページタイトル「タイトル（2026年3月27日） - 鉄道コム」
    page_title = soup.title.get_text() if soup.title else ""
    date_start = parse_date(page_title)

    if not date_start:
        # ② .event-period / ul.period から取得
        for sel in [".event-period", "ul.period"]:
            period = soup.select_one(sel)
            if period:
                date_start = parse_date(period.get_text())
                if date_start:
                    break

    if not date_start:
        return None

    # ── タイトル ─────────────────────────────────────────────────
    # ページタイトルから「（日付）」と「 - 鉄道コム」を除去
    title = re.sub(r"（[^）]*）", "", page_title).replace(" - 鉄道コム", "").strip()
    if not title:
        title = fallback_title

    # ── 説明文（本文の最初の段落）────────────────────────────────
    desc = ""
    # エントリ本文を探す（entry-body, .event-detail, article など）
    body_el = soup.select_one(".entry-body, .event-body, article, main")
    if body_el:
        # 最初の意味のある段落を取得
        paras = [p.get_text(strip=True) for p in body_el.select("p") if len(p.get_text(strip=True)) > 20]
        if paras:
            desc = paras[0][:120]
    if not desc:
        # フォールバック: 本文テキストの最初の意味ある行
        lines = [l.strip() for l in soup.get_text().split("\n") if len(l.strip()) > 20]
        if len(lines) > 1:
            desc = lines[1][:120]  # 最初はタイトルなので2行目

    # ── 場所 ─────────────────────────────────────────────────────
    body_text = (body_el.get_text() if body_el else soup.get_text())[:800]
    location_raw = ""
    pref: Optional[str] = None
    city: Optional[str] = None

    # ① 明示ラベルパターン（最優先）
    venue_m = re.search(
        r"(?:集合場所|開催場所|会場|場所)[：:は]?\s*([^\s。、\n「」（）]{2,25})", body_text
    )
    if venue_m:
        venue = venue_m.group(1).strip()
        # 「〜駅で」「〜駅（」のような駅名も取得
        station_m = re.search(r"([^\s。、\n]{2,12}駅)", venue)
        location_raw = station_m.group(1) if station_m else venue

    # ② 都道府県 + 市区町村を本文から抽出
    _, pref, city = extract_location(body_text)

    # ③ タイトルから施設・駅名を抽出
    if not location_raw:
        # 駅名パターン（〇〇駅）
        station_in_title = re.search(r"([^\s　・]{2,10}駅)", title)
        if station_in_title:
            location_raw = station_in_title.group(1)
        else:
            # 施設名パターン（〇〇書店/ホール/センター/車庫/車両所）
            facility_in_title = re.search(
                r"([^\s　・]{2,12}(?:書店|ホール|センター|車庫|車両所|工場|博物館|美術館|公園))", title
            )
            if facility_in_title:
                location_raw = facility_in_title.group(1)

    # ④ 意味のない値（助詞のみ等）はクリア
    _INVALID_LOC = {"にて", "で", "において", "より", "から", "まで", ""}
    if location_raw.strip() in _INVALID_LOC or len(location_raw.strip()) <= 1:
        location_raw = ""

    # ⑤ prefecture + city を組み合わせて location_raw とする
    if not location_raw:
        location_raw = " ".join(filter(None, [pref, city]))

    return {
        "title": title,
        "description": desc,
        "location_raw": location_raw,
        "prefecture": pref,
        "city": city,
        "date_start": date_start,
        "source": "鉄道コム",
        "source_url": url,
    }


def scrape_tetsudocom() -> list[dict]:
    """
    鉄道コム — イベント一覧
    セレクタ（2026年3月時点確認済み）:
      新着: .event-top-new .list li a
      月別: .monthly-event-box li a
      日付: 個別ページタイトル「（YYYY年M月DD日）」
    """
    base = "https://www.tetsudo.com"
    soup = fetch_html(f"{base}/event/")
    if not soup:
        return []

    # 新着 + 月別からURLを収集（上位50件に限定）
    urls: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.select(".event-top-new .list li a, .monthly-event-box li a"):
        href = a.get("href", "")
        if not href.startswith("/event/"):
            continue
        full_url = base + href
        if full_url not in seen:
            seen.add(full_url)
            urls.append((full_url, a.get("title", a.get_text(strip=True))))

    urls = urls[:50]  # 最大50件
    log.info(f"鉄道コム: {len(urls)}件のURLを収集")

    results: list[dict] = []
    # 並列取得（最大5スレッド）
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_fetch_tetsudo_event, url, title): url for url, title in urls}
        for future in as_completed(futures):
            ev = future.result()
            if ev:
                results.append(ev)
            time.sleep(0.1)

    log.info(f"鉄道コム: {len(results)}件取得（日付付き）")
    return results


def scrape_walkerplus(themes: list[str]) -> list[dict]:
    """
    Walkerplus — テーマ別イベント
    セレクタ（2026年3月時点確認済み）:
      カード: div.m-mainlist-item
      タイトル: .m-mainlist-item__ttl a
      日付: .m-mainlist-item-event__period（「2026年1月15日(木)～3月31日(火)」形式）
      URL: .m-mainlist-item__ttl a[href]
    """
    base = "https://www.walkerplus.com"
    kw_map: dict[str, list[str]] = {
        "train":    ["鉄道", "電車"],
        "creature": ["昆虫", "動物"],
        "pokemon":  ["ポケモン"],
        "car":      ["自動車"],
    }
    keywords: list[str] = []
    for t in themes:
        keywords.extend(kw_map.get(t, []))
    keywords = list(dict.fromkeys(keywords))

    results: list[dict] = []
    seen: set[str] = set()

    for kw in keywords:
        # ar0313 = 関東
        url = f"{base}/event_list/ar0313/?keyword={kw}"
        soup = fetch_html(url)
        if not soup:
            continue

        for card in soup.select("div.m-mainlist-item"):
            try:
                title_el = card.select_one(".m-mainlist-item__ttl a")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                if not title:
                    continue

                href = title_el.get("href", "")
                src_url = (base + href) if href.startswith("/") else href
                if src_url in seen:
                    continue
                seen.add(src_url)

                # 日付（開始日を取得）
                period_el = card.select_one(".m-mainlist-item-event__period")
                period_text = period_el.get_text(strip=True) if period_el else ""
                date_start = parse_date(period_text)
                if not date_start:
                    continue

                # 場所（.m-mainlist-item__txt に住所が含まれることがある）
                txt_el = card.select_one(".m-mainlist-item__txt")
                loc_text = txt_el.get_text() if txt_el else ""
                location_raw, pref, city = extract_location(loc_text)

                results.append({
                    "title": title,
                    "description": "",
                    "location_raw": location_raw,
                    "prefecture": pref,
                    "city": city,
                    "date_start": date_start,
                    "source": "Walkerplus",
                    "source_url": src_url,
                })
            except Exception as e:
                log.debug(f"Walkerplus 解析エラー: {e}")

        time.sleep(1)

    log.info(f"Walkerplus: {len(results)}件取得")
    return results


def scrape_jreast() -> list[dict]:
    """JR東日本プレスリリース"""
    base = "https://www.jreast.co.jp"
    soup = fetch_html(f"{base}/press/")
    if not soup:
        return []

    EVENT_KW = ["イベント", "体験", "公開", "見学", "フェスタ", "フェア", "親子", "キッズ", "こども"]
    results: list[dict] = []
    items = soup.select(".press-list li, .newsList li, ul.list li, .releaseList li, li[class*='press']")

    for item in items:
        try:
            link_el = item.select_one("a[href]")
            title = link_el.get_text(strip=True) if link_el else ""
            if not title or not any(kw in title for kw in EVENT_KW):
                continue

            href = link_el.get("href", "") if link_el else ""
            src_url = urljoin(base, href)

            date_el = item.select_one(".date") or item.select_one("time")
            date_start = parse_date(date_el.get_text(strip=True) if date_el else "")
            if not date_start:
                continue

            results.append({
                "title": title, "description": "",
                "location_raw": "", "prefecture": "東京都", "city": None,
                "date_start": date_start, "source": "JR東日本", "source_url": src_url,
            })
        except Exception as e:
            log.debug(f"JR東日本 解析エラー: {e}")

    log.info(f"JR東日本: {len(results)}件取得")
    return results


def fetch_eventbrite() -> list[dict]:
    """Eventbrite API（APIキーが設定されている場合のみ）"""
    if not EVENTBRITE_API_KEY:
        log.info("Eventbrite: APIキー未設定、スキップ")
        return []

    url = "https://www.eventbriteapi.com/v3/events/search/"
    params = {
        "q": "鉄道 電車 子ども",
        "location.address": "Japan",
        "start_date.range_start": datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expand": "venue",
    }
    headers_api = {**HEADERS, "Authorization": f"Bearer {EVENTBRITE_API_KEY}"}

    results: list[dict] = []
    try:
        r = requests.get(url, params=params, headers=headers_api, timeout=12)
        r.raise_for_status()
        for ev in r.json().get("events", []):
            title = ev.get("name", {}).get("text", "")
            desc = ev.get("description", {}).get("text", "")[:200]
            date_start = (ev.get("start", {}).get("local", "") or "")[:10]
            date_end = (ev.get("end", {}).get("local", "") or "")[:10] or None
            address_obj = ev.get("venue", {}).get("address", {})
            location_raw = address_obj.get("localized_address_display", "")
            _, pref, city = extract_location(location_raw)
            results.append({
                "title": title, "description": desc,
                "location_raw": location_raw, "prefecture": pref, "city": city,
                "date_start": date_start, "date_end": date_end,
                "source": "Eventbrite", "source_url": ev.get("url", ""),
            })
    except Exception as e:
        log.warning(f"Eventbrite 失敗: {e}")

    log.info(f"Eventbrite: {len(results)}件取得")
    return results


# ── フィルタ＆保存パイプライン ───────────────────────────────────
def process_and_save(raw: list[dict], conn: sqlite3.Connection, active_themes: list[str]) -> int:
    added = 0
    for ev in raw:
        title = (ev.get("title") or "").strip()
        desc = (ev.get("description") or "").strip()
        date_start = ev.get("date_start") or ""

        if not title or not date_start:
            continue
        if not is_future(date_start):
            continue

        # テーマ判定
        matched_theme = next(
            (t for t in active_themes if matches_theme(t, title, desc)),
            None,
        )
        if not matched_theme:
            continue

        # 子ども向けスコア
        score = kid_score(title, desc)
        kid_friendly = score >= 2

        # 曖昧ケース → Gemini で補完（1日20件上限）
        if score == 1 or (score >= 2 and not desc):
            llm = call_gemini_judge(title, desc)
            kid_friendly = llm["is_kid_friendly"]
            if llm.get("description"):
                desc = llm["description"]

        if not kid_friendly:
            continue

        location_raw = ev.get("location_raw") or ""
        event_id = make_id(title, date_start, location_raw)

        if event_exists(conn, event_id):
            continue

        insert_event(conn, {
            "id": event_id,
            "title": title,
            "description": desc,
            "location_raw": location_raw,
            "prefecture": ev.get("prefecture"),
            "city": ev.get("city"),
            "date_start": date_start,
            "date_end": ev.get("date_end"),
            "source": ev.get("source", ""),
            "source_url": ev.get("source_url", ""),
            "is_kid_friendly": True,
            "is_train_related": matched_theme == "train",
            "theme_id": matched_theme,
        })
        added += 1

    conn.commit()
    return added


# ── JSON エクスポート ────────────────────────────────────────────
def export_json(conn: sqlite3.Connection, path: str) -> None:
    rows = conn.execute(
        """SELECT id, title, description, location_raw, prefecture, city,
                  date_start, date_end, source, source_url, theme_id
           FROM events
           WHERE date_start >= ? AND date_start <= ?
             AND is_kid_friendly = 1
           ORDER BY date_start ASC""",
        (TODAY.isoformat(), FUTURE_LIMIT.isoformat()),
    ).fetchall()

    cols = ["id", "title", "description", "location_raw", "prefecture", "city",
            "date_start", "date_end", "source", "source_url", "theme_id"]

    app_events = []
    for row in rows:
        e = dict(zip(cols, row))
        # location = 都道府県 + 市区町村（+ 施設名があれば追記）
        parts = [p for p in [e.get("prefecture"), e.get("city")] if p]
        loc_raw = e.get("location_raw") or ""
        if parts:
            location = " ".join(parts)
            # 施設名・駅名がlocation_rawにあれば追加
            if loc_raw and loc_raw not in location:
                extra = re.sub(PREF_RE, "", loc_raw).strip()
                extra = re.sub(r"[^\s]{2,8}(?:市|区|町|村)", "", extra).strip()
                if extra and len(extra) >= 2:
                    location = f"{location} {extra}"
        else:
            # 都道府県不明の場合はlocation_rawを使う（空でも可、アプリのfilter.regionにフォールバック）
            location = loc_raw

        app_events.append({
            "id": e["id"],
            "title": e["title"],
            "date": e["date_start"],
            "location": location,
            "description": e["description"] or e["title"],
            "prefecture": e.get("prefecture") or "",
            "city": e.get("city") or "",
            "themeId": e.get("theme_id", "train"),
            "source": e.get("source", ""),
            "url": e.get("source_url", ""),
            "fetched_at": datetime.now(JST).isoformat(),
        })

    output = {
        "generated_at": datetime.now(JST).isoformat(),
        "count": len(app_events),
        "events": app_events,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info(f"エクスポート完了: {path} ({len(app_events)}件)")


# ── メイン ───────────────────────────────────────────────────────
def main() -> None:
    log.info("=== イベント収集開始 ===")

    active_themes = os.environ.get(
        "ACTIVE_THEMES", "train,creature,pokemon,car"
    ).split(",")
    log.info(f"対象テーマ: {active_themes}")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    load_existing_json(conn, OUTPUT_PATH)
    cleanup_old(conn)

    all_raw: list[dict] = []
    scrapers = [
        ("いこーよ",    lambda: scrape_ikoyo(active_themes)),
        ("鉄道コム",    scrape_tetsudocom),
        ("JR東日本",    scrape_jreast),
        ("Walkerplus", lambda: scrape_walkerplus(active_themes)),
        ("Eventbrite", fetch_eventbrite),
    ]

    for name, fn in scrapers:
        try:
            items = fn()
            log.info(f"{name}: {len(items)}件")
            all_raw.extend(items)
        except Exception as e:
            log.error(f"{name} 例外: {e}", exc_info=True)

    log.info(f"合計取得: {len(all_raw)}件")

    added = process_and_save(all_raw, conn, active_themes)
    log.info(f"新規追加: {added}件 / LLM使用: {_llm_calls}回/{MAX_LLM_CALLS}回")

    export_json(conn, OUTPUT_PATH)
    conn.close()
    log.info("=== 完了 ===")


if __name__ == "__main__":
    main()
