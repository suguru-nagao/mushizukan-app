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
        "魚", "鳥", "虫", "標本", "生物", "ビオトープ",
    ],
    "pokemon": [
        "ポケモン", "Pokemon", "Pokémon", "ポケモンカード",
        "ポケモンセンター", "ポケカ", "ポケモンGO",
    ],
    "car": [
        "自動車", "ミニカー", "モーターショー", "キッズカー", "乗り物",
        "トヨタ", "ホンダ", "日産", "マツダ", "スバル", "EV体験",
        "ドライブ体験", "カーレース",
    ],
}

# 子ども向けスコア
KID_SCORE_MAP: dict[str, int] = {
    "親子": 2, "キッズ": 2, "子ども": 2, "子供": 2, "ファミリー": 2,
    "こども": 2, "幼児": 2, "小学生": 1, "未就学": 2,
    "体験": 1, "ワークショップ": 1, "見学": 1, "乗車": 1,
    "遊び": 1, "工作": 1, "実験": 1, "教室": 1,
    "フェスタ": 1, "フェア": 1, "まつり": 1, "祭": 1,
    "展示": 1, "特別展": 1, "企画展": 1,
}
KID_PENALTY_MAP: dict[str, int] = {
    "セミナー": -2, "ビジネス": -2, "株主": -2, "採用": -2,
    "投資": -2, "転職": -2, "就活": -2, "企業向け": -2,
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
    if not text:
        return None
    t = text.strip()
    # YYYY年MM月DD日 or YYYY/MM/DD
    m = re.search(r"(\d{4})[年/](\d{1,2})[月/](\d{1,2})日?", t)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    # MM月DD日（今年）
    m = re.search(r"(\d{1,2})月(\d{1,2})日", t)
    if m:
        return f"{TODAY.year}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
    # ISO
    m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
    if m:
        return m.group(1)
    return None


def is_future(date_str: str) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return TODAY <= d <= FUTURE_LIMIT
    except Exception:
        return False


def extract_location(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """(location_raw, prefecture, city) を返す"""
    location_raw = text.strip()[:60]
    pref: Optional[str] = None
    city: Optional[str] = None

    # 都道府県+市区町村パターン
    m = ADDR_RE.search(text)
    if m:
        addr = m.group(1).replace(" ", "")
        pm = PREF_RE.search(addr)
        if pm:
            pref = pm.group(1)
        cm = re.search(r"([^\s]{2,8}(?:市|区|町|村))", addr[len(pref or ""):])
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
    """いこーよ — テーマ別キーワード検索"""
    base = "https://iko-yo.net"
    kw_map = {
        "train":   ["鉄道", "電車"],
        "creature": ["昆虫", "自然観察", "動物"],
        "pokemon": ["ポケモン"],
        "car":     ["自動車", "乗り物"],
    }
    keywords = []
    for t in themes:
        keywords.extend(kw_map.get(t, []))

    results: list[dict] = []
    seen_urls: set[str] = set()

    for kw in keywords:
        for page in range(1, 4):
            url = f"{base}/events?keyword={kw}&page={page}"
            soup = fetch_html(url)
            if not soup:
                break

            # いこーよは複数のHTMLバリアントがある
            cards = (
                soup.select(".event-list-item")
                or soup.select("article[class*='event']")
                or soup.select("li[class*='event']")
                or soup.select("div[class*='EventCard']")
            )
            if not cards:
                log.debug(f"いこーよ: カード0件 kw={kw} page={page}")
                break

            for card in cards:
                try:
                    title_el = (
                        card.select_one(".event-title")
                        or card.select_one("h2")
                        or card.select_one("h3")
                        or card.select_one("[class*='title']")
                    )
                    title = title_el.get_text(strip=True) if title_el else ""
                    if not title:
                        continue

                    link_el = card.select_one("a[href]")
                    href = link_el.get("href", "") if link_el else ""
                    src_url = urljoin(base, href) if href else url
                    if src_url in seen_urls:
                        continue
                    seen_urls.add(src_url)

                    date_el = (
                        card.select_one(".event-date")
                        or card.select_one("time")
                        or card.select_one("[class*='date']")
                    )
                    date_start = parse_date(date_el.get_text(strip=True) if date_el else "")
                    if not date_start:
                        continue

                    place_el = (
                        card.select_one(".event-place")
                        or card.select_one("[class*='place']")
                        or card.select_one("[class*='location']")
                    )
                    location_raw = place_el.get_text(strip=True) if place_el else ""
                    _, pref, city = extract_location(location_raw)

                    desc_el = card.select_one("[class*='desc']") or card.select_one("p")
                    desc = desc_el.get_text(strip=True)[:200] if desc_el else ""

                    results.append({
                        "title": title, "description": desc,
                        "location_raw": location_raw, "prefecture": pref, "city": city,
                        "date_start": date_start, "source": "いこーよ", "source_url": src_url,
                    })
                except Exception as e:
                    log.debug(f"いこーよ カード解析エラー: {e}")

            time.sleep(1)

    log.info(f"いこーよ: {len(results)}件取得")
    return results


def scrape_tetsudocom() -> list[dict]:
    """鉄道コム — イベント一覧"""
    base = "https://www.tetsudo.com"
    url = f"{base}/event/"
    soup = fetch_html(url)
    if not soup:
        return []

    results: list[dict] = []
    # 鉄道コムのイベントリスト（複数バリアント対応）
    cards = (
        soup.select(".event_list li")
        or soup.select(".eventList li")
        or soup.select("ul.event li")
        or soup.select("li[class*='event']")
    )

    for card in cards:
        try:
            link_el = card.select_one("a[href]")
            title = link_el.get_text(strip=True) if link_el else ""
            if not title:
                continue
            href = link_el.get("href", "") if link_el else ""
            src_url = urljoin(base, href) if href else url

            date_el = card.select_one(".date") or card.select_one("time")
            date_text = date_el.get_text(strip=True) if date_el else ""
            date_start = parse_date(date_text)
            if not date_start:
                continue

            _, pref, city = extract_location(title)

            results.append({
                "title": title, "description": "",
                "location_raw": "", "prefecture": pref, "city": city,
                "date_start": date_start, "source": "鉄道コム", "source_url": src_url,
            })
        except Exception as e:
            log.debug(f"鉄道コム 解析エラー: {e}")

    log.info(f"鉄道コム: {len(results)}件取得")
    return results


def scrape_jreast() -> list[dict]:
    """JR東日本プレスリリース"""
    base = "https://www.jreast.co.jp"
    url = f"{base}/press/"
    soup = fetch_html(url)
    if not soup:
        return []

    EVENT_KW = ["イベント", "体験", "公開", "見学", "フェスタ", "フェア", "親子", "キッズ", "こども"]
    results: list[dict] = []
    items = (
        soup.select(".press-list li")
        or soup.select(".newsList li")
        or soup.select("ul.list li")
        or soup.select(".releaseList li")
        or soup.select("li[class*='press']")
    )

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


def scrape_walkerplus(themes: list[str]) -> list[dict]:
    """Walkerplus — テーマ別イベント"""
    base = "https://www.walkerplus.com"
    kw_map = {
        "train":    ["鉄道", "電車"],
        "creature": ["昆虫", "動物"],
        "pokemon":  ["ポケモン"],
        "car":      ["自動車"],
    }
    keywords = []
    for t in themes:
        keywords.extend(kw_map.get(t, []))

    results: list[dict] = []
    seen: set[str] = set()

    for kw in keywords:
        # ar0313 = 全国, genre0202 = 展示・見学
        url = f"{base}/event_list/ar0313/?keyword={kw}"
        soup = fetch_html(url)
        if not soup:
            continue

        cards = (
            soup.select(".m-mainlist-item")
            or soup.select("[class*='mainlist-item']")
            or soup.select("article[class*='event']")
        )
        for card in cards:
            try:
                title_el = (
                    card.select_one(".m-mainlist-item__ttl")
                    or card.select_one("[class*='ttl']")
                    or card.select_one("h2,h3")
                )
                title = title_el.get_text(strip=True) if title_el else ""
                if not title:
                    continue

                link_el = card.select_one("a[href]")
                href = link_el.get("href", "") if link_el else ""
                src_url = urljoin(base, href) if href else url
                if src_url in seen:
                    continue
                seen.add(src_url)

                date_el = (
                    card.select_one(".m-mainlist-item-event-date")
                    or card.select_one("[class*='date']")
                    or card.select_one("time")
                )
                date_start = parse_date(date_el.get_text(strip=True) if date_el else "")
                if not date_start:
                    continue

                place_el = card.select_one("[class*='place']") or card.select_one("[class*='venue']")
                location_raw = place_el.get_text(strip=True) if place_el else ""
                _, pref, city = extract_location(location_raw or title)

                results.append({
                    "title": title, "description": "",
                    "location_raw": location_raw, "prefecture": pref, "city": city,
                    "date_start": date_start, "source": "Walkerplus", "source_url": src_url,
                })
            except Exception as e:
                log.debug(f"Walkerplus 解析エラー: {e}")
        time.sleep(1)

    log.info(f"Walkerplus: {len(results)}件取得")
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
    headers_api = {"Authorization": f"Bearer {EVENTBRITE_API_KEY}"}

    results: list[dict] = []
    try:
        r = requests.get(url, params=params, headers={**HEADERS, **headers_api}, timeout=12)
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
    """
    フィルタリング → 重複排除 → DB保存
    追加件数を返す
    """
    added = 0
    for ev in raw:
        title = (ev.get("title") or "").strip()
        desc = (ev.get("description") or "").strip()
        date_start = ev.get("date_start") or ""

        if not title or not date_start:
            continue
        if not is_future(date_start):
            continue

        # テーマ判定（いずれかのテーマに一致すること）
        matched_theme = next(
            (t for t in active_themes if matches_theme(t, title, desc)),
            None,
        )
        if not matched_theme:
            continue

        # 子ども向けスコア
        score = kid_score(title, desc)
        kid_friendly = score >= 2

        # 曖昧ケース（score=1）または説明文なし → LLMで補完（1日20件上限）
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
            "is_train_related": "train" in (matched_theme or ""),
            "theme_id": matched_theme or "train",
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
        # locationは「都道府県 + 市区町村」が揃えば組み合わせる
        parts = [p for p in [e.get("prefecture"), e.get("city")] if p]
        location = " ".join(parts) if parts else (e.get("location_raw") or "日本")
        # 施設名も含まれている場合はlocation_rawから追記
        if parts and e.get("location_raw") and e["location_raw"] not in location:
            loc_extra = re.sub(PREF_RE, "", e["location_raw"]).strip()
            loc_extra = re.sub(r".{2,8}(?:市|区|町|村)", "", loc_extra).strip(" 　")
            if loc_extra:
                location = f"{location} {loc_extra}"

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

    # 対象テーマ（env var で上書き可, デフォルト全テーマ）
    active_themes = os.environ.get(
        "ACTIVE_THEMES", "train,creature,pokemon,car"
    ).split(",")
    log.info(f"対象テーマ: {active_themes}")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    load_existing_json(conn, OUTPUT_PATH)
    cleanup_old(conn)

    # 各ソースからスクレイプ
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
            log.error(f"{name} 例外: {e}")

    log.info(f"合計取得: {len(all_raw)}件")

    added = process_and_save(all_raw, conn, active_themes)
    log.info(f"新規追加: {added}件 / LLM使用: {_llm_calls}回/{MAX_LLM_CALLS}回")

    export_json(conn, OUTPUT_PATH)
    conn.close()
    log.info("=== 完了 ===")


if __name__ == "__main__":
    main()
