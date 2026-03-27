import { EventItem, EventFilter } from './events';

// ─── フィード定義 ────────────────────────────────────────────
type FeedDef = {
  themeId: string;
  name: string;
  url: string;
  regions: string[] | null;
};

const FEEDS: FeedDef[] = [
  // ── 電車 ──────────────────────────────────────────────────
  {
    themeId: 'train', name: 'railf.jp（鉄道ファン）',
    url: 'https://railf.jp/rss/rss.xml',
    regions: null,
  },
  {
    themeId: 'train', name: '東急電鉄',
    url: 'https://www.tokyu.co.jp/rss_news.xml',
    regions: ['東京都', '神奈川県'],
  },

  // ── 生き物 ────────────────────────────────────────────────
  {
    themeId: 'creature', name: 'NHK科学・文化',
    url: 'https://news.web.nhk/n-data/conf/na/rss/cat4.xml',
    regions: null,
  },

  // ── ポケモン ──────────────────────────────────────────────
  {
    themeId: 'pokemon', name: '4Gamer.net',
    url: 'https://www.4gamer.net/rss/index.xml',
    regions: null,
  },
  {
    themeId: 'pokemon', name: 'Game Watch',
    url: 'https://game.watch.impress.co.jp/data/rss/1.0/gmw/feed.rdf',
    regions: null,
  },
  {
    themeId: 'pokemon', name: 'デンファミニコゲーマー',
    url: 'https://news.denfaminicogamer.jp/feed',
    regions: null,
  },

  // ── 車 ────────────────────────────────────────────────────
  {
    themeId: 'car', name: 'Car Watch',
    url: 'https://car.watch.impress.co.jp/data/rss/1.0/car/feed.rdf',
    regions: null,
  },
  {
    themeId: 'car', name: 'Response',
    url: 'https://response.jp/rss20/index.rdf',
    regions: null,
  },
];

// ─── テーマキーワード（全フィード共通でチェック）──────────────────
const THEME_KEYWORDS: Record<string, string[]> = {
  train: ['電車', '鉄道', '列車', '駅', 'JR', '東急', '小田急', '相鉄', '京急', '西武', '東武', '阪急', '近鉄', '新幹線', 'SL', 'ミニ電車', '路面電車', 'モノレール', 'トロッコ'],
  creature: ['生き物', '自然', '昆虫', '動物', '植物', '水族館', '動物園', '博物館', '観察', '生物', '環境', '野生', '魚', '鳥', '虫', 'カブトムシ', 'チョウ', '標本', '化石', '恐竜', '水辺'],
  pokemon: ['ポケモン', 'Pokemon', 'Pokémon', 'ポケモンカード', 'ポケモンセンター', 'ピカチュウ', 'ポケモンGO', 'ポケカ'],
  car: ['自動車', 'カー', 'EV', '電気自動車', 'モーター', 'トヨタ', 'ホンダ', 'ニッサン', '日産', 'マツダ', 'スバル', 'ドライブ', 'レース', 'ミニカー', '乗り物', 'バス', 'トラック'],
};

// ─── 子ども向けイベント判定 ──────────────────────────────────
// これらのキーワードのいずれかを含む記事のみ「イベント」として扱う
const EVENT_REQUIRED_KEYWORDS = [
  'イベント', 'フェスタ', 'フェス', 'フェア', 'まつり', '祭り', '祭',
  '体験', '見学', '観察', '展示', '展覧', '企画展', '特別展',
  'ワークショップ', '工作', '実験', '教室', '講座', '講習',
  '乗車', '乗れる', '乗り', '運転', 'ツアー', '旅', '遠足',
  'こども', '子ども', '子供', 'キッズ', '親子', 'ファミリー',
  'ショー', 'パレード', '撮影会', '見どころ', '遊び', '広場',
  '開催', '開館', 'オープン', 'グランドオープン', '登場',
  '新作', '発売', '発表会', 'ポップアップ', 'コラボ',
];

// これらのキーワードを含む記事は除外（大人向け・政治経済等）
const EXCLUDE_KEYWORDS = [
  '予算', '国会', '議会', '法案', '規制', '条例', '人事', '辞任', '就任',
  '株価', '決算', 'IR ', '経済指標', '金利', '債券', '為替',
  '訃報', '死亡', '事件', '事故', '逮捕', '起訴', '裁判',
  '地震', '台風', '災害', '警報', '避難',
  '補助金', '助成', '税制', '税金', '料金改定', '値上げ',
  '人事異動', '役員', '株主', '配当',
];

function isChildFriendlyEvent(title: string, description: string): boolean {
  const text = `${title} ${description}`;
  // 除外キーワードが含まれていたらNG
  if (EXCLUDE_KEYWORDS.some((kw) => text.includes(kw))) return false;
  // イベント系キーワードのいずれかが含まれていればOK
  return EVENT_REQUIRED_KEYWORDS.some((kw) => text.includes(kw));
}

function matchesTheme(themeId: string, title: string, description: string): boolean {
  const keywords = THEME_KEYWORDS[themeId] ?? [];
  const text = `${title} ${description}`;
  return keywords.some((kw) => text.includes(kw));
}

// ─── 開催地を本文から抽出 ──────────────────────────────────
// fallback: filter.region（都道府県市区レベル、例: 神奈川県海老名市）
function extractLocation(title: string, description: string, fallback: string): string {
  const text = `${title} ${description}`;

  // ① 明示ラベルパターン（最優先・最も信頼性高）
  // 「会場：〜」「場所：〜」「開催地：〜」「開催場所：〜」
  const labeled = text.match(
    /(?:会場|場所|開催地|開催場所|開催地域)[：:]\s*([^\s。、\n「」【】]{2,30})/
  );
  if (labeled) return labeled[1].trim();

  // ② 都道府県 + 市区町村 の住所パターン（本文中に含まれる場合）
  const prefCity = text.match(
    /((?:北海道|東京都|大阪府|京都府|[^\s]{2,4}[都道府県])[^\s。、\n]{1,8}(?:市|区|町|村))/
  );
  if (prefCity) return prefCity[1].replace(/\s+/g, '');

  // ③ 施設名（「〜で」「〜にて」「〜において」と続く場合のみ採用）
  //    単独の施設名は誤抽出が多いため「動詞」とセットの場合に限定
  const venueWithPrep = text.match(
    /([^\s。、\n「」（）【】]{2,15}(?:ホール|博物館|美術館|動物園|水族館|科学館|プラザ|ミュージアム|アリーナ|スタジアム|ドーム|タワー|センター|パーク))(?:で|にて|での|において|に於いて)/
  );
  if (venueWithPrep) return venueWithPrep[1].trim();

  // ④ フォールバック：ユーザーの都道府県市区（常に都道府県市区レベルを担保）
  return fallback;
}

// ─── HTTP フェッチ（CORS プロキシ fallback）────────────────────
const PROXY_1 = 'https://corsproxy.io/?url=';
const PROXY_2 = 'https://api.allorigins.win/get?url=';

async function fetchFeed(url: string): Promise<string | null> {
  const timeout = (ms: number) =>
    new Promise<never>((_, reject) => setTimeout(() => reject(new Error('timeout')), ms));

  // ① 直接フェッチ
  try {
    const res = await Promise.race([
      fetch(url, { headers: { Accept: 'application/rss+xml,application/xml,text/xml,*/*' } }),
      timeout(6000),
    ]) as Response;
    if (res.ok) return await res.text();
  } catch { /* CORS or timeout */ }

  // ② corsproxy.io
  try {
    const res = await Promise.race([
      fetch(`${PROXY_1}${encodeURIComponent(url)}`),
      timeout(8000),
    ]) as Response;
    if (res.ok) return await res.text();
  } catch { /* failed */ }

  // ③ allorigins.win
  try {
    const res = await Promise.race([
      fetch(`${PROXY_2}${encodeURIComponent(url)}`),
      timeout(8000),
    ]) as Response;
    if (res.ok) {
      const data = await res.json();
      return data.contents ?? null;
    }
  } catch { /* failed */ }

  return null;
}

// ─── XML パーサー ──────────────────────────────────────────────
function extractTag(chunk: string, tag: string): string {
  const cd = chunk.match(new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([\\s\\S]*?)\\]\\]>`, 'i'));
  if (cd) return cd[1].trim();
  const m = chunk.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i'));
  if (m) return m[1].replace(/<[^>]+>/g, '').trim();
  if (tag === 'link') {
    const h = chunk.match(/<link[^>]+href="([^"]+)"/i);
    if (h) return h[1];
  }
  return '';
}

type RssItem = { title: string; link: string; pubDate: string; description: string };

function parseRss(xml: string): RssItem[] {
  const items: RssItem[] = [];
  const re = /<(?:item|entry)(?:\s[^>]*)?>[\s\S]*?<\/(?:item|entry)>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml)) !== null) {
    const c = m[0];
    const title = extractTag(c, 'title');
    const link = extractTag(c, 'link');
    const pubDate =
      extractTag(c, 'pubDate') ||
      extractTag(c, 'published') ||
      extractTag(c, 'updated') ||
      '';
    const description =
      extractTag(c, 'description') ||
      extractTag(c, 'summary') ||
      extractTag(c, 'content') ||
      '';
    if (title) items.push({ title, link, pubDate, description });
  }
  return items;
}

// ─── 日本語日付抽出 ────────────────────────────────────────────
function extractEventDate(text: string): string | null {
  const y = new Date().getFullYear();
  const full = text.match(/(\d{4})[年\/](\d{1,2})[月\/](\d{1,2})日?/);
  if (full) return `${full[1]}-${full[2].padStart(2, '0')}-${full[3].padStart(2, '0')}`;
  const short = text.match(/(\d{1,2})月(\d{1,2})日/);
  if (short) return `${y}-${short[1].padStart(2, '0')}-${short[2].padStart(2, '0')}`;
  return null;
}

function cleanHtml(s: string): string {
  return s
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&#\d+;/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120);
}

// ─── メイン：RSS イベント取得 ──────────────────────────────────
export async function fetchRssEvents(filter: EventFilter): Promise<EventItem[]> {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const future = new Date(today);
  future.setDate(today.getDate() + 30);

  // テーマ一致 + 地域一致のフィードのみ対象
  const targetFeeds = FEEDS.filter((f) => {
    if (!filter.themes.includes(f.themeId)) return false;
    if (f.regions === null) return true;
    return f.regions.includes(filter.prefecture);
  });

  if (targetFeeds.length === 0) return [];

  const allEvents: EventItem[] = [];

  await Promise.allSettled(
    targetFeeds.map(async (feed) => {
      try {
        const xml = await fetchFeed(feed.url);
        if (!xml) {
          console.warn(`[RSS] ${feed.name}: 取得失敗`);
          return;
        }

        const items = parseRss(xml);
        console.log(`[RSS] ${feed.name}: ${items.length}件取得`);

        let accepted = 0;
        for (const item of items.slice(0, 30)) {
          const titleClean = cleanHtml(item.title);
          const descClean = cleanHtml(item.description);

          // ① テーマキーワードフィルタ（全フィード共通）
          if (!matchesTheme(feed.themeId, titleClean, descClean)) continue;

          // ② 子ども向けイベント判定（大人向けニュースを除外）
          if (!isChildFriendlyEvent(titleClean, descClean)) continue;

          // ③ 日付取得（本文 → pubDate の順）
          const combined = `${titleClean} ${descClean}`;
          let eventDate = extractEventDate(combined);
          if (!eventDate && item.pubDate) {
            const d = new Date(item.pubDate);
            if (!isNaN(d.getTime())) eventDate = d.toISOString().slice(0, 10);
          }
          if (!eventDate) continue;

          // ④ 期間フィルター（今日〜30日後）
          const ed = new Date(eventDate);
          if (ed < today || ed > future) continue;

          // ⑤ 開催地を本文から抽出（なければ都道府県市区レベルをフォールバック）
          const location = extractLocation(titleClean, descClean, filter.region);

          allEvents.push({
            id: `rss_${feed.themeId}_${ed.getTime()}_${Math.random().toString(36).slice(2, 6)}`,
            title: titleClean,
            date: eventDate,
            location,
            description: descClean || `${feed.name}からのお知らせ`,
            prefecture: filter.prefecture,
            themeId: feed.themeId,
            source: feed.name,
            url: item.link || undefined,
          });
          accepted++;
        }
        console.log(`[RSS] ${feed.name}: ${accepted}件採用`);
      } catch (e) {
        console.warn(`[RSS] ${feed.name} 失敗:`, e);
      }
    })
  );

  console.log(`[RSS] 合計: ${allEvents.length}件`);
  return allEvents.sort((a, b) => a.date.localeCompare(b.date));
}
