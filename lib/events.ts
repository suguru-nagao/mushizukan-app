import AsyncStorage from '@react-native-async-storage/async-storage';
import { THEME_DEFS, getThemeDef, ThemeId } from './themes';
import { fetchRssEvents } from './rss';
import { fetchBatchEvents } from './eventBatch';

export type TransportMode = 'car' | 'train';

export type EventFilter = {
  transport: TransportMode;
  minutes: number;
  region: string;
  prefecture: string;
  themes: string[]; // 選択中テーマID
};

export type EventItem = {
  id: string;
  title: string;
  date: string; // YYYY-MM-DD
  location: string;
  description: string;
  prefecture: string;
  themeId: string;
  source?: string; // 情報元ドメイン（例: sotetsu.co.jp）
  url?: string;
};

const GEMINI_KEY = process.env.EXPO_PUBLIC_GEMINI_API_KEY;
const GEMINI_SEARCH_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_KEY}`;
const GEMINI_LITE_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_KEY}`;
const CACHE_TTL_MS = 6 * 60 * 60 * 1000;

type EventsCache = {
  events: EventItem[];
  fetchedAt: string;
};

function toRadius(transport: TransportMode, minutes: number): number {
  return Math.round(((transport === 'car' ? 50 : 30) * minutes) / 60);
}

function cacheKey(filter: EventFilter): string {
  const t = [...filter.themes].sort().join(',');
  return `events_${filter.prefecture}_${filter.transport}_${filter.minutes}_${t}`;
}

// フォールバック用サンプルデータ（テーマ別）
export const SAMPLE_EVENTS: EventItem[] = [
  { id: 's1', title: '春の虫さがしウォーク', date: '2026-03-28', location: '代々木公園', description: '春の虫を一緒に探しましょう！', prefecture: '東京都', themeId: 'creature' },
  { id: 's2', title: 'こどもむし博士講座', date: '2026-03-29', location: '渋谷区立図書館', description: '虫の標本を見ながら学ぼう！', prefecture: '東京都', themeId: 'creature' },
  { id: 's3', title: '鉄道フェスタ', date: '2026-03-30', location: '横浜鉄道博物館', description: 'ミニ電車に乗ろう！運転体験もできるよ。', prefecture: '神奈川県', themeId: 'train' },
  { id: 's4', title: 'ポケモンカードゲーム大会', date: '2026-03-29', location: 'イオンモール', description: '子ども向けポケモンカード大会！', prefecture: '神奈川県', themeId: 'pokemon' },
  { id: 's5', title: 'キッズカーフェスタ', date: '2026-04-05', location: '海老名SA', description: '子どもが乗れるミニカーが大集合！', prefecture: '神奈川県', themeId: 'car' },
];

export function SAMPLE_EVENTS_FALLBACK(filter: EventFilter): EventItem[] {
  return SAMPLE_EVENTS.filter(
    (e) =>
      (e.prefecture === filter.prefecture || filter.prefecture === '東京都') &&
      filter.themes.includes(e.themeId)
  );
}

function buildPrompt(filter: EventFilter, todayStr: string, endStr: string): string {
  const radius = toRadius(filter.transport, filter.minutes);
  const transportLabel = filter.transport === 'car' ? '車' : '電車';

  const themeLines = filter.themes
    .map((id) => {
      const def = getThemeDef(id);
      const sites = def.officialDomains.slice(0, 6).join(', ');
      return `- ${def.emoji}${def.label}（キーワード: ${def.keywords.join('・')}）\n  優先検索サイト: ${sites}\n  → themeId: "${id}"`;
    })
    .join('\n');

  return `現在地「${filter.region}」から${transportLabel}で${filter.minutes}分以内（約${radius}km圏内）で、
${todayStr}から${endStr}の間に開催される子ども向けイベントを検索してください。

対象カテゴリと優先検索サイト:
${themeLines}

【検索方針】
- 上記「優先検索サイト」のイベント・ニュースページを最優先で確認すること
- 鉄道会社・メーカー・公式団体の公式サイトに掲載されているイベントを優先する
- 地域の博物館・公園・施設の公式サイトも検索対象に含める

【除外ルール】
- 公式サイト・公式SNS・報道で開催日時が明確に確認できるもののみ含める
- 過去実績のみ・日程未定・推測による日付は含めない
- dateConfirmed: true のもののみ返す

以下のJSON配列形式のみで返してください（前置き不要）：
[
  {
    "title": "イベント名",
    "date": "YYYY-MM-DD",
    "location": "開催場所（都道府県＋市区町村＋施設名。例: 神奈川県海老名市 ビナウォーク、東京都渋谷区 代々木公園）",
    "description": "1〜2文の説明（子ども向けにやさしく）",
    "themeId": "上記カテゴリのthemeId",
    "source": "情報元サイトのドメイン（例: sotetsu.co.jp）",
    "dateConfirmed": true
  }
]

確認できるイベントがない場合は [] のみ返してください。`;
}

// GeminiのURLをサニタイズ（Markdown記法・余分な文字を除去）
function sanitizeUrl(raw: any): string | undefined {
  if (!raw || typeof raw !== 'string') return undefined;
  // Markdown リンク形式 [text](url) から URL を抽出
  const mdMatch = raw.match(/\(?(https?:\/\/[^\s)\]"']+)\)?/);
  if (mdMatch) return mdMatch[1];
  // そのまま http/https URL なら使用
  if (/^https?:\/\/.+/.test(raw.trim())) return raw.trim();
  return undefined;
}

function parseEvents(text: string, filter: EventFilter): EventItem[] {
  const match = text.match(/\[[\s\S]*\]/);
  if (!match) return [];
  try {
    const parsed: any[] = JSON.parse(match[0]);
    const validThemes = new Set(THEME_DEFS.map((t) => t.id));
    return parsed
      .filter((e) =>
        e.title &&
        e.date &&
        /^\d{4}-\d{2}-\d{2}$/.test(e.date) &&
        e.dateConfirmed !== false  // 日時未確認のイベントを除外
      )
      .map((e, i) => ({
        id: `g_${Date.now()}_${i}`,
        title: e.title,
        date: e.date,
        location: e.location ?? filter.prefecture,
        description: e.description ?? '',
        prefecture: filter.prefecture,
        themeId: validThemes.has(e.themeId) ? e.themeId : filter.themes[0] ?? 'creature',
        source: typeof e.source === 'string' ? e.source : undefined,
        url: sanitizeUrl(e.url),
      }));
  } catch {
    return [];
  }
}

export async function fetchEventsFromGemini(filter: EventFilter): Promise<EventItem[]> {
  const today = new Date();
  const end = new Date(today);
  end.setDate(today.getDate() + 30);
  const todayStr = today.toISOString().slice(0, 10);
  const endStr = end.toISOString().slice(0, 10);
  const prompt = buildPrompt(filter, todayStr, endStr);

  // ① gemini-2.0-flash + Google Search
  try {
    const res = await fetch(GEMINI_SEARCH_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        tools: [{ google_search: {} }],
        generationConfig: { maxOutputTokens: 2000, temperature: 0.1 },
      }),
    });
    if (res.ok) {
      const data = await res.json();
      const text: string = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? '';
      const events = parseEvents(text, filter);
      if (events.length > 0) {
        console.log(`[Events] search: ${events.length}件`);
        return events;
      }
    } else {
      const body = await res.text().catch(() => '');
      console.warn(`[Events] search ${res.status}: ${body.slice(0, 150)}`);
    }
  } catch (e) {
    console.warn('[Events] search 失敗:', e);
  }

  // ② gemini-2.5-flash-lite（訓練データ）
  console.log('[Events] フォールバック: lite');
  const res2 = await fetch(GEMINI_LITE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 2000, temperature: 0.3 },
    }),
  });
  if (!res2.ok) throw new Error(`lite ${res2.status}`);
  const data2 = await res2.json();
  const text2: string = data2.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? '';
  const events2 = parseEvents(text2, filter);
  console.log(`[Events] lite: ${events2.length}件`);
  return events2;
}

export async function getEventsWithCache(filter: EventFilter): Promise<{
  events: EventItem[];
  fromCache: boolean;
  fetchedAt: string | null;
}> {
  const key = cacheKey(filter);
  try {
    const cached = await AsyncStorage.getItem(key);
    if (cached) {
      const parsed: EventsCache = JSON.parse(cached);
      if (Date.now() - new Date(parsed.fetchedAt).getTime() < CACHE_TTL_MS) {
        return { events: parsed.events, fromCache: true, fetchedAt: parsed.fetchedAt };
      }
    }
  } catch { /* ignore */ }

  // ① バッチ収集データ（GitHub Actions生成、最優先）
  let batchEvents: EventItem[] = [];
  try {
    batchEvents = await fetchBatchEvents(filter);
    console.log(`[Events] バッチ: ${batchEvents.length}件`);
  } catch (e) {
    console.warn('[Events] バッチ取得失敗:', e);
  }

  // バッチで十分あればそのまま返す（3件以上）
  if (batchEvents.length >= 3) {
    const fetchedAt = new Date().toISOString();
    await AsyncStorage.setItem(key, JSON.stringify({ events: batchEvents, fetchedAt })).catch(() => {});
    return { events: batchEvents, fromCache: false, fetchedAt };
  }

  // ② RSS フィード（バッチ補完 / バッチ未設定時のメイン）
  let rssEvents: EventItem[] = [];
  try {
    rssEvents = await fetchRssEvents(filter);
    console.log(`[Events] RSS: ${rssEvents.length}件`);
  } catch (e) {
    console.warn('[Events] RSS失敗:', e);
  }

  // バッチ + RSS をマージ
  const batchKeys = new Set(batchEvents.map((e) => `${e.title}_${e.date}`));
  const mergedBatchRss = [
    ...batchEvents,
    ...rssEvents.filter((e) => !batchKeys.has(`${e.title}_${e.date}`)),
  ].sort((a, b) => a.date.localeCompare(b.date));

  // 5件以上あればそのまま返す
  if (mergedBatchRss.length >= 5) {
    const fetchedAt = new Date().toISOString();
    await AsyncStorage.setItem(key, JSON.stringify({ events: mergedBatchRss, fetchedAt })).catch(() => {});
    return { events: mergedBatchRss, fromCache: false, fetchedAt };
  }

  // ③ 件数不足なら Gemini で補完（コスト最小化）
  try {
    const geminiEvents = await fetchEventsFromGemini(filter);
    console.log(`[Events] Gemini: ${geminiEvents.length}件`);
    const existingKeys = new Set(mergedBatchRss.map((e) => `${e.title}_${e.date}`));
    const merged = [
      ...mergedBatchRss,
      ...geminiEvents.filter((e) => !existingKeys.has(`${e.title}_${e.date}`)),
    ].sort((a, b) => a.date.localeCompare(b.date));

    const fetchedAt = new Date().toISOString();
    if (merged.length > 0) {
      await AsyncStorage.setItem(key, JSON.stringify({ events: merged, fetchedAt })).catch(() => {});
      return { events: merged, fromCache: false, fetchedAt };
    }
  } catch (e) {
    console.error('[Events] Gemini失敗:', e);
    if (mergedBatchRss.length > 0) {
      const fetchedAt = new Date().toISOString();
      await AsyncStorage.setItem(key, JSON.stringify({ events: mergedBatchRss, fetchedAt })).catch(() => {});
      return { events: mergedBatchRss, fromCache: false, fetchedAt };
    }
  }

  return { events: SAMPLE_EVENTS_FALLBACK(filter), fromCache: false, fetchedAt: null };
}

export async function clearEventsCache(filter?: EventFilter): Promise<void> {
  if (filter) {
    await AsyncStorage.removeItem(cacheKey(filter)).catch(() => {});
  } else {
    const keys = await AsyncStorage.getAllKeys().catch(() => [] as string[]);
    const ks = keys.filter((k) => k.startsWith('events_'));
    if (ks.length > 0) await AsyncStorage.multiRemove(ks).catch(() => {});
  }
}

export function getUpcomingEvents(events: EventItem[]): EventItem[] {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const week = new Date(today);
  week.setDate(today.getDate() + 7);
  return events
    .filter((e) => { const d = new Date(e.date); return d >= today && d <= week; })
    .sort((a, b) => a.date.localeCompare(b.date));
}

export function getMarkedDates(events: EventItem[]): Record<string, { marked: boolean; dotColor: string }> {
  const marks: Record<string, { marked: boolean; dotColor: string }> = {};
  events.forEach((e) => {
    const def = getThemeDef(e.themeId);
    marks[e.date] = { marked: true, dotColor: def.color };
  });
  return marks;
}
