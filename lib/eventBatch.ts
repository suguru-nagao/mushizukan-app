/**
 * eventBatch.ts
 * ─────────────
 * GitHub Actions バッチで生成された events.json を取得する。
 * URL は環境変数 EXPO_PUBLIC_BATCH_EVENTS_URL で設定する。
 *
 * 設定例 (.env):
 *   EXPO_PUBLIC_BATCH_EVENTS_URL=https://raw.githubusercontent.com/YOUR_USER/mushizukan-app/main/public/events.json
 */

import { EventItem, EventFilter } from './events';

const BATCH_URL = process.env.EXPO_PUBLIC_BATCH_EVENTS_URL ?? '';

type BatchEvent = {
  id: string;
  title: string;
  date: string;
  location: string;
  description: string;
  prefecture: string;
  city?: string;
  themeId: string;
  source?: string;
  url?: string;
  fetched_at?: string;
};

type BatchResponse = {
  generated_at: string;
  count: number;
  events: BatchEvent[];
};

export async function fetchBatchEvents(filter: EventFilter): Promise<EventItem[]> {
  if (!BATCH_URL) {
    console.log('[Batch] EXPO_PUBLIC_BATCH_EVENTS_URL が未設定、スキップ');
    return [];
  }

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);

    const res = await fetch(BATCH_URL, {
      headers: { 'Cache-Control': 'no-cache' },
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data: BatchResponse = await res.json();
    const allEvents = data.events ?? [];

    // テーマフィルタ
    const themeFiltered = allEvents.filter((e) =>
      filter.themes.includes(e.themeId ?? 'train')
    );

    // 日付フィルタ（今日〜30日後）
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const future = new Date(today);
    future.setDate(today.getDate() + 30);

    const result: EventItem[] = themeFiltered
      .filter((e) => {
        const d = new Date(e.date);
        return d >= today && d <= future;
      })
      .map((e) => ({
        id: e.id,
        title: e.title,
        date: e.date,
        // location は「都道府県市区町村 施設名」形式で来る想定
        location: e.location || e.prefecture || '日本',
        description: e.description || e.title,
        prefecture: e.prefecture || '',
        themeId: e.themeId,
        source: e.source,
        url: e.url || undefined,
      }));

    console.log(`[Batch] ${result.length}件 (全${allEvents.length}件中)`);
    return result;
  } catch (e: unknown) {
    if (e instanceof Error && e.name === 'AbortError') {
      console.warn('[Batch] タイムアウト');
    } else {
      console.warn('[Batch] 取得失敗:', e);
    }
    return [];
  }
}
