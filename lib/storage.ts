import AsyncStorage from '@react-native-async-storage/async-storage';

export type Badge = {
  id: string;
  name: string;
  emoji: string;
  unlockedAt: string;
};

export type UserStats = {
  totalInsects: number;
  points: number;
  level: number;
  badges: Badge[];
  lastDiscoveryDates: string[]; // ISO date strings for streak tracking
  gachaHistory: GachaResult[];
};

export type GachaResult = {
  result: 'jackpot' | 'hit' | 'miss';
  insectName: string;
  earnedAt: string;
};

export type GrowthStage = {
  label: string;            // 例: '卵', '幼虫', 'さなぎ', '成虫'
  emoji: string;
  photoUrl: string | null;  // Wikimedia Commons から取得した画像URL
  season: string;           // 例: '春〜夏'
};

export type InsectEntry = {
  id: string;
  photos: string[];       // 撮影写真（複数）
  name: string;
  scientificName: string;
  features: string;
  capturedAt: string;
  location: string;       // 例: '東京都渋谷区'
  latitude?: number;      // 撮影地点の緯度
  longitude?: number;     // 撮影地点の経度
  season: string;         // 例: '春'
  confidence: number;     // 0〜1
  growthStages: GrowthStage[];
  themeId?: string;       // 例: 'creature'（未設定は creature とみなす）
};

export type AppSettings = {
  region: string;       // 例: "東京都渋谷区"
  prefecture: string;   // 例: "東京都"
  theme: string;        // legacy（後方互換）
  themes: string[];     // 選択中テーマID配列 例: ['creature', 'train']
  onboardingDone: boolean;
};

// 旧設定との後方互換: themes が未設定なら theme から復元
export function resolveThemes(settings: AppSettings | null): string[] {
  if (!settings) return ['creature'];
  if (settings.themes && settings.themes.length > 0) return settings.themes;
  // legacy: theme='mushi' → creature
  return ['creature'];
}

const INSECTS_KEY = 'insects';
const SETTINGS_KEY = 'settings';
const USER_STATS_KEY = 'user_stats';

// 図鑑エントリ
export async function getInsects(): Promise<InsectEntry[]> {
  const json = await AsyncStorage.getItem(INSECTS_KEY);
  if (!json) return [];
  const list: any[] = JSON.parse(json);
  // 旧データ（photoUri）を photos 配列に移行
  return list.map((e) => ({
    ...e,
    photos: e.photos ?? (e.photoUri ? [e.photoUri] : []),
  }));
}

// 同名の虫があれば写真を追加、なければ新規作成。更新後のエントリを返す
export async function mergeOrSaveInsect(newEntry: InsectEntry): Promise<InsectEntry> {
  const list = await getInsects();
  const idx = list.findIndex((e) => e.name === newEntry.name);
  if (idx >= 0) {
    const merged: InsectEntry = {
      ...list[idx],
      photos: [...list[idx].photos, ...newEntry.photos],
    };
    list[idx] = merged;
    await AsyncStorage.setItem(INSECTS_KEY, JSON.stringify(list));
    return merged;
  }
  await AsyncStorage.setItem(INSECTS_KEY, JSON.stringify([newEntry, ...list]));
  return newEntry;
}

export async function deleteInsect(id: string): Promise<void> {
  const list = await getInsects();
  await AsyncStorage.setItem(
    INSECTS_KEY,
    JSON.stringify(list.filter((e) => e.id !== id))
  );
}

// アプリ設定
export async function getSettings(): Promise<AppSettings | null> {
  const json = await AsyncStorage.getItem(SETTINGS_KEY);
  return json ? JSON.parse(json) : null;
}

export async function saveSettings(settings: AppSettings): Promise<void> {
  await AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

// ---- User Stats ----

const ALL_BADGES: Omit<Badge, 'unlockedAt'>[] = [
  { id: 'first_find',   name: 'はじめての発見', emoji: '🔍' },
  { id: 'ten_insects',  name: '10匹達成',       emoji: '🌟' },
  { id: 'twenty_insects', name: '20匹達成',     emoji: '🏆' },
  { id: 'jackpot',      name: '大当たり獲得',   emoji: '🎰' },
  { id: 'week_streak',  name: '7日連続で発見',  emoji: '📅' },
];

export function getAllBadgeDefinitions(): Omit<Badge, 'unlockedAt'>[] {
  return ALL_BADGES;
}

export async function getUserStats(): Promise<UserStats> {
  const json = await AsyncStorage.getItem(USER_STATS_KEY);
  if (json) return JSON.parse(json) as UserStats;
  return {
    totalInsects: 0,
    points: 0,
    level: 1,
    badges: [],
    lastDiscoveryDates: [],
    gachaHistory: [],
  };
}

function calcLevel(totalInsects: number): number {
  return Math.floor(totalInsects / 5) + 1;
}

function isConsecutive7Days(dates: string[]): boolean {
  if (dates.length < 7) return false;
  const unique = [...new Set(dates)].sort();
  const last7 = unique.slice(-7);
  for (let i = 1; i < last7.length; i++) {
    const prev = new Date(last7[i - 1]);
    const curr = new Date(last7[i]);
    const diff = (curr.getTime() - prev.getTime()) / (1000 * 60 * 60 * 24);
    if (diff !== 1) return false;
  }
  return true;
}

export async function updateUserStats(opts: {
  newInsect: boolean;
  gachaResult?: GachaResult;
  insectName?: string;
}): Promise<UserStats> {
  const stats = await getUserStats();
  const today = new Date().toISOString().slice(0, 10);

  if (opts.newInsect) {
    stats.totalInsects += 1;
    stats.points += 5;
    stats.lastDiscoveryDates = [...stats.lastDiscoveryDates, today];
  }

  if (opts.gachaResult) {
    stats.gachaHistory = [opts.gachaResult, ...stats.gachaHistory].slice(0, 20);
    if (opts.gachaResult.result === 'jackpot') stats.points += 20;
    else if (opts.gachaResult.result === 'hit') stats.points += 10;
  }

  stats.level = calcLevel(stats.totalInsects);

  // Badge checks
  const existingIds = new Set(stats.badges.map((b) => b.id));
  const now = new Date().toISOString();

  function unlock(id: string) {
    if (existingIds.has(id)) return;
    const def = ALL_BADGES.find((b) => b.id === id);
    if (def) {
      stats.badges.push({ ...def, unlockedAt: now });
      existingIds.add(id);
    }
  }

  if (stats.totalInsects >= 1) unlock('first_find');
  if (stats.totalInsects >= 10) unlock('ten_insects');
  if (stats.totalInsects >= 20) unlock('twenty_insects');
  if (opts.gachaResult?.result === 'jackpot') unlock('jackpot');
  if (isConsecutive7Days(stats.lastDiscoveryDates)) unlock('week_streak');

  await AsyncStorage.setItem(USER_STATS_KEY, JSON.stringify(stats));
  return stats;
}

export async function addGachaResult(result: GachaResult): Promise<void> {
  const stats = await getUserStats();
  stats.gachaHistory = [result, ...stats.gachaHistory].slice(0, 20);
  if (result.result === 'jackpot') stats.points += 20;
  else if (result.result === 'hit') stats.points += 10;
  // unlock jackpot badge
  const existingIds = new Set(stats.badges.map((b) => b.id));
  if (result.result === 'jackpot' && !existingIds.has('jackpot')) {
    const def = ALL_BADGES.find((b) => b.id === 'jackpot');
    if (def) stats.badges.push({ ...def, unlockedAt: new Date().toISOString() });
  }
  await AsyncStorage.setItem(USER_STATS_KEY, JSON.stringify(stats));
}
