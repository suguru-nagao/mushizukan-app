export type ThemeId = 'creature' | 'train' | 'pokemon' | 'car';

export type ThemeDef = {
  id: ThemeId;
  label: string;
  emoji: string;
  color: string;
  bgColor: string;
  lightBg: string;
  keywords: string[];
  // イベント情報を持つ公式サイトドメイン（Gemini検索の優先ターゲット）
  officialDomains: string[];
};

export const THEME_DEFS: ThemeDef[] = [
  {
    id: 'creature',
    label: '生き物',
    emoji: '🐛',
    color: '#16a34a',
    bgColor: '#dcfce7',
    lightBg: '#f0fdf4',
    keywords: ['生き物', '自然観察', '昆虫', '動物', '水族館', '動物園'],
    officialDomains: [
      'nhk.or.jp',           // NHK for School 自然番組イベント
      'museum.or.jp',        // 博物館系
      'env.go.jp',           // 環境省 自然観察
      'city.*.lg.jp',        // 各市区町村（自然観察会）
      'shinrinkyoku.jp',     // 森林浴・自然観察
      'insect.or.jp',        // 日本昆虫協会
    ],
  },
  {
    id: 'train',
    label: '電車',
    emoji: '🚃',
    color: '#2563eb',
    bgColor: '#dbeafe',
    lightBg: '#eff6ff',
    keywords: ['電車', '鉄道', '列車', '鉄道博物館', 'ミニ電車', '乗り鉄'],
    officialDomains: [
      'jreast.co.jp',        // JR東日本
      'jrcentral.co.jp',     // JR東海
      'jrwest.co.jp',        // JR西日本
      'odakyu.jp',           // 小田急電鉄
      'tokyu.co.jp',         // 東急電鉄
      'keio.co.jp',          // 京王電鉄
      'seibu-railway.co.jp', // 西武鉄道
      'tobu.co.jp',          // 東武鉄道
      'sotetsu.co.jp',       // 相模鉄道
      'keikyu.co.jp',        // 京急電鉄
      'kintetsu.co.jp',      // 近畿日本鉄道
      'hankyu.co.jp',        // 阪急電鉄
      'railway-museum.jp',   // 鉄道博物館（大宮）
      'scmaglev-railway.com',// リニア・鉄道館
    ],
  },
  {
    id: 'pokemon',
    label: 'ポケモン',
    emoji: '⭐',
    color: '#ca8a04',
    bgColor: '#fef9c3',
    lightBg: '#fefce8',
    keywords: ['ポケモン', 'Pokemon', 'ポケットモンスター', 'ポケモンセンター'],
    officialDomains: [
      'pokemon.co.jp',       // ポケモン公式
      'pokemoncenter-online.com', // ポケモンセンター
      'bandai.co.jp',        // バンダイ（ポケモングッズ・イベント）
    ],
  },
  {
    id: 'car',
    label: '車',
    emoji: '🚗',
    color: '#dc2626',
    bgColor: '#fee2e2',
    lightBg: '#fef2f2',
    keywords: ['車', '自動車', 'ミニカー', 'モーターショー', 'キッズカー'],
    officialDomains: [
      'toyota.jp',           // トヨタ
      'honda.co.jp',         // ホンダ
      'nissan.co.jp',        // 日産
      'mazda.co.jp',         // マツダ
      'autobacs.co.jp',      // オートバックス
      'jaos.or.jp',          // 日本自動車用品協会
      'motorshow.or.jp',     // モーターショー
      'jaaa.ne.jp',          // 自動車関連
    ],
  },
];

export function getThemeDef(id: string): ThemeDef {
  return THEME_DEFS.find((t) => t.id === id) ?? THEME_DEFS[0];
}

export function getActiveThemes(themes?: string[]): ThemeDef[] {
  if (!themes || themes.length === 0) return [THEME_DEFS[0]];
  return THEME_DEFS.filter((t) => themes.includes(t.id));
}
