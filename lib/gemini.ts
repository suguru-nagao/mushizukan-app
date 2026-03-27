import { GrowthStage } from './storage';

const INAT_API = 'https://api.inaturalist.org/v1';
const INAT_TOKEN = process.env.EXPO_PUBLIC_INAT_TOKEN;
const GEMINI_KEY = process.env.EXPO_PUBLIC_GEMINI_API_KEY;
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_KEY}`;

// 指定ミリ秒待機
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Gemini テキストAPIを呼ぶ共通関数（429時は最大3回リトライ）
async function callGemini(prompt: string, maxTokens = 200): Promise<string> {
  const delays = [5000, 15000, 30000]; // 5秒 → 15秒 → 30秒
  let lastError = '';

  for (let attempt = 0; attempt <= delays.length; attempt++) {
    const res = await fetch(GEMINI_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: maxTokens, temperature: 0.7 },
      }),
    });

    if (res.ok) {
      const data = await res.json();
      return data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? '';
    }

    const body = await res.text().catch(() => '');
    lastError = `Gemini error ${res.status}: ${body.slice(0, 100)}`;

    // 429以外はリトライしない
    if (res.status !== 429) throw new Error(lastError);

    // 最後の試行後はスローせずループを抜ける
    if (attempt < delays.length) {
      console.warn(`Gemini 429 - ${delays[attempt] / 1000}秒後にリトライ (${attempt + 1}/${delays.length})`);
      await sleep(delays[attempt]);
    }
  }

  // 3回リトライしても失敗 → エラーをスローせず空文字を返してフォールバックへ
  console.warn('Gemini 429 リトライ上限。フォールバックします:', lastError);
  return '';
}

// 学名 or 日本語名 → 日本語名・特徴・必殺技を1回のGeminiリクエストで取得
async function generateInsectContent(
  scientificName: string,
  jaNameFromInat: string
): Promise<{ name: string; features: string }> {
  try {
    const nameHint = jaNameFromInat
      ? `日本語名はすでに「${jaNameFromInat}」とわかっています。`
      : `学名は「${scientificName}」です。まず日本語の和名（例：カブトムシ）を特定してください。`;

    const prompt = `アプリ「Zuku-Zuku」の虫カード用コンテンツを作成してください。
${nameHint}

以下のJSON形式で返してください（日本語のみ、前置き不要）：
{
  "name": "${jaNameFromInat || ''}（和名がなければ学名をそのまま）",
  "features": "たとえ話を1つ使った3文の特徴説明。見た目・大きさ・生態を含む。具体的な数字や比較を使う。むずかしい漢字はひらがなに。絶対に「知ってた？」「さがしてみてね」「みつけてみてね」などの前置きや締めの言葉を含めないこと。例：カブトムシのツノは、工事現場のクレーン車みたいに力持ちなんだよ！体の色は黒くてつやつやしていて、大きいオスは8センチにもなるんだ。夏の夜にくぬぎの木のしるをなめにくるよ！"
}`;

    const raw = await callGemini(prompt, 450);
    const match = raw.match(/\{[\s\S]*\}/);
    if (!match) throw new Error('JSON not found');
    const parsed = JSON.parse(match[0]);
    return {
      name: parsed.name || jaNameFromInat || scientificName,
      features: parsed.features ?? '',
    };
  } catch (e) {
    console.error('generateInsectContent error:', e);
    return {
      name: jaNameFromInat || scientificName,
      features: '',
    };
  }
}

export type InsectAnalysis = {
  name: string;
  scientificName: string;
  features: string;
  confidence: number;
  taxonId: number;
  growthStages: GrowthStage[];
};

// iNaturalist ライフステージ term_value_id（term_id=1）
// keyword → termValueId のマッピング
const KEYWORD_TO_TERM_ID: Record<string, number> = {
  egg:      7,
  larva:    6,
  pupa:     4,
  adult:    2,
  nymph:    5,
  juvenile: 8,
  naiad:    5,
};

type StageDef = {
  label: string;   // 日本語（例: 卵、稚貝、成貝）
  emoji: string;
  season: string;  // 例: 6月〜8月
  keyword: string; // 英語検索キーワード（例: egg, larva, adult）
};

// Gemini に生き物の種類を問わず正しい成長ステージを生成させる
async function fetchStageDefsFromGemini(name: string, scientificName: string): Promise<StageDef[]> {
  const prompt = `「${name}」（学名: ${scientificName}）の正しい成長過程を教えてください。
昆虫に限らず、カタツムリ・クモ・ムカデなど何でも正確に答えてください。

以下のJSON配列のみ返してください（前置き不要）:
[
  {
    "label": "ステージ名（日本語）",
    "emoji": "絵文字1文字",
    "season": "X月〜Y月（日本の気候）",
    "keyword": "英語キーワード（egg/larva/pupa/adult/nymph/juvenile から最も近いもの）"
  }
]

ルール:
- この生き物に正しいステージのみ（2〜5個）
- カタツムリなら「卵→稚貝→成貝」など正確に
- 昆虫なら変態タイプに応じて正確に`;

  try {
    const raw = await callGemini(prompt, 400);
    const match = raw.match(/\[[\s\S]*\]/);
    if (match) {
      const parsed: StageDef[] = JSON.parse(match[0]);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {
    // フォールバック
  }

  // Gemini 失敗時のデフォルト（完全変態）
  return [
    { label: '卵',    emoji: '🥚', season: '4月〜7月',  keyword: 'egg'   },
    { label: '幼虫',  emoji: '🐛', season: '5月〜8月',  keyword: 'larva' },
    { label: 'さなぎ', emoji: '🫘', season: '6月〜8月',  keyword: 'pupa'  },
    { label: '成虫',  emoji: '🦋', season: '4月〜10月', keyword: 'adult' },
  ];
}

// ① iNaturalist: taxon_id + ライフステージで絞り込み
async function fetchInatStagePhoto(taxonId: number, termValueId: number): Promise<string | null> {
  try {
    const url =
      `${INAT_API}/observations?taxon_id=${taxonId}&term_id=1&term_value_id=${termValueId}` +
      `&photos=true&per_page=3&order_by=votes&quality_grade=research`;
    const res = await fetch(url, { headers: { Authorization: `Bearer ${INAT_TOKEN}` } });
    if (!res.ok) return null;
    const data = await res.json();
    const photo = data.results?.[0]?.photos?.[0];
    return photo ? photo.url.replace('square', 'medium') : null;
  } catch {
    return null;
  }
}

// ② Wikimedia Commons: キーワード検索
async function searchWikimedia(query: string): Promise<string | null> {
  try {
    const url =
      `https://commons.wikimedia.org/w/api.php?action=query` +
      `&generator=search&gsrsearch=${encodeURIComponent(query)}` +
      `&gsrnamespace=6&gsrlimit=5&prop=imageinfo&iiprop=url|mime&format=json&origin=*`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    const pages: any[] = Object.values(data.query?.pages ?? {});
    for (const page of pages) {
      const info = page.imageinfo?.[0];
      if (info?.url && (info.mime === 'image/jpeg' || info.mime === 'image/png')) {
        return info.url;
      }
    }
    return null;
  } catch {
    return null;
  }
}

// ③ Wikipedia: 種ページのトップ画像（成虫代表写真）
async function fetchWikipediaTopImage(scientificName: string): Promise<string | null> {
  try {
    const url =
      `https://en.wikipedia.org/w/api.php?action=query` +
      `&titles=${encodeURIComponent(scientificName)}` +
      `&prop=pageimages&pithumbsize=400&format=json&origin=*`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    const pages: any[] = Object.values(data.query?.pages ?? {});
    return pages[0]?.thumbnail?.source ?? null;
  } catch {
    return null;
  }
}

// 3段階フォールバックで写真を取得
async function fetchStagePhotoWithFallback(
  taxonId: number,
  keyword: string,
  scientificName: string,
  fallbackNames: string[],
): Promise<string | null> {
  const termValueId = KEYWORD_TO_TERM_ID[keyword] ?? KEYWORD_TO_TERM_ID['adult'];

  // ① iNaturalist（ステージ完全一致）
  const inat = await fetchInatStagePhoto(taxonId, termValueId);
  if (inat) return inat;

  // ② Wikimedia Commons（学名 + キーワード）
  const wiki = await searchWikimedia(`${scientificName} ${keyword}`);
  if (wiki) return wiki;

  // ② Wikimedia（目・科名でリトライ）
  for (const name of fallbackNames) {
    const wiki2 = await searchWikimedia(`${name} ${keyword}`);
    if (wiki2) return wiki2;
  }

  // ③ Wikipedia トップ画像（代表写真で代替）
  return fetchWikipediaTopImage(scientificName);
}

async function fetchGrowthStages(scientificName: string, insectName: string, taxonId: number, ancestors: { name: string; rank: string }[]): Promise<GrowthStage[]> {
  // 目・科の学名をフォールバック候補として抽出
  const fallbackNames = ancestors
    .filter((a) => a.rank === 'order' || a.rank === 'family')
    .map((a) => a.name);

  // Gemini でステージ定義（ラベル・絵文字・季節・キーワード）を取得
  const stageDefs = await fetchStageDefsFromGemini(insectName, scientificName);

  // 写真取得を並列実行
  const photoUrls = await Promise.all(
    stageDefs.map((stage) =>
      fetchStagePhotoWithFallback(taxonId, stage.keyword, scientificName, fallbackNames)
    )
  );

  return stageDefs.map((stage, i) => ({
    label: stage.label,
    emoji: stage.emoji,
    season: stage.season,
    photoUrl: photoUrls[i],
  }));
}

export type ChatCharacter = 'doctor' | 'friend';

export async function chatWithBugCharacter(
  message: string,
  character: ChatCharacter,
  history: { role: string; text: string }[]
): Promise<string> {
  const persona =
    character === 'doctor'
      ? 'あなたは「虫博士」というキャラクターです。子供向けに虫の質問に答えます。語尾は「〜ですよ」「〜なのです」などの丁寧語を使い、知識豊富で穏やかな口調で話してください。難しい漢字はひらがなにしてください。'
      : 'あなたは「むしむしフレンド」というキャラクターです。子供の友達として虫の質問に答えます。語尾は「〜だよ！」「すごいね！」などの元気な口調を使い、明るく楽しく話してください。絵文字も使ってOKです。難しい漢字はひらがなにしてください。';

  const historyText = history
    .slice(-6)
    .map((h) => `${h.role === 'user' ? 'ユーザー' : 'キャラクター'}: ${h.text}`)
    .join('\n');

  const prompt = `${persona}

これまでの会話:
${historyText}

ユーザーの質問: ${message}

キャラクターとして返答してください（2〜4文、日本語のみ）:`;

  return callGemini(prompt, 300);
}

export async function analyzeInsectPhoto(photoUri: string): Promise<InsectAnalysis> {
  const formData = new FormData();
  formData.append('image', {
    uri: photoUri,
    type: 'image/jpeg',
    name: 'photo.jpg',
  } as any);

  const scoreRes = await fetch(
    `${INAT_API}/computervision/score_image?locale=ja&nb_results=1`,
    {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${INAT_TOKEN}` },
      body: formData,
    }
  );

  if (!scoreRes.ok) {
    const bodyText = await scoreRes.text().catch(() => '');
    throw new Error(`iNaturalist error: ${scoreRes.status} - ${bodyText.slice(0, 200)}`);
  }

  const scoreData = await scoreRes.json();
  const topResult = scoreData.results?.[0];

  if (!topResult || topResult.combined_score < 0.2) {
    return {
      name: '虫がみつかりませんでした',
      scientificName: '',
      features: '写真に虫が写っていないか、識別できませんでした。もう少し近づいて撮ってみてね！',
      confidence: 0,
      taxonId: 0,
      growthStages: [],
    };
  }

  const taxon = topResult.taxon;
  const scientificName: string = taxon.name ?? '';
  const confidence: number = Math.round((topResult.combined_score ?? 0) * 100);

  const jaNameFromInat: string = taxon.preferred_common_name ?? '';

  // タクソン詳細取得（祖先情報）と Gemini コンテンツ生成を並列実行（Gemini は1回のみ）
  const [taxonDetail, insectContent] = await Promise.all([
    fetch(
      `${INAT_API}/taxa/${taxon.id}?locale=ja&all_names=true`,
      { headers: { 'Authorization': `Bearer ${INAT_TOKEN}` } }
    )
      .then((r) => r.json())
      .then((d) => d.results?.[0])
      .catch(() => null),
    generateInsectContent(scientificName, jaNameFromInat),
  ]);

  const name = insectContent.name || jaNameFromInat || scientificName;
  const ancestors: { name: string; rank: string }[] =
    taxonDetail?.ancestors?.map((a: any) => ({ name: a.name, rank: a.rank })) ?? [];

  const growthStages = await fetchGrowthStages(scientificName, name, taxon.id, ancestors);

  const features = insectContent.features || `${name}はとても不思議な生き物だよ。じっくり観察してみてね！`;

  return { name, scientificName, features, confidence, taxonId: taxon.id, growthStages };
}
