import * as FileSystem from 'expo-file-system/legacy';
import { Audio } from 'expo-av';

const GEMINI_KEY = process.env.EXPO_PUBLIC_GEMINI_API_KEY ?? '';
const TTS_MODEL = 'gemini-2.5-flash-preview-tts';
const SAMPLE_RATE = 24000;

let currentSound: Audio.Sound | null = null;

// 事前生成したファイルURIをキャッシュ（エントリIDをキーに）
const audioCache = new Map<string, Promise<string | null>>();

// バックグラウンドで音声を事前生成してキャッシュに保存
export function preloadSpeech(cacheKey: string, text: string): void {
  if (audioCache.has(cacheKey)) return;
  audioCache.set(cacheKey, generateAudioFile(text));
}

// PCM（生データ）に WAV ヘッダを付けて base64 に変換
function pcmBase64ToWavBase64(pcmBase64: string): string {
  const pcmBinary = atob(pcmBase64);
  const pcmLength = pcmBinary.length;
  const header = new Uint8Array(44);
  const view = new DataView(header.buffer);

  [82, 73, 70, 70].forEach((b, i) => (header[i] = b));          // "RIFF"
  view.setUint32(4, 36 + pcmLength, true);
  [87, 65, 86, 69].forEach((b, i) => (header[8 + i] = b));      // "WAVE"
  [102, 109, 116, 32].forEach((b, i) => (header[12 + i] = b));  // "fmt "
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);               // PCM
  view.setUint16(22, 1, true);               // mono
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true); // byte rate
  view.setUint16(32, 2, true);               // block align
  view.setUint16(34, 16, true);              // 16-bit
  [100, 97, 116, 97].forEach((b, i) => (header[36 + i] = b));   // "data"
  view.setUint32(40, pcmLength, true);

  let headerStr = '';
  for (let i = 0; i < 44; i++) headerStr += String.fromCharCode(header[i]);
  return btoa(headerStr + pcmBinary);
}

async function generateAudioFile(text: string): Promise<string | null> {
  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${TTS_MODEL}:generateContent?key=${GEMINI_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text }] }],
          generationConfig: {
            responseModalities: ['AUDIO'],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: 'Aoede' },
              },
            },
          },
        }),
      }
    );
    if (!res.ok) return null;
    const data = await res.json();
    const part = data.candidates?.[0]?.content?.parts?.[0]?.inlineData;
    if (!part?.data) return null;
    const isPcm = (part.mimeType ?? '').includes('pcm');
    const audioBase64 = isPcm ? pcmBase64ToWavBase64(part.data) : part.data;
    const fileUri = `${FileSystem.cacheDirectory}tts_${Date.now()}.wav`;
    await FileSystem.writeAsStringAsync(fileUri, audioBase64, {
      encoding: FileSystem.EncodingType.Base64,
    });
    return fileUri;
  } catch {
    return null;
  }
}

export async function speakText(text: string, cacheKey?: string, onFinish?: () => void): Promise<void> {
  await stopSpeaking();
  await Audio.setAudioModeAsync({ playsInSilentModeIOS: true });

  // キャッシュがあれば即座に使用、なければ生成
  let fileUri: string | null = null;
  if (cacheKey && audioCache.has(cacheKey)) {
    fileUri = await audioCache.get(cacheKey)!;
  }
  if (!fileUri) {
    fileUri = await generateAudioFile(text);
  }
  if (!fileUri) throw new Error('音声データを取得できませんでした');

  const { sound } = await Audio.Sound.createAsync({ uri: fileUri });
  currentSound = sound;
  await sound.playAsync();

  sound.setOnPlaybackStatusUpdate((status) => {
    if (status.isLoaded && status.didJustFinish) {
      sound.unloadAsync();
      currentSound = null;
      onFinish?.(); // 再生完了を通知
    }
  });
}

export async function stopSpeaking(): Promise<void> {
  if (currentSound) {
    try {
      await currentSound.stopAsync();
      await currentSound.unloadAsync();
    } catch {}
    currentSound = null;
  }
}

export async function isSpeaking(): Promise<boolean> {
  if (!currentSound) return false;
  try {
    const status = await currentSound.getStatusAsync();
    return status.isLoaded && (status as any).isPlaying;
  } catch {
    return false;
  }
}
