import { useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  ActivityIndicator, Alert, SafeAreaView, ScrollView,
} from 'react-native';
import * as Location from 'expo-location';
import { useRouter } from 'expo-router';
import { saveSettings } from '../lib/storage';
import { THEME_DEFS } from '../lib/themes';

export default function OnboardingScreen() {
  const router = useRouter();
  const [step, setStep] = useState<'theme' | 'location'>('theme');
  const [selectedThemes, setSelectedThemes] = useState<string[]>(['creature']);
  const [loading, setLoading] = useState(false);

  function toggleTheme(id: string) {
    setSelectedThemes((prev) =>
      prev.includes(id)
        ? prev.length > 1 ? prev.filter((t) => t !== id) : prev // 最低1つは選択
        : [...prev, id]
    );
  }

  async function handleLocationAndStart() {
    setLoading(true);
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      let region = '東京都';
      let prefecture = '東京都';

      if (status === 'granted') {
        const loc = await Location.getCurrentPositionAsync({});
        const [address] = await Location.reverseGeocodeAsync({
          latitude: loc.coords.latitude,
          longitude: loc.coords.longitude,
        });
        const reg = address.region ?? '';
        const city = address.city ?? address.subregion ?? '';
        prefecture = reg || '東京都';
        region = (reg + city) || address.name || `${loc.coords.latitude.toFixed(3)},${loc.coords.longitude.toFixed(3)}`;
      } else {
        Alert.alert('位置情報を取得できませんでした', 'デフォルトの地域（東京都）でイベントを表示します。');
      }

      await saveSettings({
        region,
        prefecture,
        theme: selectedThemes[0],
        themes: selectedThemes,
        onboardingDone: true,
      });

      router.replace('/(tabs)/calendar');
    } catch {
      Alert.alert('エラー', '設定の保存に失敗しました。もう一度お試しください。');
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.emoji}>🐛</Text>
        <Text style={styles.title}>むし図鑑へようこそ！</Text>
        <Text style={styles.subtitle}>こどもの「好き」を応援しよう</Text>

        {step === 'theme' && (
          <>
            <Text style={styles.sectionTitle}>テーマを選んでね</Text>
            <Text style={styles.sectionHint}>複数選んでOK！</Text>
            <View style={styles.themeGrid}>
              {THEME_DEFS.map((t) => {
                const active = selectedThemes.includes(t.id);
                return (
                  <TouchableOpacity
                    key={t.id}
                    style={[
                      styles.themeCard,
                      active && { borderColor: t.color, backgroundColor: t.bgColor },
                    ]}
                    onPress={() => toggleTheme(t.id)}
                    activeOpacity={0.7}
                  >
                    {active && <Text style={[styles.checkmark, { color: t.color }]}>✓</Text>}
                    <Text style={styles.themeEmoji}>{t.emoji}</Text>
                    <Text style={[styles.themeLabel, active && { color: t.color }]}>
                      {t.label}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
            <TouchableOpacity style={styles.button} onPress={() => setStep('location')}>
              <Text style={styles.buttonText}>つぎへ →</Text>
            </TouchableOpacity>
          </>
        )}

        {step === 'location' && (
          <>
            <Text style={styles.sectionTitle}>あなたの地域を教えてね</Text>
            <Text style={styles.description}>
              位置情報を使って、近くのイベントを{'\n'}カレンダーに表示します。
            </Text>
            <Text style={styles.descriptionSmall}>
              ※ 位置情報はこの端末にのみ保存され、{'\n'}外部に送信されることはありません。
            </Text>
            {loading ? (
              <ActivityIndicator size="large" color="#16a34a" style={{ marginTop: 32 }} />
            ) : (
              <>
                <TouchableOpacity style={styles.button} onPress={handleLocationAndStart}>
                  <Text style={styles.buttonText}>📍 位置情報を許可してはじめる</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.skipButton}
                  onPress={async () => {
                    await saveSettings({
                      region: '東京都',
                      prefecture: '東京都',
                      theme: selectedThemes[0],
                      themes: selectedThemes,
                      onboardingDone: true,
                    });
                    router.replace('/(tabs)/calendar');
                  }}
                >
                  <Text style={styles.skipText}>スキップする（東京都で表示）</Text>
                </TouchableOpacity>
              </>
            )}
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f0fdf4' },
  container: { flexGrow: 1, alignItems: 'center', padding: 24, paddingTop: 60 },
  emoji: { fontSize: 72, marginBottom: 12 },
  title: { fontSize: 26, fontWeight: 'bold', color: '#14532d', textAlign: 'center', marginBottom: 8 },
  subtitle: { fontSize: 16, color: '#d97706', fontWeight: '600', marginBottom: 40 },
  sectionTitle: { fontSize: 18, fontWeight: 'bold', color: '#166534', marginBottom: 6 },
  sectionHint: { fontSize: 13, color: '#9ca3af', marginBottom: 20 },
  themeGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginBottom: 32 },
  themeCard: {
    width: 140, height: 120, backgroundColor: '#fff', borderRadius: 16,
    alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: '#d1fae5',
  },
  checkmark: { position: 'absolute', top: 8, right: 12, fontSize: 18, fontWeight: 'bold' },
  themeEmoji: { fontSize: 40, marginBottom: 4 },
  themeLabel: { fontSize: 15, fontWeight: '600', color: '#6b7280' },
  description: { fontSize: 16, color: '#374151', textAlign: 'center', lineHeight: 26, marginBottom: 12 },
  descriptionSmall: { fontSize: 12, color: '#9ca3af', textAlign: 'center', lineHeight: 20, marginBottom: 32 },
  button: { backgroundColor: '#16a34a', paddingVertical: 16, paddingHorizontal: 32, borderRadius: 32, marginBottom: 12 },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  skipButton: { paddingVertical: 12 },
  skipText: { color: '#9ca3af', fontSize: 14 },
});
