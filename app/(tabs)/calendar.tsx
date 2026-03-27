import { useEffect, useState, useRef } from 'react';
import {
  View, Text, StyleSheet, ScrollView, SafeAreaView,
  TouchableOpacity, ActivityIndicator, Linking, Alert,
} from 'react-native';
import * as Location from 'expo-location';
import { Calendar, LocaleConfig } from 'react-native-calendars';
import { getSettings, saveSettings, AppSettings, resolveThemes } from '../../lib/storage';
import {
  getEventsWithCache, clearEventsCache, getUpcomingEvents,
  getMarkedDates, SAMPLE_EVENTS_FALLBACK,
  EventItem, EventFilter, TransportMode,
} from '../../lib/events';
import { THEME_DEFS, getThemeDef } from '../../lib/themes';

LocaleConfig.locales['ja'] = {
  monthNames: ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],
  monthNamesShort: ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],
  dayNames: ['日曜日','月曜日','火曜日','水曜日','木曜日','金曜日','土曜日'],
  dayNamesShort: ['日','月','火','水','木','金','土'],
  today: '今日',
};
LocaleConfig.defaultLocale = 'ja';

const MINUTE_OPTIONS = [10, 20, 30, 40, 50, 60];

export default function CalendarScreen() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedEvents, setSelectedEvents] = useState<EventItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [transport, setTransport] = useState<TransportMode>('train');
  const [travelMinutes, setTravelMinutes] = useState(30);
  const [activeThemes, setActiveThemes] = useState<string[]>(['creature']);

  const settingsRef = useRef<AppSettings | null>(null);
  const initialized = useRef(false);

  useEffect(() => {
    (async () => {
      const s = await getSettings();
      const themes = resolveThemes(s);
      setSettings(s);
      settingsRef.current = s;
      setActiveThemes(themes);
      initialized.current = true;
      await loadEvents(s, transport, travelMinutes, themes, false);
    })();
  }, []);

  async function loadEvents(
    s: AppSettings | null,
    t: TransportMode,
    minutes: number,
    themes: string[],
    forceRefresh: boolean,
  ) {
    const filter: EventFilter = {
      transport: t,
      minutes,
      region: s?.region ?? '東京都',
      prefecture: s?.prefecture ?? '東京都',
      themes,
    };
    setLoading(true);
    setError(null);
    try {
      if (forceRefresh) await clearEventsCache(filter);
      const result = await getEventsWithCache(filter);
      setEvents(result.events);
      setFetchedAt(result.fetchedAt);
    } catch {
      setError('イベントの取得に失敗しました');
      setEvents(SAMPLE_EVENTS_FALLBACK(filter));
    } finally {
      setLoading(false);
    }
  }

  function handleRefresh() {
    loadEvents(settingsRef.current, transport, travelMinutes, activeThemes, true);
  }

  function handleTransportChange(t: TransportMode) {
    setTransport(t);
    if (initialized.current) loadEvents(settingsRef.current, t, travelMinutes, activeThemes, true);
  }

  function handleMinutesChange(m: number) {
    setTravelMinutes(m);
    if (initialized.current) loadEvents(settingsRef.current, transport, m, activeThemes, true);
  }

  async function handleThemeToggle(id: string) {
    const next = activeThemes.includes(id)
      ? activeThemes.length > 1 ? activeThemes.filter((t) => t !== id) : activeThemes
      : [...activeThemes, id];
    setActiveThemes(next);

    // 設定に保存
    if (settingsRef.current) {
      const newSettings = { ...settingsRef.current, themes: next };
      settingsRef.current = newSettings;
      setSettings(newSettings);
      await saveSettings(newSettings);
    }
    loadEvents(settingsRef.current, transport, travelMinutes, next, true);
  }

  async function handleRedetectLocation() {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('位置情報が必要です', '設定から位置情報のアクセスを許可してください。');
        return;
      }
      const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
      const [address] = await Location.reverseGeocodeAsync({ latitude: loc.coords.latitude, longitude: loc.coords.longitude });
      const reg = address.region ?? '';
      const city = address.city ?? address.subregion ?? '';
      const prefecture = reg || '不明';
      const region = (reg + city) || `${loc.coords.latitude.toFixed(3)},${loc.coords.longitude.toFixed(3)}`;
      const newSettings: AppSettings = { ...(settingsRef.current ?? { theme: 'creature', themes: activeThemes, onboardingDone: true }), region, prefecture };
      await saveSettings(newSettings);
      settingsRef.current = newSettings;
      setSettings(newSettings);
      await clearEventsCache();
      await loadEvents(newSettings, transport, travelMinutes, activeThemes, false);
      Alert.alert('地域を更新しました', `📍 ${region}`);
    } catch {
      Alert.alert('エラー', '位置情報の取得に失敗しました。');
    }
  }

  // 表示するイベントはアクティブテーマのみに絞る（キャッシュの混入防止）
  const filteredEvents = events.filter((e) => activeThemes.includes(e.themeId));
  const upcomingEvents = getUpcomingEvents(filteredEvents);

  const markedDates = {
    ...getMarkedDates(filteredEvents),
    ...(selectedDate ? {
      [selectedDate]: {
        selected: true, selectedColor: '#16a34a',
        marked: events.some((e) => e.date === selectedDate), dotColor: '#fff',
      },
    } : {}),
  };

  function onDayPress(day: { dateString: string }) {
    setSelectedDate(day.dateString);
    setSelectedEvents(filteredEvents.filter((e) => e.date === day.dateString));
  }

  return (
    <SafeAreaView style={styles.root}>
      {/* ヘッダー */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>📅 お出かけカレンダー</Text>
          <TouchableOpacity onPress={handleRedetectLocation}>
            <Text style={styles.headerSub}>
              📍 {settings?.region ?? '地域未設定'}
              <Text style={styles.headerSubHint}>  （タップで更新）</Text>
            </Text>
          </TouchableOpacity>
        </View>
        <TouchableOpacity style={styles.refreshBtn} onPress={handleRefresh} disabled={loading}>
          {loading
            ? <ActivityIndicator size="small" color="#16a34a" />
            : <Text style={styles.refreshBtnText}>↻ 更新</Text>}
        </TouchableOpacity>
      </View>

      {/* フィルターバー */}
      <View style={styles.filterBar}>
        {/* テーマ選択 */}
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>テーマ</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.chipScroll}>
            {THEME_DEFS.map((t) => {
              const active = activeThemes.includes(t.id);
              return (
                <TouchableOpacity
                  key={t.id}
                  style={[styles.themeChip, active && { borderColor: t.color, backgroundColor: t.bgColor }]}
                  onPress={() => handleThemeToggle(t.id)}
                >
                  <Text style={[styles.themeChipText, active && { color: t.color }]}>
                    {t.emoji} {t.label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </View>

        {/* 移動手段 */}
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>移動手段</Text>
          <View style={styles.transportToggle}>
            {(['train', 'car'] as TransportMode[]).map((mode) => (
              <TouchableOpacity
                key={mode}
                style={[styles.transportBtn, transport === mode && styles.transportBtnActive]}
                onPress={() => handleTransportChange(mode)}
              >
                <Text style={[styles.transportBtnText, transport === mode && styles.transportBtnTextActive]}>
                  {mode === 'car' ? '🚗 車' : '🚃 電車'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* 所要時間 */}
        <View style={styles.filterRow}>
          <Text style={styles.filterLabel}>所要時間</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.chipScroll}>
            {MINUTE_OPTIONS.map((min) => (
              <TouchableOpacity
                key={min}
                style={[styles.minuteChip, travelMinutes === min && styles.minuteChipActive]}
                onPress={() => handleMinutesChange(min)}
              >
                <Text style={[styles.minuteChipText, travelMinutes === min && styles.minuteChipTextActive]}>
                  {min}分
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </View>

      {fetchedAt && !loading && (
        <View style={styles.updatedBar}>
          <Text style={styles.updatedText}>📡 {new Date(fetchedAt).toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })} 更新</Text>
        </View>
      )}
      {error && <View style={styles.errorBar}><Text style={styles.errorText}>⚠️ {error}</Text></View>}

      <ScrollView>
        <Calendar
          onDayPress={onDayPress}
          markedDates={markedDates}
          theme={{
            backgroundColor: '#fff', calendarBackground: '#fff',
            selectedDayBackgroundColor: '#16a34a', selectedDayTextColor: '#fff',
            todayTextColor: '#16a34a', arrowColor: '#16a34a',
            monthTextColor: '#14532d', textDayFontWeight: '500',
            textMonthFontWeight: 'bold', textDayHeaderFontWeight: '600',
          }}
          style={styles.calendar}
        />

        <View style={styles.eventsSection}>
          {selectedDate ? (
            <>
              <Text style={styles.sectionTitle}>{selectedDate.replace(/-/g, '/')} のイベント</Text>
              {selectedEvents.length > 0
                ? selectedEvents.map((ev) => <EventCard key={ev.id} event={ev} />)
                : <Text style={styles.noEvent}>この日のイベントはありません</Text>}
            </>
          ) : (
            <>
              <Text style={styles.sectionTitle}>📆 先1週間のイベント</Text>
              {loading ? (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator color="#16a34a" />
                  <Text style={styles.loadingText}>Geminiでイベントを検索中...</Text>
                </View>
              ) : upcomingEvents.length > 0
                ? upcomingEvents.map((ev) => <EventCard key={ev.id} event={ev} />)
                : <Text style={styles.noEvent}>今週のイベントはありません</Text>}
            </>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function EventCard({ event }: { event: EventItem }) {
  const theme = getThemeDef(event.themeId);

  async function handlePress() {
    // GeminiのURLはハルシネーションが多いため、常にGoogle検索を使用
    const query = `${event.title} ${event.location} ${event.date.slice(0, 7)}`;
    const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
    try {
      await Linking.openURL(searchUrl);
    } catch {
      Alert.alert('ブラウザを開けませんでした');
    }
  }

  return (
    <TouchableOpacity
      style={[styles.eventCard, { borderLeftColor: theme.color, borderLeftWidth: 4 }]}
      onPress={handlePress}
      activeOpacity={0.75}
    >
      {/* 日付バッジ */}
      <View style={[styles.eventDateBadge, { backgroundColor: theme.bgColor }]}>
        <Text style={[styles.eventDateDay, { color: theme.color }]}>{event.date.slice(8)}</Text>
        <Text style={[styles.eventDateMonth, { color: theme.color }]}>{event.date.slice(5, 7)}月</Text>
      </View>

      <View style={styles.eventInfo}>
        {/* テーマバッジ */}
        <View style={[styles.themeBadge, { backgroundColor: theme.bgColor }]}>
          <Text style={[styles.themeBadgeText, { color: theme.color }]}>
            {theme.emoji} {theme.label}
          </Text>
        </View>
        <Text style={styles.eventTitle}>{event.title}</Text>
        <Text style={styles.eventLocation}>📍 {event.location}</Text>
        <Text style={styles.eventDesc}>{event.description}</Text>
        {event.source && (
          <Text style={styles.eventSource}>出典: {event.source}</Text>
        )}
        <View style={[styles.eventLinkBadge, { backgroundColor: theme.lightBg ?? '#eff6ff' }]}>
          <Text style={[styles.eventLinkText, { color: theme.color }]}>
            🔍 詳細を検索 →
          </Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f0fdf4' },
  header: {
    paddingHorizontal: 20, paddingTop: 16, paddingBottom: 10,
    backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#e5e7eb',
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  headerTitle: { fontSize: 20, fontWeight: 'bold', color: '#14532d' },
  headerSub: { fontSize: 13, color: '#6b7280', marginTop: 2 },
  headerSubHint: { fontSize: 11, color: '#d1d5db' },
  refreshBtn: { paddingHorizontal: 14, paddingVertical: 8, backgroundColor: '#dcfce7', borderRadius: 20, minWidth: 60, alignItems: 'center' },
  refreshBtnText: { fontSize: 14, color: '#16a34a', fontWeight: '600' },

  filterBar: {
    backgroundColor: '#fff', paddingHorizontal: 14, paddingTop: 10, paddingBottom: 10,
    borderBottomWidth: 1, borderBottomColor: '#e5e7eb', gap: 8,
  },
  filterRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  filterLabel: { fontSize: 11, color: '#9ca3af', fontWeight: '600', width: 46 },
  chipScroll: { flexShrink: 1 },

  themeChip: {
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20,
    borderWidth: 1.5, borderColor: '#d1d5db', backgroundColor: '#f9fafb', marginRight: 6,
  },
  themeChipText: { fontSize: 13, color: '#6b7280', fontWeight: '600' },

  transportToggle: { flexDirection: 'row', gap: 8 },
  transportBtn: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20, borderWidth: 1.5, borderColor: '#d1d5db', backgroundColor: '#f9fafb' },
  transportBtnActive: { borderColor: '#16a34a', backgroundColor: '#dcfce7' },
  transportBtnText: { fontSize: 13, color: '#6b7280', fontWeight: '600' },
  transportBtnTextActive: { color: '#15803d' },

  minuteChip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, borderWidth: 1.5, borderColor: '#d1d5db', backgroundColor: '#f9fafb', marginRight: 6 },
  minuteChipActive: { borderColor: '#16a34a', backgroundColor: '#dcfce7' },
  minuteChipText: { fontSize: 13, color: '#6b7280', fontWeight: '600' },
  minuteChipTextActive: { color: '#15803d' },

  updatedBar: { backgroundColor: '#f0fdf4', paddingHorizontal: 16, paddingVertical: 5, borderBottomWidth: 1, borderBottomColor: '#e5e7eb' },
  updatedText: { fontSize: 11, color: '#6b7280' },
  errorBar: { backgroundColor: '#fef2f2', paddingHorizontal: 16, paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#fecaca' },
  errorText: { fontSize: 13, color: '#dc2626' },

  calendar: { borderBottomWidth: 1, borderBottomColor: '#e5e7eb' },
  eventsSection: { padding: 16 },
  sectionTitle: { fontSize: 16, fontWeight: 'bold', color: '#166534', marginBottom: 12 },
  noEvent: { color: '#9ca3af', textAlign: 'center', paddingVertical: 24 },
  loadingContainer: { alignItems: 'center', paddingVertical: 32, gap: 10 },
  loadingText: { color: '#6b7280', fontSize: 13 },

  eventCard: {
    flexDirection: 'row', backgroundColor: '#fff', borderRadius: 14,
    padding: 12, marginBottom: 10, gap: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.07, shadowRadius: 4, elevation: 2,
    overflow: 'hidden',
  },
  eventDateBadge: { borderRadius: 10, paddingHorizontal: 8, paddingVertical: 8, alignItems: 'center', justifyContent: 'center', minWidth: 44 },
  eventDateDay: { fontSize: 20, fontWeight: 'bold', lineHeight: 24 },
  eventDateMonth: { fontSize: 10, fontWeight: '600' },
  eventInfo: { flex: 1 },
  themeBadge: { alignSelf: 'flex-start', borderRadius: 6, paddingHorizontal: 8, paddingVertical: 2, marginBottom: 4 },
  themeBadgeText: { fontSize: 11, fontWeight: '700' },
  eventTitle: { fontSize: 15, fontWeight: 'bold', color: '#111827', marginBottom: 2 },
  eventLocation: { fontSize: 12, color: '#6b7280', marginBottom: 3 },
  eventDesc: { fontSize: 13, color: '#374151', lineHeight: 18 },
  eventSource: { fontSize: 10, color: '#9ca3af', marginTop: 2, marginBottom: 2 },
  eventLinkBadge: { marginTop: 6, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4, alignSelf: 'flex-start' },
  eventLinkText: { fontSize: 12, fontWeight: '600' },
});
