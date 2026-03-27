import { useState, useCallback, useRef, useEffect } from 'react';
import {
  View, Text, StyleSheet, FlatList, TouchableOpacity,
  Image, Alert, ActivityIndicator, SafeAreaView, Modal, ScrollView,
  Dimensions,
} from 'react-native';
import { THEME_DEFS } from '../../lib/themes';
import MapView, { Marker, Callout } from 'react-native-maps';
import * as ImagePicker from 'expo-image-picker';
import { useFocusEffect } from 'expo-router';
import { getInsects, mergeOrSaveInsect, deleteInsect, InsectEntry, GrowthStage, updateUserStats, addGachaResult, GachaResult } from '../../lib/storage';
import { analyzeInsectPhoto } from '../../lib/gemini';
import { getCaptureLocation, getSeason } from '../../lib/location';
import { speakText, stopSpeaking, isSpeaking, preloadSpeech } from '../../lib/tts';
import GachaModal, { GachaResultType } from '../../components/GachaModal';
import { getNearbyInsects, NearbyInsect } from '../../lib/nearby';

function rollGacha(): GachaResultType {
  const r = Math.random();
  if (r < 0.10) return 'jackpot';
  if (r < 0.40) return 'hit';
  return 'miss';
}

export default function ZukanScreen() {
  const [insects, setInsects] = useState<InsectEntry[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [selected, setSelected] = useState<InsectEntry | null>(null);
  const [mapVisible, setMapVisible] = useState(false);
  const [gachaVisible, setGachaVisible] = useState(false);
  const [gachaResult, setGachaResult] = useState<GachaResultType | null>(null);
  const [nearbyInsects, setNearbyInsects] = useState<NearbyInsect[]>([]);
  const [nearbyLoading, setNearbyLoading] = useState(false);
  const [activeThemeFilter, setActiveThemeFilter] = useState<string | null>(null); // null=すべて
  const pendingEntry = useRef<InsectEntry | null>(null);

  useFocusEffect(
    useCallback(() => { loadInsects(); }, [])
  );

  useEffect(() => {
    loadNearby();
  }, []);

  async function loadNearby() {
    setNearbyLoading(true);
    try {
      setNearbyInsects(await getNearbyInsects());
    } catch {
      setNearbyInsects([]);
    } finally {
      setNearbyLoading(false);
    }
  }

  async function loadInsects() {
    setInsects(await getInsects());
  }

  async function handleCapture() {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('カメラのアクセスが必要です', '設定からカメラのアクセスを許可してください。');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.7 });
    if (!result.canceled && result.assets[0]) await analyze(result.assets[0].uri);
  }

  async function handlePickFromLibrary() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('写真ライブラリのアクセスが必要です');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.7 });
    if (!result.canceled && result.assets[0]) await analyze(result.assets[0].uri);
  }

  async function analyze(uri: string) {
    setAnalyzing(true);
    try {
      const [analysis, location] = await Promise.all([
        analyzeInsectPhoto(uri),
        getCaptureLocation(),
      ]);

      const entry: InsectEntry = {
        id: Date.now().toString(),
        photos: [uri],
        name: analysis.name,
        scientificName: analysis.scientificName,
        features: analysis.features,
        capturedAt: new Date().toLocaleDateString('ja-JP'),
        location: location.label,
        latitude: location.latitude,
        longitude: location.longitude,
        season: getSeason(),
        confidence: analysis.confidence,
        growthStages: analysis.growthStages,
        themeId: 'creature',
      };
      const saved = await mergeOrSaveInsect(entry);
      await loadInsects();

      // Update user stats for new insect
      const isNew = saved.id === entry.id; // new entry has same id (not merged)
      await updateUserStats({ newInsect: isNew });

      // Show gacha
      const roll = rollGacha();
      const gachaRecord: GachaResult = {
        result: roll,
        insectName: saved.name,
        earnedAt: new Date().toISOString(),
      };
      await addGachaResult(gachaRecord);

      pendingEntry.current = saved;
      setGachaResult(roll);
      setGachaVisible(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      Alert.alert('分析エラー', msg);
    } finally {
      setAnalyzing(false);
    }
  }

  function handleGachaNext() {
    setGachaVisible(false);
    if (pendingEntry.current) {
      setSelected(pendingEntry.current);
      pendingEntry.current = null;
    }
  }

  async function handleDelete(id: string) {
    Alert.alert('削除しますか？', 'この虫を図鑑から削除します。', [
      { text: 'キャンセル', style: 'cancel' },
      {
        text: '削除', style: 'destructive',
        onPress: async () => {
          await deleteInsect(id);
          setSelected(null);
          await loadInsects();
        },
      },
    ]);
  }

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>📖 じぶん図鑑</Text>
        <Text style={styles.headerSub}>{insects.length}匹みつけた！</Text>
      </View>

      {analyzing && (
        <View style={styles.analyzingBanner}>
          <ActivityIndicator color="#fff" size="small" />
          <Text style={styles.analyzingText}>虫を調べています...</Text>
        </View>
      )}

      {/* テーマフィルター */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.themeFilterBar}
        contentContainerStyle={styles.themeFilterContent}
      >
        <TouchableOpacity
          style={[styles.themeFilterChip, activeThemeFilter === null && styles.themeFilterChipAll]}
          onPress={() => setActiveThemeFilter(null)}
        >
          <Text style={[styles.themeFilterText, activeThemeFilter === null && styles.themeFilterTextAll]}>
            すべて
          </Text>
        </TouchableOpacity>
        {THEME_DEFS.map((t) => {
          const active = activeThemeFilter === t.id;
          const count = insects.filter((i) => (i.themeId ?? 'creature') === t.id).length;
          return (
            <TouchableOpacity
              key={t.id}
              style={[styles.themeFilterChip, active && { borderColor: t.color, backgroundColor: t.bgColor }]}
              onPress={() => setActiveThemeFilter(active ? null : t.id)}
            >
              <Text style={[styles.themeFilterText, active && { color: t.color }]}>
                {t.emoji} {t.label}
                {count > 0 && <Text style={styles.themeFilterCount}> {count}</Text>}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>

      <FlatList
        data={activeThemeFilter ? insects.filter((i) => (i.themeId ?? 'creature') === activeThemeFilter) : insects}
        keyExtractor={(item) => item.id}
        numColumns={2}
        contentContainerStyle={styles.list}
        ListEmptyComponent={<EmptyState />}
        ListFooterComponent={
          <NearbyInsectsSection
            insects={nearbyInsects}
            loading={nearbyLoading}
            onRefresh={loadNearby}
          />
        }
        renderItem={({ item }) => (
          <TouchableOpacity style={styles.card} onPress={() => setSelected(item)}>
            <Image source={{ uri: item.photos[0] }} style={styles.cardImage} />
            {item.photos.length > 1 && (
              <View style={styles.photoCountBadge}>
                <Text style={styles.photoCountText}>📷 {item.photos.length}</Text>
              </View>
            )}
            <View style={styles.cardBody}>
              <Text style={styles.cardName} numberOfLines={1}>{item.name}</Text>
              <Text style={styles.cardMeta}>{item.season} · {item.location}</Text>
            </View>
          </TouchableOpacity>
        )}
      />

      <View style={styles.fab}>
        <TouchableOpacity style={styles.fabButton} onPress={handleCapture} disabled={analyzing}>
          <Text style={styles.fabIcon}>📷</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.fabButton, styles.fabButtonSecondary]} onPress={handlePickFromLibrary} disabled={analyzing}>
          <Text style={styles.fabIcon}>🖼️</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.fabButton, styles.fabButtonMap]} onPress={() => setMapVisible(true)}>
          <Text style={styles.fabIcon}>🗺️</Text>
        </TouchableOpacity>
      </View>

      <Modal visible={!!selected} animationType="slide" presentationStyle="pageSheet">
        {selected && <DetailModal entry={selected} onClose={() => setSelected(null)} onDelete={handleDelete} />}
      </Modal>

      <GachaModal
        visible={gachaVisible}
        result={gachaResult}
        onNext={handleGachaNext}
      />

      <Modal visible={mapVisible} animationType="slide" presentationStyle="pageSheet">
        <InsectMapModal
          insects={insects}
          onSelect={(item) => { setMapVisible(false); setSelected(item); }}
          onClose={() => setMapVisible(false)}
        />
      </Modal>
    </SafeAreaView>
  );
}

function InsectMapModal({ insects, onSelect, onClose }: {
  insects: InsectEntry[];
  onSelect: (item: InsectEntry) => void;
  onClose: () => void;
}) {
  const mapped = insects.filter((i) => i.latitude && i.longitude);
  const initialRegion = mapped.length > 0
    ? {
        latitude: mapped[0].latitude!,
        longitude: mapped[0].longitude!,
        latitudeDelta: 0.5,
        longitudeDelta: 0.5,
      }
    : { latitude: 35.6812, longitude: 139.7671, latitudeDelta: 5, longitudeDelta: 5 };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={styles.mapHeader}>
        <Text style={styles.mapTitle}>🗺️ 発見マップ</Text>
        <TouchableOpacity onPress={onClose} style={styles.mapCloseBtn}>
          <Text style={styles.mapCloseTxt}>✕ とじる</Text>
        </TouchableOpacity>
      </View>

      {mapped.length === 0 ? (
        <View style={styles.mapEmpty}>
          <Text style={styles.emptyEmoji}>📍</Text>
          <Text style={styles.emptyTitle}>まだ場所の記録がないよ</Text>
          <Text style={styles.emptyDesc}>次に虫を撮るとマップにのるよ！</Text>
        </View>
      ) : (
        <MapView style={{ flex: 1 }} initialRegion={initialRegion} showsUserLocation>
          {mapped.map((item) => (
            <Marker
              key={item.id}
              coordinate={{ latitude: item.latitude!, longitude: item.longitude! }}
              onCalloutPress={() => onSelect(item)}
            >
              <View style={styles.markerPin}>
                <Text style={styles.markerEmoji}>🐛</Text>
              </View>
              <Callout tooltip>
                <View style={styles.callout}>
                  <Image source={{ uri: item.photos[0] }} style={styles.calloutImage} />
                  <Text style={styles.calloutName}>{item.name}</Text>
                  <Text style={styles.calloutDate}>{item.capturedAt}</Text>
                  <Text style={styles.calloutTap}>タップで詳細 →</Text>
                </View>
              </Callout>
            </Marker>
          ))}
        </MapView>
      )}
    </SafeAreaView>
  );
}

function DetailModal({ entry, onClose, onDelete }: {
  entry: InsectEntry;
  onClose: () => void;
  onDelete: (id: string) => void;
}) {
  const [zoomPhoto, setZoomPhoto] = useState<string | null>(null);
  const [photoIndex, setPhotoIndex] = useState(0);
  const [speakState, setSpeakState] = useState<'idle' | 'loading' | 'playing'>('idle');
  const screenWidth = Dimensions.get('window').width;
  const cacheKey = `tts_${entry.id}`;
  const speakLock = useRef(false);

  // モーダルを開いた瞬間にバックグラウンドで音声を事前生成
  useEffect(() => {
    preloadSpeech(cacheKey, entry.features);
  }, []);

  async function handleSpeak() {
    if (speakLock.current) return; // 多重押し防止
    if (speakState === 'playing') {
      stopSpeaking();
      setSpeakState('idle');
      return;
    }
    speakLock.current = true;
    setSpeakState('loading');
    try {
      await speakText(entry.features, cacheKey, () => setSpeakState('idle'));
      setSpeakState('playing');
    } catch {
      setSpeakState('idle');
    } finally {
      speakLock.current = false;
    }
  }

  return (
    <SafeAreaView style={styles.modal}>
      <ScrollView>
        {/* 写真カルーセル */}
        <View>
          <FlatList
            data={entry.photos}
            horizontal
            pagingEnabled
            showsHorizontalScrollIndicator={false}
            keyExtractor={(_, i) => i.toString()}
            onMomentumScrollEnd={(e) => {
              const idx = Math.round(e.nativeEvent.contentOffset.x / screenWidth);
              setPhotoIndex(idx);
            }}
            renderItem={({ item }) => (
              <TouchableOpacity onPress={() => setZoomPhoto(item)} activeOpacity={0.9}>
                <Image source={{ uri: item }} style={[styles.modalImage, { width: screenWidth }]} />
                <View style={styles.zoomHint}>
                  <Text style={styles.zoomHintText}>🔍 タップで拡大</Text>
                </View>
              </TouchableOpacity>
            )}
          />
          {entry.photos.length > 1 && (
            <View style={styles.dotsContainer}>
              {entry.photos.map((_, i) => (
                <View key={i} style={[styles.dot, i === photoIndex && styles.dotActive]} />
              ))}
            </View>
          )}
        </View>

        <View style={styles.modalBody}>
          <Text style={styles.modalName}>{entry.name}</Text>

          <View style={styles.badgeRow}>
            <BadgeChip icon="📅" text={entry.capturedAt} />
            <BadgeChip icon="📍" text={entry.location} />
            {entry.confidence > 0 && (
              <BadgeChip icon="🎯" text={`確信度 ${entry.confidence}%`} />
            )}
          </View>

          {/* Features section with TTS button */}
          <View style={styles.featuresTitleRow}>
            <SectionTitle title="🔍 特徴" />
            <TouchableOpacity
              style={[
                styles.speakBtn,
                speakState === 'loading' && styles.speakBtnLoading,
                speakState === 'playing' && styles.speakBtnActive,
              ]}
              onPress={handleSpeak}
              disabled={speakState === 'loading'}
            >
              <Text style={styles.speakBtnText}>
                {speakState === 'loading' ? '⏳ よみこみ中...' : speakState === 'playing' ? '⏹ 停止' : '🔊 よむ'}
              </Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.featuresText}>{entry.features}</Text>

          {entry.growthStages.length > 0 && (
            <>
              <SectionTitle title="🌱 成長過程" />
              <Text style={styles.stageNote}>※ 季節は目安です。タップで拡大できます。</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.stagesScroll}>
                {entry.growthStages.map((stage, i) => (
                  <View key={i} style={styles.stageRow}>
                    <StageCard stage={stage} onZoom={setZoomPhoto} />
                    {i < entry.growthStages.length - 1 && (
                      <View style={styles.arrowContainer}>
                        <Text style={styles.arrow}>→</Text>
                      </View>
                    )}
                  </View>
                ))}
              </ScrollView>
              <Text style={styles.stageCredit}>写真: Wikimedia Commons（CC）</Text>
            </>
          )}
        </View>
      </ScrollView>

      <View style={styles.modalFooter}>
        <TouchableOpacity style={styles.deleteButton} onPress={() => onDelete(entry.id)}>
          <Text style={styles.deleteButtonText}>🗑️ 削除</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Text style={styles.closeButtonText}>とじる</Text>
        </TouchableOpacity>
      </View>

      {/* 拡大表示モーダル */}
      <Modal visible={!!zoomPhoto} transparent animationType="fade">
        <TouchableOpacity style={styles.zoomOverlay} onPress={() => setZoomPhoto(null)} activeOpacity={1}>
          {zoomPhoto && (
            <Image
              source={{ uri: zoomPhoto }}
              style={styles.zoomImage}
              resizeMode="contain"
            />
          )}
          <Text style={styles.zoomClose}>✕ とじる</Text>
        </TouchableOpacity>
      </Modal>
    </SafeAreaView>
  );
}

function StageCard({ stage, onZoom }: { stage: GrowthStage; onZoom: (url: string) => void }) {
  return (
    <View style={styles.stageCard}>
      <TouchableOpacity
        onPress={() => stage.photoUrl && onZoom(stage.photoUrl)}
        activeOpacity={stage.photoUrl ? 0.8 : 1}
      >
        {stage.photoUrl ? (
          <>
            <Image source={{ uri: stage.photoUrl }} style={styles.stageImage} />
            <View style={styles.stageZoomBadge}>
              <Text style={styles.stageZoomText}>🔍</Text>
            </View>
          </>
        ) : (
          <View style={styles.stageImagePlaceholder}>
            <Text style={{ fontSize: 36 }}>{stage.emoji}</Text>
          </View>
        )}
      </TouchableOpacity>
      <Text style={styles.stageLabel}>{stage.emoji} {stage.label}</Text>
      <Text style={styles.stageSeason}>📅 {stage.season}</Text>
    </View>
  );
}

function BadgeChip({ icon, text }: { icon: string; text: string }) {
  return (
    <View style={styles.badge}>
      <Text style={styles.badgeText}>{icon} {text}</Text>
    </View>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <Text style={styles.sectionTitle}>{title}</Text>;
}

function NearbyInsectsSection({
  insects,
  loading,
  onRefresh,
}: {
  insects: NearbyInsect[];
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <View style={styles.nearbySection}>
      <View style={styles.nearbySectionHeader}>
        <Text style={styles.nearbySectionTitle}>📍 このあたりで見られる虫</Text>
        <TouchableOpacity onPress={onRefresh} disabled={loading}>
          <Text style={styles.nearbyRefreshText}>{loading ? '読込中...' : '↻ 更新'}</Text>
        </TouchableOpacity>
      </View>
      {loading ? (
        <ActivityIndicator color="#16a34a" style={{ paddingVertical: 20 }} />
      ) : insects.length === 0 ? (
        <Text style={styles.nearbyEmpty}>近くの虫情報が見つかりませんでした</Text>
      ) : (
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {insects.map((insect) => (
            <View key={insect.id} style={styles.nearbyCard}>
              {insect.photoUrl ? (
                <Image source={{ uri: insect.photoUrl }} style={styles.nearbyPhoto} />
              ) : (
                <View style={styles.nearbyPhotoPlaceholder}>
                  <Text style={{ fontSize: 32 }}>🐛</Text>
                </View>
              )}
              <Text style={styles.nearbyName} numberOfLines={2}>{insect.name}</Text>
              {insect.observedOn ? (
                <Text style={styles.nearbyDate}>{insect.observedOn}</Text>
              ) : null}
            </View>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

function EmptyState() {
  return (
    <View style={styles.empty}>
      <Text style={styles.emptyEmoji}>🔍</Text>
      <Text style={styles.emptyTitle}>まだ虫がいないよ</Text>
      <Text style={styles.emptyDesc}>カメラボタンを押して{'\n'}虫の写真を撮ってみよう！</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f0fdf4' },
  header: {
    paddingHorizontal: 20, paddingTop: 16, paddingBottom: 8,
    backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#e5e7eb',
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  headerTitle: { fontSize: 20, fontWeight: 'bold', color: '#14532d' },
  headerSub: { fontSize: 13, color: '#16a34a', fontWeight: '600' },
  themeFilterBar: { backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#e5e7eb', maxHeight: 52 },
  themeFilterContent: { paddingHorizontal: 12, paddingVertical: 10, gap: 8, flexDirection: 'row', alignItems: 'center' },
  themeFilterChip: {
    paddingHorizontal: 12, paddingVertical: 5, borderRadius: 20,
    borderWidth: 1.5, borderColor: '#d1d5db', backgroundColor: '#f9fafb',
  },
  themeFilterChipAll: { borderColor: '#16a34a', backgroundColor: '#dcfce7' },
  themeFilterText: { fontSize: 13, color: '#6b7280', fontWeight: '600' },
  themeFilterTextAll: { color: '#16a34a' },
  themeFilterCount: { fontSize: 11, fontWeight: '400' },
  analyzingBanner: {
    backgroundColor: '#16a34a', flexDirection: 'row', alignItems: 'center',
    justifyContent: 'center', gap: 8, paddingVertical: 10,
  },
  analyzingText: { color: '#fff', fontWeight: '600' },
  list: { padding: 12, paddingBottom: 0 },
  card: {
    flex: 1, margin: 6, backgroundColor: '#fff', borderRadius: 12,
    overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08, shadowRadius: 4, elevation: 2,
  },
  cardImage: { width: '100%', aspectRatio: 1 },
  cardBody: { padding: 8 },
  cardName: { fontSize: 13, fontWeight: 'bold', color: '#111827' },
  cardMeta: { fontSize: 11, color: '#9ca3af', marginTop: 2 },
  fab: { position: 'absolute', bottom: 24, right: 20, flexDirection: 'column', gap: 10, alignItems: 'center' },
  fabButton: {
    width: 60, height: 60, borderRadius: 30, backgroundColor: '#16a34a',
    alignItems: 'center', justifyContent: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.2, shadowRadius: 6, elevation: 5,
  },
  fabButtonSecondary: { backgroundColor: '#d97706', width: 52, height: 52, borderRadius: 26 },
  fabButtonMap: { backgroundColor: '#2563eb', width: 52, height: 52, borderRadius: 26 },
  mapHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 20, paddingVertical: 14,
    borderBottomWidth: 1, borderBottomColor: '#e5e7eb',
  },
  mapTitle: { fontSize: 18, fontWeight: 'bold', color: '#14532d' },
  mapCloseBtn: { paddingHorizontal: 12, paddingVertical: 6, backgroundColor: '#f3f4f6', borderRadius: 20 },
  mapCloseTxt: { fontSize: 14, color: '#374151', fontWeight: '600' },
  mapEmpty: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  markerPin: {
    backgroundColor: '#16a34a', borderRadius: 20, padding: 6,
    borderWidth: 2, borderColor: '#fff',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4,
  },
  markerEmoji: { fontSize: 18 },
  callout: {
    backgroundColor: '#fff', borderRadius: 12, padding: 10, width: 160,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6,
  },
  calloutImage: { width: '100%', height: 90, borderRadius: 8, marginBottom: 6 },
  calloutName: { fontSize: 14, fontWeight: 'bold', color: '#14532d', marginBottom: 2 },
  calloutDate: { fontSize: 11, color: '#9ca3af', marginBottom: 4 },
  calloutTap: { fontSize: 11, color: '#16a34a', fontWeight: '600' },
  fabIcon: { fontSize: 26 },
  empty: { flex: 1, alignItems: 'center', paddingTop: 80 },
  emptyEmoji: { fontSize: 64, marginBottom: 16 },
  emptyTitle: { fontSize: 18, fontWeight: 'bold', color: '#166534', marginBottom: 8 },
  emptyDesc: { fontSize: 14, color: '#6b7280', textAlign: 'center', lineHeight: 22 },
  // モーダル
  modal: { flex: 1, backgroundColor: '#f0fdf4' },
  modalImage: { width: '100%', height: 280 },
  modalBody: { padding: 20 },
  modalName: { fontSize: 26, fontWeight: 'bold', color: '#14532d', marginBottom: 4 },
  modalScientific: { fontSize: 14, color: '#6b7280', fontStyle: 'italic', marginBottom: 12 },
  badgeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 20 },
  badge: { backgroundColor: '#dcfce7', borderRadius: 20, paddingHorizontal: 10, paddingVertical: 4 },
  badgeText: { fontSize: 12, color: '#166534', fontWeight: '600' },
  featuresTitleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 },
  sectionTitle: { fontSize: 16, fontWeight: 'bold', color: '#166534', marginTop: 4 },
  speakBtn: {
    backgroundColor: '#dcfce7', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 6,
    borderWidth: 1, borderColor: '#16a34a',
  },
  speakBtnLoading: { backgroundColor: '#fef9c3', borderColor: '#ca8a04' },
  speakBtnActive: { backgroundColor: '#16a34a' },
  speakBtnText: { fontSize: 13, color: '#166534', fontWeight: '600' },
  featuresText: { fontSize: 15, color: '#374151', lineHeight: 24, marginBottom: 16 },
  photoCountBadge: {
    position: 'absolute', top: 8, right: 8,
    backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: 10, paddingHorizontal: 6, paddingVertical: 2,
  },
  photoCountText: { color: '#fff', fontSize: 11, fontWeight: '600' },
  dotsContainer: {
    flexDirection: 'row', justifyContent: 'center', alignItems: 'center',
    paddingVertical: 8, gap: 6, backgroundColor: '#000',
  },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: 'rgba(255,255,255,0.4)' },
  dotActive: { backgroundColor: '#fff', width: 8, height: 8, borderRadius: 4 },
  zoomHint: {
    position: 'absolute', bottom: 10, right: 12,
    backgroundColor: 'rgba(0,0,0,0.45)', borderRadius: 12, paddingHorizontal: 8, paddingVertical: 4,
  },
  zoomHintText: { color: '#fff', fontSize: 11 },
  stageNote: { fontSize: 11, color: '#9ca3af', marginBottom: 10 },
  stagesScroll: { marginBottom: 8 },
  stageRow: { flexDirection: 'row', alignItems: 'center' },
  arrowContainer: { paddingHorizontal: 4, paddingBottom: 24 },
  arrow: { fontSize: 22, color: '#16a34a', fontWeight: 'bold' },
  stageCard: { width: 110, alignItems: 'center' },
  stageImage: { width: 100, height: 100, borderRadius: 12, marginBottom: 6 },
  stageImagePlaceholder: {
    width: 100, height: 100, borderRadius: 12, marginBottom: 6,
    backgroundColor: '#f3f4f6', alignItems: 'center', justifyContent: 'center',
  },
  stageZoomBadge: {
    position: 'absolute', bottom: 10, right: 4,
    backgroundColor: 'rgba(0,0,0,0.4)', borderRadius: 10, paddingHorizontal: 5, paddingVertical: 2,
  },
  stageZoomText: { fontSize: 11 },
  stageLabel: { fontSize: 13, fontWeight: '600', color: '#374151' },
  stageSeason: { fontSize: 11, color: '#16a34a', marginTop: 2 },
  stageCredit: { fontSize: 10, color: '#9ca3af', marginBottom: 16 },
  // このあたりで見られる虫
  nearbySection: { paddingHorizontal: 12, paddingBottom: 100, marginTop: 8 },
  nearbySectionHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10,
  },
  nearbySectionTitle: { fontSize: 15, fontWeight: 'bold', color: '#166534' },
  nearbyRefreshText: { fontSize: 13, color: '#16a34a', fontWeight: '600' },
  nearbyEmpty: { color: '#9ca3af', textAlign: 'center', paddingVertical: 16, fontSize: 13 },
  nearbyCard: {
    width: 110, backgroundColor: '#fff', borderRadius: 12, marginRight: 10,
    overflow: 'hidden',
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06, shadowRadius: 4, elevation: 2,
  },
  nearbyPhoto: { width: 110, height: 90 },
  nearbyPhotoPlaceholder: {
    width: 110, height: 90, backgroundColor: '#f3f4f6',
    alignItems: 'center', justifyContent: 'center',
  },
  nearbyName: { fontSize: 11, fontWeight: '600', color: '#111827', padding: 6, paddingBottom: 2 },
  nearbyDate: { fontSize: 10, color: '#9ca3af', paddingHorizontal: 6, paddingBottom: 6 },
  // 拡大モーダル
  zoomOverlay: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.92)',
    alignItems: 'center', justifyContent: 'center',
  },
  zoomImage: { width: Dimensions.get('window').width, height: Dimensions.get('window').height * 0.8 },
  zoomClose: { color: '#fff', fontSize: 16, marginTop: 20, fontWeight: '600' },
  modalFooter: {
    flexDirection: 'row', padding: 16, gap: 12,
    borderTopWidth: 1, borderTopColor: '#e5e7eb', backgroundColor: '#fff',
  },
  deleteButton: { flex: 1, paddingVertical: 14, borderRadius: 12, backgroundColor: '#fee2e2', alignItems: 'center' },
  deleteButtonText: { color: '#dc2626', fontWeight: 'bold', fontSize: 15 },
  closeButton: { flex: 2, paddingVertical: 14, borderRadius: 12, backgroundColor: '#16a34a', alignItems: 'center' },
  closeButtonText: { color: '#fff', fontWeight: 'bold', fontSize: 15 },
});
