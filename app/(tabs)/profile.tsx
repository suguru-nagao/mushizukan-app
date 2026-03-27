import { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import { useFocusEffect } from 'expo-router';
import {
  getUserStats,
  UserStats,
  getAllBadgeDefinitions,
  GachaResult,
} from '../../lib/storage';

export default function ProfileScreen() {
  const [stats, setStats] = useState<UserStats | null>(null);

  useFocusEffect(
    useCallback(() => {
      (async () => {
        const s = await getUserStats();
        setStats(s);
      })();
    }, [])
  );

  if (!stats) {
    return (
      <SafeAreaView style={styles.root}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>👤 プロフィール</Text>
        </View>
      </SafeAreaView>
    );
  }

  const insectsForNextLevel = 5 - (stats.totalInsects % 5);
  const progressRatio = (stats.totalInsects % 5) / 5;
  const allBadges = getAllBadgeDefinitions();
  const unlockedIds = new Set(stats.badges.map((b) => b.id));
  const recentGacha = stats.gachaHistory.slice(0, 5);

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>👤 プロフィール</Text>
      </View>

      <ScrollView contentContainerStyle={styles.scroll}>
        {/* Level card */}
        <View style={styles.levelCard}>
          <Text style={styles.levelEmoji}>🌿</Text>
          <Text style={styles.levelText}>レベル {stats.level}</Text>
          <Text style={styles.levelSub}>虫はかせ</Text>

          <View style={styles.progressBarBg}>
            <View style={[styles.progressBarFill, { width: `${progressRatio * 100}%` as any }]} />
          </View>
          <Text style={styles.progressLabel}>
            次のレベルまであと {insectsForNextLevel} 匹
          </Text>
        </View>

        {/* Stats row */}
        <View style={styles.statsRow}>
          <StatBox icon="🐛" value={stats.totalInsects} label="総発見数" />
          <StatBox icon="⭐" value={stats.points} label="ポイント" />
          <StatBox icon="🏅" value={stats.badges.length} label="バッジ数" />
        </View>

        {/* Badges */}
        <SectionTitle title="🏅 獲得バッジ" />
        <View style={styles.badgesGrid}>
          {allBadges.map((def) => {
            const unlocked = unlockedIds.has(def.id);
            const badge = stats.badges.find((b) => b.id === def.id);
            return (
              <View
                key={def.id}
                style={[styles.badgeCard, !unlocked && styles.badgeCardLocked]}
              >
                <Text style={[styles.badgeEmoji, !unlocked && styles.badgeEmojiLocked]}>
                  {def.emoji}
                </Text>
                <Text style={[styles.badgeName, !unlocked && styles.badgeNameLocked]}>
                  {def.name}
                </Text>
                {badge && (
                  <Text style={styles.badgeDate}>
                    {badge.unlockedAt.slice(0, 10)}
                  </Text>
                )}
                {!unlocked && (
                  <Text style={styles.badgeLockText}>？？？</Text>
                )}
              </View>
            );
          })}
        </View>

        {/* Gacha history */}
        <SectionTitle title="🎰 ガチャ結果（直近5件）" />
        {recentGacha.length === 0 ? (
          <Text style={styles.emptyText}>まだガチャをしていないよ</Text>
        ) : (
          recentGacha.map((g, i) => <GachaHistoryRow key={i} item={g} />)
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function StatBox({ icon, value, label }: { icon: string; value: number; label: string }) {
  return (
    <View style={styles.statBox}>
      <Text style={styles.statIcon}>{icon}</Text>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <Text style={styles.sectionTitle}>{title}</Text>;
}

function GachaHistoryRow({ item }: { item: GachaResult }) {
  const cfg: Record<GachaResult['result'], { emoji: string; label: string; color: string }> = {
    jackpot: { emoji: '🎊', label: '大当たり', color: '#b45309' },
    hit:     { emoji: '⭐', label: 'あたり',   color: '#16a34a' },
    miss:    { emoji: '💨', label: 'ハズレ',   color: '#6b7280' },
  };
  const c = cfg[item.result];
  return (
    <View style={styles.gachaRow}>
      <Text style={styles.gachaEmoji}>{c.emoji}</Text>
      <View style={styles.gachaInfo}>
        <Text style={styles.gachaInsect}>{item.insectName}</Text>
        <Text style={styles.gachaDate}>{item.earnedAt.slice(0, 10)}</Text>
      </View>
      <Text style={[styles.gachaLabel, { color: c.color }]}>{c.label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f0fdf4' },
  header: {
    paddingHorizontal: 20, paddingTop: 16, paddingBottom: 8,
    backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#e5e7eb',
  },
  headerTitle: { fontSize: 20, fontWeight: 'bold', color: '#14532d' },
  scroll: { padding: 16, paddingBottom: 40 },
  levelCard: {
    backgroundColor: '#16a34a', borderRadius: 20, padding: 24,
    alignItems: 'center', marginBottom: 16,
  },
  levelEmoji: { fontSize: 48, marginBottom: 4 },
  levelText: { fontSize: 30, fontWeight: 'bold', color: '#fff' },
  levelSub: { fontSize: 14, color: '#bbf7d0', marginBottom: 16 },
  progressBarBg: {
    width: '100%', height: 12, backgroundColor: 'rgba(255,255,255,0.3)',
    borderRadius: 6, overflow: 'hidden', marginBottom: 6,
  },
  progressBarFill: {
    height: '100%', backgroundColor: '#fff', borderRadius: 6,
  },
  progressLabel: { fontSize: 13, color: '#dcfce7' },
  statsRow: { flexDirection: 'row', gap: 10, marginBottom: 20 },
  statBox: {
    flex: 1, backgroundColor: '#fff', borderRadius: 14, padding: 14,
    alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06, shadowRadius: 4, elevation: 2,
  },
  statIcon: { fontSize: 26, marginBottom: 4 },
  statValue: { fontSize: 22, fontWeight: 'bold', color: '#14532d' },
  statLabel: { fontSize: 11, color: '#6b7280', marginTop: 2 },
  sectionTitle: { fontSize: 16, fontWeight: 'bold', color: '#166534', marginBottom: 12, marginTop: 4 },
  badgesGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 20 },
  badgeCard: {
    width: '30%', backgroundColor: '#fff', borderRadius: 14, padding: 12,
    alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06, shadowRadius: 4, elevation: 2,
  },
  badgeCardLocked: { backgroundColor: '#f3f4f6' },
  badgeEmoji: { fontSize: 30, marginBottom: 4 },
  badgeEmojiLocked: { opacity: 0.3 },
  badgeName: { fontSize: 11, fontWeight: '600', color: '#374151', textAlign: 'center' },
  badgeNameLocked: { color: '#9ca3af' },
  badgeDate: { fontSize: 10, color: '#9ca3af', marginTop: 2 },
  badgeLockText: { fontSize: 13, color: '#9ca3af', fontWeight: 'bold', marginTop: 2 },
  gachaRow: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#fff',
    borderRadius: 12, padding: 12, marginBottom: 8, gap: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06, shadowRadius: 4, elevation: 2,
  },
  gachaEmoji: { fontSize: 28 },
  gachaInfo: { flex: 1 },
  gachaInsect: { fontSize: 14, fontWeight: '600', color: '#111827' },
  gachaDate: { fontSize: 11, color: '#9ca3af', marginTop: 2 },
  gachaLabel: { fontSize: 13, fontWeight: 'bold' },
  emptyText: { color: '#9ca3af', textAlign: 'center', paddingVertical: 20 },
});
