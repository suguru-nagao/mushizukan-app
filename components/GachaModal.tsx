import { useEffect, useRef } from 'react';
import {
  Modal,
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
  Easing,
} from 'react-native';

export type GachaResultType = 'jackpot' | 'hit' | 'miss';

type Props = {
  visible: boolean;
  result: GachaResultType | null;
  onNext: () => void;
};

const RESULT_CONFIG: Record<
  GachaResultType,
  { emoji: string; label: string; message: string; color: string; bg: string }
> = {
  jackpot: {
    emoji: '🎊',
    label: '大当たり！',
    message: 'レアバッジ獲得！',
    color: '#b45309',
    bg: '#fef3c7',
  },
  hit: {
    emoji: '⭐',
    label: 'あたり！',
    message: 'ポイント+10！',
    color: '#16a34a',
    bg: '#dcfce7',
  },
  miss: {
    emoji: '💨',
    label: 'ハズレ...',
    message: 'また挑戦してね！',
    color: '#6b7280',
    bg: '#f3f4f6',
  },
};

export default function GachaModal({ visible, result, onNext }: Props) {
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0)).current;
  const shakeAnim = useRef(new Animated.Value(0)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (!visible || !result) return;

    // Reset
    rotateAnim.setValue(0);
    scaleAnim.setValue(0);
    shakeAnim.setValue(0);
    opacityAnim.setValue(0);

    // Phase 1: spin + grow
    Animated.sequence([
      Animated.parallel([
        Animated.timing(rotateAnim, {
          toValue: 3,
          duration: 700,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.spring(scaleAnim, {
          toValue: 1,
          friction: 4,
          tension: 80,
          useNativeDriver: true,
        }),
      ]),
      // Phase 2: shake for jackpot / bounce for hit
      result === 'jackpot'
        ? Animated.sequence([
            Animated.timing(shakeAnim, { toValue: 10, duration: 60, useNativeDriver: true }),
            Animated.timing(shakeAnim, { toValue: -10, duration: 60, useNativeDriver: true }),
            Animated.timing(shakeAnim, { toValue: 8, duration: 60, useNativeDriver: true }),
            Animated.timing(shakeAnim, { toValue: -8, duration: 60, useNativeDriver: true }),
            Animated.timing(shakeAnim, { toValue: 0, duration: 60, useNativeDriver: true }),
          ])
        : Animated.spring(shakeAnim, { toValue: 0, useNativeDriver: true }),
      // Phase 3: fade in message
      Animated.timing(opacityAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();
  }, [visible, result]);

  if (!result) return null;

  const cfg = RESULT_CONFIG[result];
  const rotate = rotateAnim.interpolate({
    inputRange: [0, 3],
    outputRange: ['0deg', '1080deg'],
  });

  return (
    <Modal visible={visible} transparent animationType="fade">
      <View style={styles.overlay}>
        <View style={[styles.card, { backgroundColor: cfg.bg }]}>
          {/* Spinning emoji */}
          <Animated.Text
            style={[
              styles.emoji,
              {
                transform: [
                  { rotate },
                  { scale: scaleAnim },
                  { translateX: shakeAnim },
                ],
              },
            ]}
          >
            {cfg.emoji}
          </Animated.Text>

          {/* Result label */}
          <Animated.View style={{ opacity: opacityAnim }}>
            <Text style={[styles.label, { color: cfg.color }]}>{cfg.label}</Text>
            <Text style={styles.message}>{cfg.message}</Text>
          </Animated.View>

          {/* Next button */}
          <Animated.View style={[styles.buttonWrap, { opacity: opacityAnim }]}>
            <TouchableOpacity style={styles.button} onPress={onNext} activeOpacity={0.8}>
              <Text style={styles.buttonText}>つぎへ →</Text>
            </TouchableOpacity>
          </Animated.View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.65)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    width: 280,
    borderRadius: 24,
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.25,
    shadowRadius: 16,
    elevation: 10,
  },
  emoji: {
    fontSize: 80,
    marginBottom: 20,
  },
  label: {
    fontSize: 32,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  message: {
    fontSize: 18,
    color: '#374151',
    textAlign: 'center',
    marginBottom: 28,
    fontWeight: '600',
  },
  buttonWrap: {
    width: '100%',
  },
  button: {
    backgroundColor: '#16a34a',
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: 'bold',
  },
});
