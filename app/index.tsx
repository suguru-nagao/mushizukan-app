import { useEffect } from 'react';
import { useRouter } from 'expo-router';
import { View, ActivityIndicator } from 'react-native';
import { getSettings } from '../lib/storage';

export default function EntryScreen() {
  const router = useRouter();

  useEffect(() => {
    (async () => {
      const settings = await getSettings();
      if (settings?.onboardingDone) {
        router.replace('/(tabs)/calendar');
      } else {
        router.replace('/onboarding');
      }
    })();
  }, []);

  return (
    <View style={{ flex: 1, backgroundColor: '#f0fdf4', justifyContent: 'center', alignItems: 'center' }}>
      <ActivityIndicator size="large" color="#16a34a" />
    </View>
  );
}
