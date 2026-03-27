import { useEffect, useState } from 'react';
import { Stack } from 'expo-router';
import { getSettings } from '../lib/storage';

export default function RootLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" />
      <Stack.Screen name="onboarding" />
      <Stack.Screen name="(tabs)" />
    </Stack>
  );
}
