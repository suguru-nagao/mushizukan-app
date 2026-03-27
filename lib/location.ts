import * as Location from 'expo-location';

export type CaptureLocation = {
  label: string;
  latitude?: number;
  longitude?: number;
};

export async function getCaptureLocation(): Promise<CaptureLocation> {
  try {
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== 'granted') return { label: '不明な場所' };

    const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
    const { latitude, longitude } = loc.coords;
    const [address] = await Location.reverseGeocodeAsync({ latitude, longitude });

    // iOS: region=都道府県, subregion=市区町村, city=市区町村(別フィールド), district=丁目/地区
    const region = address.region ?? '';
    const city = address.city ?? address.subregion ?? '';
    const label = (region + city) || address.name || address.street || `${latitude.toFixed(4)},${longitude.toFixed(4)}`;
    return { label, latitude, longitude };
  } catch {
    return { label: '不明な場所' };
  }
}

export function getSeason(date: Date = new Date()): string {
  const month = date.getMonth() + 1; // 1〜12
  if (month >= 3 && month <= 5) return '春';
  if (month >= 6 && month <= 8) return '夏';
  if (month >= 9 && month <= 11) return '秋';
  return '冬';
}
