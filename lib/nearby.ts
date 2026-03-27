import * as Location from 'expo-location';

const INAT_TOKEN = process.env.EXPO_PUBLIC_INAT_TOKEN;

export type NearbyInsect = {
  id: number;
  name: string;
  photoUrl: string | null;
  observedOn: string;
};

export async function getNearbyInsects(): Promise<NearbyInsect[]> {
  try {
    const { status } = await Location.getForegroundPermissionsAsync();
    if (status !== 'granted') {
      return getFallbackInsects();
    }

    const loc = await Location.getCurrentPositionAsync({
      accuracy: Location.Accuracy.Balanced,
    });

    const { latitude, longitude } = loc.coords;

    const url = `https://api.inaturalist.org/v1/observations?taxon_name=Insecta&lat=${latitude}&lng=${longitude}&radius=10&per_page=5&order_by=observed_on&locale=ja`;

    const res = await fetch(url, {
      headers: INAT_TOKEN ? { Authorization: `Bearer ${INAT_TOKEN}` } : {},
    });

    if (!res.ok) return getFallbackInsects();

    const data = await res.json();
    const results: any[] = data.results ?? [];

    return results.map((obs: any) => {
      const taxon = obs.taxon;
      const name =
        taxon?.preferred_common_name ?? taxon?.name ?? '不明な虫';
      const photo = obs.photos?.[0];
      const photoUrl = photo
        ? photo.url.replace('square', 'medium')
        : null;
      const observedOn: string = obs.observed_on ?? obs.time_observed_at?.slice(0, 10) ?? '';
      return { id: obs.id, name, photoUrl, observedOn };
    });
  } catch {
    return getFallbackInsects();
  }
}

function getFallbackInsects(): NearbyInsect[] {
  return [
    { id: 1, name: 'カブトムシ', photoUrl: null, observedOn: '' },
    { id: 2, name: 'アゲハチョウ', photoUrl: null, observedOn: '' },
    { id: 3, name: 'カナブン', photoUrl: null, observedOn: '' },
  ];
}
