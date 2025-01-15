import random
from typing import List, Dict
import pandas as pd
import geopandas as gpd
import requests
from time import sleep


class LocationSampler:
    def __init__(self, min_locations: int = 1000, population_threshold: int = 10000):
        self.min_locations = min_locations
        self.population_threshold = population_threshold

        self.lat_bounds = (-60, 75)
        self.lon_bounds = (-180, 180)

        self.world_cities = self._load_world_cities()
        self.valid_locations_cache = set()

        self.location_types = {
            'urban': ['cityscape', 'street', 'architecture', 'building', 'city'],
            'nature': ['landscape', 'nature', 'wilderness'],
            'beach': ['beach', 'coast', 'ocean', 'sea'],
            'mountain': ['mountain', 'hills', 'alps', 'peak'],
            'forest': ['forest', 'woods', 'trees'],
            'park': ['park', 'garden', 'nationalpark'],
            'historic': ['historic', 'monument', 'ruins', 'castle'],
            'rural': ['rural', 'countryside', 'village', 'farm'],
            'waterfront': ['lake', 'river', 'waterfront', 'harbor']
        }

    def _load_world_cities(self) -> pd.DataFrame:
        """Load world cities database"""
        try:
            ne_file = "ne_10m_populated_places_simple.geojson"
            populated_places = gpd.read_file(ne_file)
            return populated_places[populated_places['pop_min'] > self.population_threshold]
        except Exception as e:
            print(f"Error loading GeoJSON file: {e}")
            return pd.DataFrame()

    def _get_location_type(self, lat:float, lon: float) -> str:
        try:
            url= f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
            headers={'User-Agent': 'LocationSampler/1.0'}
            response = requests.get(url, headers=headers, timeout=5)
            data= response.json()

            if 'address' not in data:
                return 'unknown'

            address = data.get('address', {})

            if any(key in address for key in ['bay', 'sea', 'ocean']):
                return 'beach'

            if any(key in address for key in ['peak', 'ridge', 'mountain']):
                return 'mountain'

            if any(key in address for key in ['national_park', 'park']):
                return 'park'

            if any(key in address for key in ['forest', 'wood']):
                return 'forest'

            if any(key in address for key in ['castle', 'monument', 'ruins']):
                return 'historic'

            if any(key in address for key in ['city', 'town']):
                return 'urban'

            if any(key in address for key in ['village', 'hamlet', 'farm']):
                return 'rural'

            if any(key in address for key in ['lake', 'river']):
                return 'waterfront'

            if 'suburb' not in address and 'city' not in address:
                return 'nature'

            return 'urban'

        except Exception:
            return 'unknown'

    def generate_locations(self) -> List[Dict[str, float]]:
        locations = []

        try:
            city_samples = self.world_cities.sample(
                n=min(len(self.world_cities), self.min_locations // 4),
                weights='pop_min',
                replace=False
            )

            for _, city in city_samples.iterrows():
                location_type = self._get_location_type(city.geometry.y, city.geometry.x)
                locations.append({
                    'lat': float(city.geometry.y),
                    'lon': float(city.geometry.x),
                    'type': location_type,
                    'name': city['name']
                })
        except Exception as e:
            print(f"Warning: Error sampling cities: {e}")

        remaining_locations = self.min_locations - len(locations)
        attempts = 0
        max_attempts = remaining_locations * 3

        batch_size = 10
        while len(locations) < self.min_locations and attempts < max_attempts:
            new_locations = []
            for _ in range(min(batch_size, self.min_locations - len(locations))):
                lat = random.uniform(*self.lat_bounds)
                lon = random.uniform(*self.lon_bounds)

                location_type = self._get_location_type(lat, lon)
                new_locations.append({
                    'lat': lat,
                    'lon': lon,
                    'type': location_type,
                    'name': f'location_{len(locations) + len(new_locations)}'
                })

            valid_locations = self._validate_locations(new_locations)
            locations.extend(valid_locations)
            attempts += batch_size

            sleep(0.5)

            if attempts % 50 == 0:
                print(f"Progress: {len(locations)}/{self.min_locations} locations found")

        return self._check_diversity(locations)

    def _validate_locations(self, locations: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Validate multiple locations at once"""
        valid_locations = []

        for loc in locations:
            loc_key = (loc['lat'], loc['lon'])

            if loc_key in self.valid_locations_cache:
                valid_locations.append(loc)
                continue

            if self._is_valid_location(loc['lat'], loc['lon']):
                self.valid_locations_cache.add(loc_key)
                valid_locations.append(loc)

        return valid_locations

    def _is_valid_location(self, lat: float, lon: float) -> bool:
        """Check if location is on land"""
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
            headers = {'User-Agent': 'LocationSampler/1.0'}
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            return 'address' in data
        except Exception:
            return False

    def _check_diversity(self, locations: List[Dict[str, float]]) -> List[Dict[str, float]]:
        df = pd.DataFrame(locations)
        if df.empty:
            return locations

        continents = self._assign_continents(df)
        balanced_locations = []

        for continent in continents.unique():
            continent_locations = df[continents == continent]

            for location_type in df['type'].unique():
                type_locations = continent_locations[continent_locations['type'] == location_type]

                if not type_locations.empty:
                    sample_size = max(
                        int(self.min_locations / (len(continents.unique()) * len(df['type'].unique()))),
                        min(len(continent_locations), 20)
                    )

                    balanced_locations.extend(
                        type_locations.sample(
                            n=min(len(type_locations), sample_size),
                            replace=False
                        ).to_dict('records')
                    )

        return balanced_locations

    def _assign_continents(self, df: pd.DataFrame) -> pd.Series:
        def get_continent(row):
            lat, lon = row['lat'], row['lon']
            if lon >= -170 and lon <= -30:
                return 'Americas'
            elif lon > -30 and lon <= 60:
                return 'Europe_Africa'
            else:
                return 'Asia_Pacific'

        return df.apply(get_continent, axis=1)


def main():
    sampler = LocationSampler(min_locations=100)
    print("Starting location sampling...")

    try:
        locations = sampler.generate_locations()
        df = pd.DataFrame(locations)
        df.to_csv('dataset/sampling_locations.csv', index=False)

        print(f"\nTotal locations generated: {len(locations)}")
        print("\nDistribution by type:")
        print(df['type'].value_counts())
        print("\nDistribution by continent:")
        print(sampler._assign_continents(df).value_counts())

    except Exception as e:
        print(f"Error during sampling: {e}")


if __name__ == '__main__':
    main()