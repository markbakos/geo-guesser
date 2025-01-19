import os
import time
import random
import requests
import logging
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from data_collection import GeolocationDataCollector
from dotenv import load_dotenv


class MapillaryImageCollector:
    def __init__(self, api_key: Optional[str] = None, base_path: str = "dataset", images_per_location: int = 10, images_per_region: int = 1000, min_image_quality: float = 0.7):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv('MAPILLARY_KEY')

        self.base_url = "https://graph.mapillary.com/images"
        self.images_per_location = images_per_location
        self.images_per_region = images_per_region
        self.min_image_quality = min_image_quality
        self.data_collector = GeolocationDataCollector(base_path)
        self.metadata_file = "mapillary_metadata.csv"

        self.regions = {
            'north_america': {'lat': (25, 70), 'lon': (-170, -50)},
            'south_america': {'lat': (-60, 15), 'lon': (-80, -35)},
            'europe': {'lat': (35, 70), 'lon': (-10, 40)},
            'africa': {'lat': (-35, 35), 'lon': (-20, 50)},
            'asia': {'lat': (0, 70), 'lon': (60, 180)},
            'oceania': {'lat': (-50, 0), 'lon': (110, 180)}
        }

        self._setup_logging(base_path)

    def _load_existing_metadata(self) -> Tuple[pd.DataFrame, Set[str], Dict[str, int]]:
        metadata_path = self.data_collector.metadata_path / self.metadata_file
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            existing_ids = set(df['image_id'].str.replace('mapillary_', ''))
            region_counts = df['region'].value_counts().to_dict()
            return df, existing_ids, region_counts
        return pd.DataFrame(), set(), {region: 0 for region in self.regions}

    def _save_metadata(self, new_metadata: List[Dict], existing_df: pd.DataFrame = None) -> None:
        new_df = pd.DataFrame(new_metadata)
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        self.data_collector.save_metadata(combined_df, self.metadata_file)
        self._log_statistics(combined_df)

    def _setup_logging(self, base_path: str) -> None:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=Path(base_path) / 'mapillary_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _log_statistics(self, df: pd.DataFrame) -> None:
        try:
            stats = {
                "Total Images": len(df),
                "Images per Region": df['region'].value_counts().to_dict(),
                "Average Quality Score": df['quality_score'].mean(),
                "Quality Score Distribution": df['quality_score'].describe().to_dict(),
                "Coordinate Spread per Region": df.groupby('region').agg({
                    'latitude': ['std', 'min', 'max'],
                    'longitude': ['std', 'min', 'max']
                }).to_dict()
            }

            if 'captured_at' in df.columns and not df['captured_at'].isna().all():
                stats["Images by Year"] = pd.to_datetime(df['captured_at']).dt.year.value_counts().to_dict()

            self.logger.info("Collection statistics:")
            for key, value in stats.items():
                self.logger.info(f"{key}: {value}")

            print("\nCollection Summary:")
            print(f"Total Images: {stats['Total Images']}")
            print("\nImages per Region:")
            for region, count in stats['Images per Region'].items():
                print(f"{region}: {count}")
            print(f"\nAverage Quality Score: {stats['Average Quality Score']:.2f}")

        except Exception as e:
            self.logger.error(f"Error generating statistics: {str(e)}")
            print(f"Warning: Error generating statistics: {str(e)}")

    def get_random_coordinates(self, region: Dict[str, Dict[float, float]]) -> Tuple[float, float]:
        lat_range = region['lat']
        lon_range = region['lon']
        lat = random.uniform(lat_range[0], lat_range[1])
        lon = random.uniform(lon_range[0], lon_range[1])
        return lat, lon

    def search_images(self, lat: float, lon: float, radius: int = 1000) -> List[Dict]:
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }

        params = {
            'fields': 'id,geometry,thumb_2048_url,captured_at,quality_score',
            'bbox': f'{lon - 0.1},{lat - 0.1},{lon + 0.1},{lat + 0.1}',
            'limit': self.images_per_location
        }

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            return [img for img in data.get('data', [])
                    if img.get('quality_score', 0) >= self.min_image_quality]
        except Exception as e:
            self.logger.error(f"Error searching images: {str(e)}")
            return []

    def download_image(self, url: str, save_path: str) -> bool:
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {str(e)}")
            return False

    def collect_images(self, target_per_region: Optional[int] = None) -> pd.DataFrame:
        if target_per_region is None:
            target_per_region = self.images_per_region

        existing_df, collected_ids, region_counts = self._load_existing_metadata()
        new_metadata = []

        remaining_images = {
            region: max(0, target_per_region - region_counts.get(region, 0))
            for region in self.regions
        }

        total_remaining = sum(remaining_images.values())
        if total_remaining == 0:
            print("Mumber of images already collected for all regions!")
            return existing_df

        progress_bar = tqdm(total=total_remaining, desc="Collecting images")

        while any(count > 0 for count in remaining_images.values()):
            region_name = max(remaining_images.items(), key=lambda x: x[1])[0]
            region_data = self.regions[region_name]

            if remaining_images[region_name] <= 0:
                continue

            lat, lon = self.get_random_coordinates(region_data)
            photos = self.search_images(lat, lon)

            for photo in photos:
                if photo['id'] in collected_ids or remaining_images[region_name] <= 0:
                    continue

                try:
                    coords = photo['geometry']['coordinates']
                    photo_lon, photo_lat = coords[0], coords[1]
                    bounds = region_data

                    if not (bounds['lat'][0] <= photo_lat <= bounds['lat'][1] and
                            bounds['lon'][0] <= photo_lon <= bounds['lon'][1]):
                        continue

                    image_filename = f"mapillary_{photo['id']}.jpg"
                    image_path = self.data_collector.images_path / image_filename

                    if self.download_image(photo['thumb_2048_url'], str(image_path)):
                        if self.data_collector.validate_image(str(image_path)):
                            processed_path = self.data_collector.process_image(str(image_path))

                            if processed_path:
                                os.remove(str(image_path))

                                new_metadata.append({
                                    'image_id': f"mapillary_{photo['id']}",
                                    'filename': f"processed_{image_filename}",
                                    'latitude': photo_lat,
                                    'longitude': photo_lon,
                                    'source': 'mapillary',
                                    'region': region_name,
                                    'quality_score': photo.get('quality_score', 0),
                                    'captured_at': photo.get('captured_at', '')
                                })

                                collected_ids.add(photo['id'])
                                remaining_images[region_name] -= 1
                                progress_bar.update(1)

                                if len(new_metadata) % 100 == 0:
                                    self._save_metadata(new_metadata, existing_df)
                            else:
                                if os.path.exists(str(image_path)):
                                    os.remove(str(image_path))

                    time.sleep(0.2)

                except Exception as e:
                    self.logger.error(f"Error processing photo {photo['id']}: {str(e)}")
                    if os.path.exists(str(image_path)):
                        os.remove(str(image_path))
                    continue

            time.sleep(0.5)

        progress_bar.close()

        self._save_metadata(new_metadata, existing_df)

        return pd.read_csv(self.data_collector.metadata_path / self.metadata_file)


def main():
    collector = MapillaryImageCollector(images_per_region=500)
    print("Starting image collection...")

    metadata_df = collector.collect_images(target_per_region=500)

    print("\nCollection completed!")
    print(f"Total images collected: {len(metadata_df)}")
    print("\nRegional distribution:")
    print(metadata_df['region'].value_counts())

    print("\nQuality score statistics:")
    print(metadata_df['quality_score'].describe())


if __name__ == '__main__':
    main()