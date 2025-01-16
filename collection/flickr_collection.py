import logging
import os
import time
import random
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List, Dict, Set
from pathlib import Path
import requests
from data_collection import GeolocationDataCollector

class FlickrImageCollector:
    def __init__(self, api_key: Optional[str] = None, base_path: str = "dataset", images_per_location: int = 10, min_images_per_type: int = 100, images_per_region: int = 150):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv('FLICKR_KEY')

        self.base_url = "https://www.flickr.com/services/rest/"
        self.images_per_location = images_per_location
        self.min_images_per_type = min_images_per_type
        self.images_per_region = images_per_region
        self.data_collector = GeolocationDataCollector()

        self.location_types = {
            'urban': ['cityscape', 'street', 'building', 'architecture', 'city'],
            'nature': ['landscape', 'wilderness', 'nature', 'outdoor'],
            'beach': ['beach', 'coast', 'ocean', 'sea', 'shore'],
            'mountain': ['mountain', 'peak', 'alps', 'hiking', 'summit'],
            'forest': ['forest', 'woods', 'trees', 'woodland'],
            'park': ['park', 'garden', 'nationalpark'],
            'historic': ['historic', 'monument', 'ruins', 'castle', 'temple'],
            'rural': ['rural', 'countryside', 'village', 'farm', 'field'],
            'waterfront': ['lake', 'river', 'waterfront', 'harbor', 'port']
        }

        self.regions = {
            'north_america': {'lat': (25, 70), 'lon': (-170, -50)},
            'south_america': {'lat': (-60, 15), 'lon': (-80, -35)},
            'europe': {'lat': (35, 70), 'lon': (-10, 40)},
            'africa': {'lat': (-35, 35), 'lon': (-20, 50)},
            'asia': {'lat': (0, 70), 'lon': (60, 180)},
            'oceania': {'lat': (-50, 0), 'lon': (110, 180)}
        }

        self._setup_logging(base_path)

    def _setup_logging(self, base_path: str) -> None:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=Path(base_path) / 'flickr_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_random_coordinates(self, region: Dict[str, Dict[float, float]]) -> tuple:
        lat_range = region['lat']
        lon_range = region['lon']

        lat = random.uniform(lat_range[0], lat_range[1])
        lon = random.uniform(lon_range[0], lon_range[1])

        return lat, lon

    def search_photos(self, lat: float, lon: float, page: int = 1, radius: int = 10) -> List[Dict]:
        params = {
            'method': 'flickr.photos.search',
            'api_key': self.api_key,
            'tags': ','.join(random.choice(list(self.location_types.values()))),
            'tag_mode': 'any',
            'lat': lat,
            'lon': lon,
            'radius': radius,
            'has_geo': 1,
            'extras': 'url_l,geo,tags',
            'format': 'json',
            'nojsoncallback': 1,
            'per_page': self.images_per_location,
            'page': page,
            'accuracy': 16,
            'content_type': 1,
            'sort': 'interestingness-desc'
        }



        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'photos' in data and 'photo' in data['photos']:
                return [photo for photo in data['photos']['photo']
                        if 'url_l' in photo and 'latitude' in photo and 'longitude' in photo]
            return []
        except Exception as e:
            self.logger.error(f"Error searching photos: {str(e)}")
            return []

    def download_image(self, url: str, save_path: str) -> bool:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                response.raw.decode_content = True
                f.write(response.content)

            return True

        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {str(e)}")
            return False

    def collect_images(self) -> pd.DataFrame:
        """Collect images using the LocationSampler"""
        metadata = []
        collected_ids: Set[str] = set()

        region_counts = {region: 0 for region in self.regions}

        progress_bar = tqdm(total=len(self.regions) * self.images_per_region, desc="Collecting images")

        while min(region_counts.values()) < self.images_per_region:
            region_name = min(region_counts.items(), key=lambda x: x[1])[0]
            region_data = self.regions[region_name]

            lat, lon = self.get_random_coordinates(region_data)

            for page in range(1, 4):
                if region_counts[region_name] >= self.images_per_region:
                    break

                photos = self.search_photos(lat, lon, page=page)

                for photo in photos:
                    if (photo['id'] in collected_ids or
                            region_counts[region_name] >= self.images_per_region):
                        continue

                    try:
                        photo_lat = float(photo['latitude'])
                        photo_lon = float(photo['longitude'])
                        bounds = region_data

                        if not (bounds['lat'][0] <= photo_lat <= bounds['lat'][1] and
                                bounds['lon'][0] <= photo_lon <= bounds['lon'][1]):
                            continue

                        image_filename = f"flickr_{photo['id']}.jpg"
                        image_path = self.data_collector.images_path / image_filename

                        if self.download_image(photo['url_l'], str(image_path)):
                            if self.data_collector.validate_image(str(image_path)):
                                processed_path = self.data_collector.process_image(str(image_path))
                                if processed_path:
                                    metadata.append({
                                        'image_id': f"flickr_{photo['id']}",
                                        'filename': image_filename,
                                        'latitude': photo_lat,
                                        'longitude': photo_lon,
                                        'source': 'flickr',
                                        'region': region_name,
                                        'tags': photo.get('tags', '')
                                    })
                                    collected_ids.add(photo['id'])
                                    region_counts[region_name] += 1
                                    progress_bar.update(1)
                                else:
                                    os.remove(str(image_path))
                            else:
                                os.remove(str(image_path))

                        time.sleep(0.5)

                    except Exception as e:
                        self.logger.error(f"Error processing photo {photo['id']}: {str(e)}")
                        continue

            time.sleep(0.5)

        progress_bar.close()

        metadata_df = pd.DataFrame(metadata)
        self._log_statistics(metadata_df)
        self.data_collector.save_metadata(metadata_df, "flickr_metadata.csv")
        return metadata_df

    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Log collection statistics"""
        stats = {
            "Total Images": len(df),
            "Images per Region": df['region'].value_counts().to_dict(),
            "Average Latitude per Region": df.groupby('region')['latitude'].mean().to_dict(),
            "Average Longitude per Region": df.groupby('region')['longitude'].mean().to_dict(),
            "Coordinate Spread per Region": df.groupby('region').agg({
                'latitude': ['std', 'min', 'max'],
                'longitude': ['std', 'min', 'max']
            }).to_dict()
        }

        self.logger.info("Collection statistics:")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")

def main():
    collector = FlickrImageCollector(images_per_region=150)
    print("Starting balanced image collection...")

    metadata_df = collector.collect_images()

    print("\nCollection completed!")
    print(f"Total images collected: {len(metadata_df)}")
    print("\nRegional distribution:")
    print(metadata_df['region'].value_counts())

    print("\nCoordinate spread by region:")
    spread_stats = metadata_df.groupby('region').agg({
        'latitude': ['mean', 'std'],
        'longitude': ['mean', 'std']
    })
    print(spread_stats)

if __name__ == '__main__':
    main()