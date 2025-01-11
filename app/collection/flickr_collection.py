import logging
import os
import time
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List, Dict
from pathlib import Path
import requests
import sys
from data_collection import GeolocationDataCollector

class FlickrImageCollector:
    def __init__(self, api_key: Optional[str] = None, base_path: str = "dataset", images_per_location: int = 100):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv('FLICKR_KEY')

        self.base_url = "https://www.flickr.com/services/rest/"
        self.images_per_location = images_per_location
        self.data_collector = GeolocationDataCollector()

        logging.basicConfig(
            filename=Path(base_path) / 'flickr_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.logger = logging.getLogger(__name__)

    def search_photos(self, lat: float, lon: float, radius: int = 5) -> List[Dict]:
        params = {
            'method': 'flickr.photos.search',
            'api_key': self.api_key,
            'lat': lat,
            'lon': lon,
            'radius': radius,
            'has_geo': 1,
            'extras': 'url_l,geo',
            'format': 'json',
            'nojsoncallback': 1,
            'per_page': self.images_per_location,
            'accuracy': 16,
            'content_type': 1,
            'sort': 'relevance'
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'photos' in data and 'photo' in data['photos']:
                return [photo for photo in data['photos']['photo'] if 'url_l' in photo and 'latitude' in photo and 'longitude' in photo]
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

    def collect_images(self, locations: List[Dict[str, float]]) -> pd.DataFrame:
        """Collect images for a list of locations"""
        metadata = []

        for location in tqdm(locations, desc="Collecting locations"):
            photos = self.search_photos(location['lat'], location['lon'])

            for photo in tqdm(photos, desc=f"Processing images for {location['lat']}, {location['lon']}", leave=False):
                try:
                    image_filename = f"{photo['id']}.jpg"
                    image_path = self.data_collector.images_path / image_filename

                    if self.download_image(photo['url_l'], str(image_path)):
                        if self.data_collector.validate_image(str(image_path)):
                            processed_path = self.data_collector.process_image(str(image_path))
                            if processed_path:
                                metadata.append({
                                    'image_id': photo['id'],
                                    'filename': image_filename,
                                    'latitude': float(photo['latitude']),
                                    'longitude': float(photo['longitude']),
                                    'original_url': photo['url_l']
                                })
                            else:
                                os.remove(str(image_path))
                        else:
                            os.remove(str(image_path))

                    time.sleep(0.5)

                except Exception as e:
                    self.logger.error(f"Error processing photo {photo['id']}: {str(e)}")
                    continue

        return pd.DataFrame(metadata)

def main():
    locations = [
        {"lat": 40.7128, "lon": -74.0060},
        {"lat": 51.5074, "lon": -0.1278},
        {"lat": 35.6762, "lon": 139.6503},
        {"lat": 48.8566, "lon": 2.3522},
        {"lat": -33.8688, "lon": 151.2093},
    ]

    collector = FlickrImageCollector()
    metadata_df = collector.collect_images(locations)

    collector.data_collector.save_metadata(metadata_df)

    splits = collector.data_collector.create_dataset_split()

    print(f"Total images collected: {len(metadata_df)}")
    print(f"Training samples: {len(splits['train'])}")
    print(f"Validation samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")

if __name__ == '__main__':
    main()