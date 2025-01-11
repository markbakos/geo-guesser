import logging
import os
import time

from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List, Dict
from pathlib import Path
import requests
from data_collection import GeolocationDataCollector
from location_sampler import LocationSampler

class FlickrImageCollector:
    def __init__(self, api_key: Optional[str] = None, base_path: str = "dataset", images_per_location: int = 10):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv('FLICKR_KEY')

        self.base_url = "https://www.flickr.com/services/rest/"
        self.images_per_location = images_per_location
        self.data_collector = GeolocationDataCollector()

        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
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

    def collect_images(self, min_locations: int = 200) -> pd.DataFrame:
        """Collect images using the LocationSampler"""
        try:
            sampler = LocationSampler(min_locations=min_locations)
            locations = sampler.generate_locations()

            self.logger.info(f"Generated {len(locations)} diverse locations")

            pd.DataFrame(locations).to_csv(
                Path(self.data_collector.base_path) / "flickr_sampling_locations.csv",
                index=False
            )

            metadata = []
            for location in tqdm(locations, desc="Processing locations"):
                photos = self.search_photos(location['lat'], location['lon'])

                for photo in tqdm(photos, desc=f"Collecting images for {location.get('name', 'location')}",
                                  leave=False):
                    try:
                        image_filename = f"flickr_{photo['id']}.jpg"
                        image_path = self.data_collector.images_path / image_filename

                        if self.download_image(photo['url_l'], str(image_path)):
                            if self.data_collector.validate_image(str(image_path)):
                                processed_path = self.data_collector.process_image(str(image_path))
                                if processed_path:
                                    metadata.append({
                                        'image_id': f"flickr_{photo['id']}",
                                        'filename': image_filename,
                                        'latitude': float(photo['latitude']),
                                        'longitude': float(photo['longitude']),
                                        'source': 'flickr',
                                        'location_name': location.get('name', 'unknown'),
                                        'location_type': location.get('type', 'unknown')
                                    })
                                else:
                                    os.remove(str(image_path))
                            else:
                                os.remove(str(image_path))

                        time.sleep(0.5)

                    except Exception as e:
                        self.logger.error(f"Error processing photo {photo['id']}: {str(e)}")
                        continue

            metadata_df = pd.DataFrame(metadata)

            self.data_collector.save_metadata(metadata_df, "flickr_metadata.csv")

            self._log_statistics(metadata_df)

            return metadata_df

        except Exception as e:
            self.logger.error(f"Error in image collection: {str(e)}")
            raise

    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Log collection statistics"""
        stats = {
            "Total Images": len(df),
            "Unique Locations": len(df.groupby(['latitude', 'longitude'])),
            "Location Types": df['location_type'].value_counts().to_dict()
        }

        self.logger.info("Collection statistics:")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")

def main():
    collector = FlickrImageCollector()

    metadata_df = collector.collect_images(min_locations=250)

    collector.data_collector.save_metadata(metadata_df)

    print("\nCollection completed!")
    print(f"Total images collected: {len(metadata_df)}")
    print(f"Unique locations: {len(metadata_df.groupby(['latitude', 'longitude']))}")
    print("\nLocation type distribution:")
    print(metadata_df['location_type'].value_counts())

if __name__ == '__main__':
    main()