import os
import time
import random
import requests
import logging
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_collection import GeolocationDataCollector
from dotenv import load_dotenv
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed


class FlickrImageCollector:
    def __init__(
            self,
            api_key: Optional[str] = None,
            base_path: str = "dataset",
            images_per_location: int = 50,
            images_per_region: int = 5000,
            num_processes: int = 4,
            num_threads_per_process: int = 4,
            min_accuracy: float = 10.0
    ):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path)

        self.api_keys = []
        key_index = 1
        while True:
            key_name = f'FLICKR_KEY{key_index}' if key_index > 1 else 'FLICKR_KEY'
            api_key = os.getenv(key_name)
            if api_key:
                self.api_keys.append(api_key)
                key_index += 1
            else:
                break

        if not self.api_keys:
            raise ValueError("No Flickr API keys found in .env file")

        self.base_url = "https://api.flickr.com/services/rest/"
        self.images_per_location = images_per_location
        self.images_per_region = images_per_region
        self.num_processes = num_processes
        self.num_threads = num_threads_per_process
        self.min_accuracy = min_accuracy

        self.data_collector = GeolocationDataCollector(base_path)
        self.metadata_file = "flickr_metadata.csv"

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

    def _load_existing_metadata(self) -> Tuple[pd.DataFrame, Set[str], Dict[str, int]]:
        metadata_path = self.data_collector.metadata_path / self.metadata_file
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            existing_ids = set(df['image_id'].str.replace('flickr_', ''))
            region_counts = df['region'].value_counts().to_dict()
            return df, existing_ids, region_counts
        return pd.DataFrame(), set(), {region: 0 for region in self.regions}

    def get_random_coordinates(self, region: Dict[str, Dict[float, float]]) -> Tuple[float, float]:
        lat_range = region['lat']
        lon_range = region['lon']
        lat = random.uniform(lat_range[0], lat_range[1])
        lon = random.uniform(lon_range[0], lon_range[1])
        return lat, lon

    def search_images(self, lat: float, lon: float, api_key: str) -> List[Dict]:
        params = {
            'method': 'flickr.photos.search',
            'api_key': api_key,
            'format': 'json',
            'nojsoncallback': 1,
            'lat': lat,
            'lon': lon,
            'radius': 10,
            'accuracy': 16, 
            'extras': 'geo,tags,machine_tags,original_format,url_o,date_taken,license',
            'per_page': self.images_per_location
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            return [photo for photo in data.get('photos', {}).get('photo', [])
                    if float(photo.get('accuracy', 0)) >= self.min_accuracy]
        except Exception as e:
            self.logger.error(f"Error searching images: {str(e)}")
            return []

    def _extract_informative_tags(self, photo: Dict) -> List[str]:
        """Extract tags that could help in geolocation"""
        tags = photo.get('tags', '').split()

        # Filter and prioritize location-informative tags
        location_tags = [
            tag for tag in tags
            if any(keyword in tag.lower() for keyword in [
                'landscape', 'city', 'street', 'building', 'landmark',
                'mountain', 'beach', 'forest', 'river', 'architecture',
                'nature', 'urban', 'rural', 'countryside'
            ])
        ]

        return location_tags[:10]  # Limit to 10 most relevant tags

    def _process_region(self, region_name: str, region_data: Dict, target_count: int, api_key: str,
                        collected_ids: Set[str], result_queue: mp.Queue, progress_queue: mp.Queue):
        """Process images for a specific region"""
        local_collected = 0
        local_metadata = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []

            while local_collected < target_count:
                try:
                    lat, lon = self.get_random_coordinates(region_data)
                    photos = self.search_images(lat, lon, api_key)

                    for photo in photos:
                        if photo['id'] in collected_ids or local_collected >= target_count:
                            continue

                        future = executor.submit(self._process_photo, photo, region_name)
                        futures.append((future, photo['id']))
                        local_collected += 1

                    done_futures = [f for f in futures if f[0].done()]
                    for future, photo_id in done_futures:
                        try:
                            result = future.result()
                            if result:
                                local_metadata.append(result)
                                progress_queue.put(1)
                                collected_ids.add(photo_id)
                        except Exception as e:
                            self.logger.error(f"Error processing photo {photo_id}: {str(e)}")
                        futures.remove((future, photo_id))

                    if len(local_metadata) >= 50:
                        result_queue.put(local_metadata)
                        local_metadata = []

                    time.sleep(0.2)
                except Exception as e:
                    self.logger.error(f"Error in region processing loop: {str(e)}")

            for future, photo_id in futures:
                try:
                    result = future.result()
                    if result:
                        local_metadata.append(result)
                        progress_queue.put(1)
                except Exception as e:
                    self.logger.error(f"Error processing final photo {photo_id}: {str(e)}")

            if local_metadata:
                result_queue.put(local_metadata)

    def _process_photo(self, photo: Dict, region_name: str) -> Optional[Dict]:
        """Download and process a single photo"""
        try:
            # Validate coordinates
            coords = photo.get('geometry', {}).get('coordinates', [0, 0])
            photo_lon, photo_lat = coords[0], coords[1]

            # Additional processing from previous implementation
            image_filename = f"flickr_{photo['id']}.jpg"
            image_path = self.data_collector.images_path / image_filename

            # Get highest resolution image URL
            image_url = self._get_photo_url(photo)
            if not image_url:
                return None

            # Download image
            if self._download_image(image_url, str(image_path)):
                if self.data_collector.validate_image(str(image_path)):
                    processed_path = self.data_collector.process_image(str(image_path))

                    if processed_path:
                        os.remove(str(image_path))

                        return {
                            'image_id': f"flickr_{photo['id']}",
                            'filename': f"processed_{image_filename}",
                            'latitude': photo_lat,
                            'longitude': photo_lon,
                            'source': 'flickr',
                            'region': region_name,
                            'tags': ' '.join(self._extract_informative_tags(photo)),
                            'date_taken': photo.get('datetaken', ''),
                            'license': photo.get('license', '')
                        }
                    else:
                        if os.path.exists(str(image_path)):
                            os.remove(str(image_path))

            return None
        except Exception as e:
            self.logger.error(f"Error processing photo {photo.get('id', 'Unknown')}: {str(e)}")
            return None

    def _get_photo_url(self, photo: Dict) -> Optional[str]:
        """Get the highest resolution image URL"""
        size_priority = ['url_o', 'url_k', 'url_h', 'url_b', 'url_c']
        for size in size_priority:
            if size in photo:
                return photo[size]
        return None

    def _download_image(self, url: str, save_path: str) -> bool:
        """Download image with timeout and error handling"""
        try:
            response = requests.get(url, stream=True, timeout=10)
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
        """Collect images with region-based limits"""
        if target_per_region is None:
            target_per_region = self.images_per_region

        # Load existing metadata to continue or resume collection
        existing_df, collected_ids, region_counts = self._load_existing_metadata()
        new_metadata = []

        # Calculate remaining images per region
        remaining_images = {
            region: max(0, target_per_region - region_counts.get(region, 0))
            for region in self.regions
        }

        total_remaining = sum(remaining_images.values())
        if total_remaining == 0:
            print("Number of images already collected for all regions!")
            return existing_df

        # Multiprocessing setup
        result_queue = mp.Queue()
        progress_queue = mp.Queue()
        processes = []
        collected_ids = set(collected_ids)

        # Progress tracking
        progress_bar = tqdm(total=total_remaining, desc="Collecting images")

        # Prepare active regions
        active_regions = [(region, data, remaining_images[region])
                          for region, data in self.regions.items()
                          if remaining_images[region] > 0]

        try:
            # Process regions with multiprocessing
            while active_regions or processes:
                # Start new processes
                while len(processes) < self.num_processes and active_regions:
                    region_name, region_data, target_count = active_regions.pop(0)
                    api_key = self.api_keys[len(processes) % len(self.api_keys)]

                    p = mp.Process(
                        target=self._process_region,
                        args=(region_name, region_data, target_count, api_key,
                              collected_ids, result_queue, progress_queue)
                    )
                    p.start()
                    processes.append(p)

                # Collect results
                try:
                    while True:
                        try:
                            batch_results = result_queue.get_nowait()
                            new_metadata.extend(batch_results)

                            # Save metadata periodically
                            self._save_metadata(new_metadata, existing_df)

                            # Update progress
                            for _ in range(len(batch_results)):
                                progress_bar.update(1)
                        except Exception:
                            break
                except Exception as e:
                    self.logger.error(f"Error processing results: {str(e)}")

                # Clean up finished processes
                processes = [p for p in processes if p.is_alive()]
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, stopping collection...")
            for p in processes:
                p.terminate()
        finally:
            progress_bar.close()
            for p in processes:
                p.join()

            self._save_metadata(new_metadata, existing_df)

        return pd.read_csv(self.data_collector.metadata_path / self.metadata_file)

    def _save_metadata(self, new_metadata: List[Dict], existing_df: Optional[pd.DataFrame] = None) -> None:
        new_df = pd.DataFrame(new_metadata)
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        self.data_collector.save_metadata(combined_df, self.metadata_file)


def main():
    collector = FlickrImageCollector(
        images_per_region=2000,
        num_processes=4,
        num_threads_per_process=4
    )
    print("Starting Flickr image collection...")

    metadata_df = collector.collect_images()

    print("\nCollection completed!")
    print(f"Total images collected: {len(metadata_df)}")
    print("\nRegional distribution:")
    print(metadata_df['region'].value_counts())


if __name__ == '__main__':
    main()