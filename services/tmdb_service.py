import requests
import os
import json
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
import logging
import threading

logger = logging.getLogger(__name__)


class TMDBService:
    """
    Service for interacting with TheMovieDB API
    - Caches all downloaded data to filesystem
    - Reduces API calls and speeds up re-identification
    - Uses locks to prevent duplicate concurrent requests for the same data
    """

    BASE_URL = 'https://api.themoviedb.org/3'
    IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/original'
    CACHE_MAX_AGE_DAYS = 7  # Cache data for 7 days

    def __init__(self, api_key, cache_dir='/app/temp/downloads', frame_height=480):
        self.api_key = api_key
        self.session = requests.Session()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.frame_height = frame_height  # Store frame height for resizing images

        # Dictionary to track locks for each show/season combination
        self._request_locks = {}
        # Master lock to protect the locks dictionary itself
        self._locks_lock = threading.Lock()

        logger.info(f"TMDBService initialized with cache: {self.cache_dir}, frame_height: {self.frame_height}")

    def _get_lock_for_request(self, show_id, season_number):
        """
        Get or create a lock for a specific show/season request.
        This ensures only one thread downloads data for a given show/season at a time.
        """
        lock_key = f"{show_id}_{season_number}"

        with self._locks_lock:
            if lock_key not in self._request_locks:
                self._request_locks[lock_key] = threading.Lock()
            return self._request_locks[lock_key]

    def search_tv_show(self, show_name):
        """Search for a TV show by name"""
        url = f'{self.BASE_URL}/search/tv'
        params = {
            'api_key': self.api_key,
            'query': show_name,
            'language': 'en-US'
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data['results']:
                logger.info(f"Found TV show: {data['results'][0]['name']} (ID: {data['results'][0]['id']})")
                return data['results'][0]

            logger.warning(f"No results found for TV show: {show_name}")
            return None

        except Exception as e:
            logger.error(f"Error searching for TV show: {e}")
            return None

    def get_season_data_cached(self, show_id, season_number):
        """
        Get season data with filesystem caching and request deduplication.
        Returns episode metadata and image paths from cache or downloads fresh.

        If multiple threads request the same data simultaneously, only one will
        download while others wait and use the cached result.

        Args:
            show_id: TMDB show ID
            season_number: Season number

        Returns:
            tuple: (episodes_dict, image_paths_dict) where image_paths_dict contains file paths
        """
        cache_path = self._get_cache_path(show_id, season_number)

        # Quick check without lock - if cache is valid, return immediately
        if self._is_cache_valid(cache_path):
            logger.info(f"✓ Loading season data from cache: {cache_path.name}")
            return self._load_from_cache(cache_path)

        # Get the lock for this specific show/season combination
        request_lock = self._get_lock_for_request(show_id, season_number)

        # Acquire the lock - only one thread will download at a time
        with request_lock:
            # Double-check cache validity after acquiring lock
            # Another thread may have downloaded the data while we were waiting
            if self._is_cache_valid(cache_path):
                logger.info(f"✓ Loading season data from cache (downloaded by another thread): {cache_path.name}")
                return self._load_from_cache(cache_path)

            # Cache miss or expired - download fresh data
            logger.info(f"Cache miss or expired - downloading season {season_number} data from TMDB")

            # Get episode metadata
            episodes = self.get_season_episodes(show_id, season_number)
            if not episodes:
                return None, None

            # Download and cache episode images
            episode_images = self._download_and_cache_images(episodes, show_id, season_number, cache_path)

            # Save metadata to cache
            self._save_metadata_to_cache(cache_path, episodes, episode_images)

            logger.info(f"✓ Cached season data to: {cache_path.name}")
            return episodes, episode_images

    def get_season_episodes(self, show_id, season_number):
        """Get episode information for a specific season (from API)"""
        url = f'{self.BASE_URL}/tv/{show_id}/season/{season_number}'
        params = {
            'api_key': self.api_key,
            'language': 'en-US'
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            episodes = {}
            for episode in data.get('episodes', []):
                ep_num = episode['episode_number']
                episodes[ep_num] = {
                    'episode_number': ep_num,
                    'name': episode['name'],
                    'still_path': episode.get('still_path'),
                    'overview': episode.get('overview', '')
                }

            logger.info(f"Retrieved {len(episodes)} episodes for season {season_number}")
            return episodes

        except Exception as e:
            logger.error(f"Error getting season episodes: {e}")
            return None

    def get_episode_images(self, show_id, season_number, episode_number):
        """Get all available image paths for a specific episode"""
        url = f'{self.BASE_URL}/tv/{show_id}/season/{season_number}/episode/{episode_number}/images'
        params = {
            'api_key': self.api_key
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            stills = data.get('stills', [])
            logger.debug(f"Found {len(stills)} stills for S{season_number}E{episode_number}")
            return [still['file_path'] for still in stills]

        except Exception as e:
            logger.error(f"Error getting episode images: {e}")
            return []

    def _download_and_cache_images(self, episodes, show_id, season_number, cache_path):
        """
        Download all episode images and save to cache directory

        Returns:
            dict: {episode_num: {'images': [file_paths], 'name': str, 'episode_number': int}}
        """
        episode_images = {}
        images_dir = cache_path / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading images for {len(episodes)} episodes...")

        for episode_num, episode_data in episodes.items():
            image_paths = []

            # Try to get multiple images per episode
            tmdb_image_paths = self.get_episode_images(show_id, season_number, episode_num)

            # Download all available images for this episode
            for idx, img_path in enumerate(tmdb_image_paths):
                if img_path:
                    local_path = images_dir / f'ep{episode_num:02d}_img{idx+1}.jpg'

                    if self._download_image(img_path, local_path):
                        image_paths.append(str(local_path))

            # Fallback to default still_path if no additional images found
            if not image_paths and episode_data.get('still_path'):
                local_path = images_dir / f'ep{episode_num:02d}_default.jpg'

                if self._download_image(episode_data['still_path'], local_path):
                    image_paths.append(str(local_path))

            # Only add if we have at least one image
            if image_paths:
                episode_images[episode_num] = {
                    'images': image_paths,  # List of file paths
                    'name': episode_data['name'],
                    'episode_number': episode_num
                }
                logger.info(f"Episode {episode_num}: {len(image_paths)} image(s) cached")

        logger.info(f"✓ Cached images for {len(episode_images)} episodes")
        return episode_images

    def _download_image(self, tmdb_path, local_path):
        """
        Download a single image from TMDB and save to local path
        Resizes image to match frame_height while maintaining aspect ratio

        Args:
            tmdb_path: TMDB image path (e.g., /abc123.jpg)
            local_path: Local filesystem path to save to

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            image_url = f"{self.IMAGE_BASE_URL}{tmdb_path}"
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()

            # Load image from response
            image = Image.open(BytesIO(response.content))

            # Resize to match frame height while maintaining aspect ratio
            # This ensures TMDB images match the scale of extracted video frames
            original_width, original_height = image.size
            if original_height != self.frame_height:
                # Calculate new width maintaining aspect ratio
                aspect_ratio = original_width / original_height
                new_width = int(self.frame_height * aspect_ratio)
                new_size = (new_width, self.frame_height)

                # Use LANCZOS for high-quality downsampling
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {original_width}x{original_height} to {new_width}x{self.frame_height}")

            # Save resized image to disk
            image.save(local_path, 'JPEG', quality=95)

            logger.debug(f"Downloaded and saved: {local_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error downloading image {tmdb_path}: {e}")
            return False

    def _get_cache_path(self, show_id, season_number):
        """Get cache directory path for a specific show/season"""
        cache_name = f"{show_id}_s{season_number:02d}"
        return self.cache_dir / cache_name

    def _is_cache_valid(self, cache_path):
        """
        Check if cache exists and is not expired

        Args:
            cache_path: Path to cache directory

        Returns:
            bool: True if cache is valid and not expired
        """
        metadata_file = cache_path / 'metadata.json'

        if not metadata_file.exists():
            return False

        # Check age
        try:
            file_age_days = (time.time() - metadata_file.stat().st_mtime) / (24 * 3600)
            if file_age_days > self.CACHE_MAX_AGE_DAYS:
                logger.info(f"Cache expired ({file_age_days:.1f} days old, max {self.CACHE_MAX_AGE_DAYS} days)")
                return False

            logger.debug(f"Cache valid ({file_age_days:.1f} days old)")
            return True

        except Exception as e:
            logger.warning(f"Error checking cache age: {e}")
            return False

    def _save_metadata_to_cache(self, cache_path, episodes, episode_images):
        """Save episode metadata to cache"""
        metadata_file = cache_path / 'metadata.json'

        try:
            metadata = {
                'episodes': episodes,
                'episode_images': episode_images,
                'cached_at': time.time()
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved metadata to {metadata_file}")

        except Exception as e:
            logger.error(f"Error saving metadata to cache: {e}")

    def _load_from_cache(self, cache_path):
        """
        Load episode data from cache

        Returns:
            tuple: (episodes_dict, episode_images_dict)
        """
        metadata_file = cache_path / 'metadata.json'

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Convert episode keys back to integers
            episodes = {int(k): v for k, v in metadata['episodes'].items()}
            episode_images = {int(k): v for k, v in metadata['episode_images'].items()}

            # Verify image files still exist
            for ep_num, ep_data in list(episode_images.items()):
                valid_images = [img for img in ep_data['images'] if Path(img).exists()]
                if not valid_images:
                    logger.warning(f"Episode {ep_num}: No valid cached images found, will re-download")
                    del episode_images[ep_num]
                else:
                    episode_images[ep_num]['images'] = valid_images

            logger.info(f"Loaded {len(episodes)} episodes from cache ({len(episode_images)} with images)")
            return episodes, episode_images

        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None, None

    def clear_cache(self, show_id=None, season_number=None):
        """
        Clear cache for specific show/season or all cache

        Args:
            show_id: Optional show ID to clear
            season_number: Optional season number to clear
        """
        import shutil

        if show_id and season_number:
            cache_path = self._get_cache_path(show_id, season_number)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for show {show_id} season {season_number}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all TMDB cache")
