import requests
import os
from pathlib import Path
from PIL import Image
from io import BytesIO


class TMDBService:
    """Service for interacting with TheMovieDB API"""

    BASE_URL = 'https://api.themoviedb.org/3'
    IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/original'

    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()

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
                # Return the first (most relevant) result
                return data['results'][0]
            return None

        except Exception as e:
            print(f"Error searching for TV show: {e}")
            return None

    def get_season_episodes(self, show_id, season_number):
        """Get episode information for a specific season"""
        url = f'{self.BASE_URL}/tv/{show_id}/season/{season_number}'
        params = {
            'api_key': self.api_key,
            'language': 'en-US'
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            episodes = []
            for episode in data.get('episodes', []):
                episodes.append({
                    'episode_number': episode['episode_number'],
                    'name': episode['name'],
                    'still_path': episode.get('still_path'),
                    'overview': episode.get('overview', '')
                })

            return episodes

        except Exception as e:
            print(f"Error getting season episodes: {e}")
            return None

    def download_episode_stills(self, episodes):
        """Download still images for episodes"""
        episode_images = {}

        for episode in episodes:
            if episode['still_path']:
                image_url = f"{self.IMAGE_BASE_URL}{episode['still_path']}"

                try:
                    response = self.session.get(image_url)
                    response.raise_for_status()

                    # Convert to PIL Image
                    image = Image.open(BytesIO(response.content))

                    episode_images[episode['episode_number']] = {
                        'image': image,
                        'name': episode['name'],
                        'episode_number': episode['episode_number']
                    }

                except Exception as e:
                    print(f"Error downloading image for episode {episode['episode_number']}: {e}")

        return episode_images
