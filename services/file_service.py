import re
import os
import shutil
from pathlib import Path


class FileService:
    """Service for file operations"""

    def parse_folder_name(self, folder_name):
        """
        Parse folder name in format: {tv show name} s{season #}d{disk #}
        Example: "The Office s01d1" -> {'show_name': 'The Office', 'season': 1, 'disk': 1}
        """
        # Pattern: anything followed by s{number}d{number}
        pattern = r'^(.+?)\s*s(\d+)d(\d+)$'
        match = re.match(pattern, folder_name, re.IGNORECASE)

        if match:
            return {
                'show_name': match.group(1).strip(),
                'season': int(match.group(2)),
                'disk': int(match.group(3))
            }
        return None

    def rename_files(self, folder_path, episodes):
        """
        Rename files in the folder based on episode information
        Format: {TV Show Name} s{Season # padded to 2}e{Episode # padded to 2}.{extension}
        """
        results = []

        for episode in episodes:
            original_path = os.path.join(folder_path, episode['original_name'])
            if not os.path.exists(original_path):
                results.append({
                    'original': episode['original_name'],
                    'new': None,
                    'success': False,
                    'error': 'File not found'
                })
                continue

            # Get file extension
            ext = Path(original_path).suffix

            # Create new filename
            new_filename = f"{episode['show_name']} s{episode['season']:02d}e{episode['episode_number']:02d}{ext}"
            new_path = os.path.join(folder_path, new_filename)

            try:
                os.rename(original_path, new_path)
                results.append({
                    'original': episode['original_name'],
                    'new': new_filename,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'original': episode['original_name'],
                    'new': new_filename,
                    'success': False,
                    'error': str(e)
                })

        return results

    def move_files(self, folder_path, episodes, show_name, season, shows_base_path):
        """
        Move files to the shows directory (using their current names)
        Destination: /app/output/{TV Show Name}/Season {Season # padded to 2}/
        Files that are still named "title_*" are moved to an extras folder instead
        """
        results = []

        # Create destination directories
        season_dir = os.path.join(shows_base_path, show_name, f"Season {season:02d}")
        extras_dir = os.path.join(season_dir, "extras")
        os.makedirs(season_dir, exist_ok=True)

        for episode in episodes:
            original_path = os.path.join(folder_path, episode['original_name'])
            if not os.path.exists(original_path):
                results.append({
                    'original': episode['original_name'],
                    'new': None,
                    'success': False,
                    'error': 'File not found'
                })
                continue

            # Use the current filename from the episode data
            current_filename = Path(original_path).name

            # Check if file is still named "title_*" (unrenamed)
            if re.match(r'^title_\d+', current_filename, re.IGNORECASE):
                # Move to extras folder
                os.makedirs(extras_dir, exist_ok=True)
                new_path = os.path.join(extras_dir, current_filename)
            else:
                # Move to season folder
                new_path = os.path.join(season_dir, current_filename)

            try:
                shutil.move(original_path, new_path)
                results.append({
                    'original': episode['original_name'],
                    'new': new_path,
                    'success': True,
                    'moved_to_extras': 'extras' in new_path
                })
            except Exception as e:
                results.append({
                    'original': episode['original_name'],
                    'new': new_path,
                    'success': False,
                    'error': str(e)
                })

        return results
