from flask import Flask, render_template, jsonify, request, send_file
import os
import re
import base64
from io import BytesIO
from pathlib import Path
from services.tmdb_service import TMDBService
from services.video_service import VideoService
from services.image_matcher import ImageMatcher
from services.file_service import FileService

app = Flask(__name__)

# Store data temporarily (in production, use Redis or similar)
temp_data = {}

# Configuration
RIPPER_PATH = os.environ.get('RIPPER_PATH', '/mnt/ripper')
SHOWS_PATH = os.environ.get('SHOWS_PATH', '/mnt/shows')
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')

# Initialize services
tmdb_service = TMDBService(TMDB_API_KEY)
video_service = VideoService()
image_matcher = ImageMatcher()
file_service = FileService()


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/folders', methods=['GET'])
def list_folders():
    """List all folders in the ripper directory"""
    try:
        folders = []
        if not os.path.exists(RIPPER_PATH):
            return jsonify({'error': 'Ripper path does not exist'}), 404

        for item in os.listdir(RIPPER_PATH):
            item_path = os.path.join(RIPPER_PATH, item)
            if os.path.isdir(item_path):
                # Parse folder name
                parsed = file_service.parse_folder_name(item)
                if parsed:
                    # Count .mkv files
                    mkv_files = list(Path(item_path).glob('*.mkv'))
                    folders.append({
                        'name': item,
                        'path': item_path,
                        'show_name': parsed['show_name'],
                        'season': parsed['season'],
                        'disk': parsed['disk'],
                        'file_count': len(mkv_files)
                    })

        return jsonify({'folders': folders})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/identify', methods=['POST'])
def identify_episodes():
    """Identify episodes in a folder"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        show_name = data.get('show_name')
        season_num = data.get('season')

        if not folder_path or not show_name or season_num is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Search for TV show on TMDB
        show_info = tmdb_service.search_tv_show(show_name)
        if not show_info:
            return jsonify({'error': 'TV show not found on TheMovieDB'}), 404

        show_id = show_info['id']

        # Get episode information for the season
        episodes = tmdb_service.get_season_episodes(show_id, season_num)
        if not episodes:
            return jsonify({'error': f'Season {season_num} not found'}), 404

        # Download episode stills
        episode_images = tmdb_service.download_episode_stills(episodes)

        # Get all .mkv files in the folder
        mkv_files = sorted(Path(folder_path).glob('*.mkv'))

        # Extract frames from each video file
        video_frames = {}
        for mkv_file in mkv_files:
            frames = video_service.extract_frames(str(mkv_file))
            video_frames[mkv_file.name] = frames

        # Match videos to episodes
        matches = image_matcher.match_videos_to_episodes(video_frames, episode_images)

        # Store data for image retrieval
        session_id = f"{folder_path}_{season_num}"
        temp_data[session_id] = {
            'video_frames': video_frames,
            'episode_images': episode_images,
            'matches': matches
        }

        # Convert images to base64 for sending to frontend
        def image_to_base64(img):
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()

        # Format response
        results = []
        for video_file, match_info in matches.items():
            # Get the matched frame
            matched_frame_idx = match_info.get('matched_frame_index', 0)
            matched_frame = video_frames[video_file][matched_frame_idx] if video_frames[video_file] else None

            # Get the matched episode still
            matched_episode_num = match_info.get('best_match_episode', 0)
            matched_still = episode_images.get(matched_episode_num, {}).get('image') if matched_episode_num else None

            result = {
                'original_name': video_file,
                'episode_number': match_info['episode_number'],
                'episode_name': match_info['episode_name'],
                'confidence': round(match_info['confidence'] * 100, 2),
                'show_name': show_info['name'],
                'season': season_num,
                'matched_frame': image_to_base64(matched_frame) if matched_frame else None,
                'matched_still': image_to_base64(matched_still) if matched_still else None,
                'all_scores': match_info.get('all_scores', {}),
                'is_match': match_info['confidence'] >= 0.4  # 40% threshold for "match"
            }
            results.append(result)

        # Sort by episode number
        results.sort(key=lambda x: x['episode_number'])

        return jsonify({
            'show_name': show_info['name'],
            'season': season_num,
            'episodes': results,
            'session_id': session_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/rename', methods=['POST'])
def rename_files():
    """Rename files based on identified episodes"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        episodes = data.get('episodes')

        if not folder_path or not episodes:
            return jsonify({'error': 'Missing required parameters'}), 400

        results = file_service.rename_files(folder_path, episodes)

        return jsonify({'results': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/move', methods=['POST'])
def move_files():
    """Move files to the shows directory (without renaming)"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        episodes = data.get('episodes')
        show_name = data.get('show_name')
        season = data.get('season')

        if not folder_path or not episodes or not show_name or season is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        results = file_service.move_files(
            folder_path, episodes, show_name, season, SHOWS_PATH
        )

        return jsonify({'results': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
