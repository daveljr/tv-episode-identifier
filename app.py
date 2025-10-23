from flask import Flask, render_template, jsonify, request, send_file
import os
import re
import base64
from io import BytesIO
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image
from services.tmdb_service import TMDBService
from services.video_service import VideoService
from services.image_matcher import ImageMatcher
from services.file_service import FileService

app = Flask(__name__)

# Store data temporarily (in production, use Redis or similar)
temp_data = {}

# Configuration
INPUT_PATH = os.environ.get('INPUT_PATH', '/app/input')
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/app/output')
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')

# Frame extraction settings
FRAME_HEIGHT = int(os.environ.get('FRAME_HEIGHT', '480'))
FRAME_START_OFFSET = float(os.environ.get('FRAME_START_OFFSET', '0.0'))
FRAME_END_OFFSET = float(os.environ.get('FRAME_END_OFFSET', '1.0'))

# Setup logging
log_directory = os.environ.get('LOG_PATH', '/app/logs')
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, 'app.log')

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize services
tmdb_service = TMDBService(api_key=TMDB_API_KEY, cache_dir='/app/temp/downloads')
video_service = VideoService(frame_height=FRAME_HEIGHT, start_offset=FRAME_START_OFFSET, end_offset=FRAME_END_OFFSET, base_temp_dir='/app/temp')
image_matcher = ImageMatcher()  # Automatically uses GPU if available
file_service = FileService()

logger.info(f"Application starting with configuration:")
logger.info(f"  INPUT_PATH={INPUT_PATH}")
logger.info(f"  OUTPUT_PATH={OUTPUT_PATH}")
logger.info(f"  FRAME_HEIGHT={FRAME_HEIGHT}")
logger.info(f"  FRAME_RANGE={FRAME_START_OFFSET*100:.0f}%-{FRAME_END_OFFSET*100:.0f}%")


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/folders', methods=['GET'])
def list_folders():
    """List all folders in the ripper directory"""
    try:
        logger.info(f"Listing folders in {INPUT_PATH}")
        folders = []
        if not os.path.exists(INPUT_PATH):
            logger.error(f"Ripper path does not exist: {INPUT_PATH}")
            return jsonify({'error': 'Ripper path does not exist'}), 404

        for item in os.listdir(INPUT_PATH):
            item_path = os.path.join(INPUT_PATH, item)
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

        logger.info(f"Found {len(folders)} valid folders")
        return jsonify({'folders': folders})
    except Exception as e:
        logger.exception(f"Error listing folders: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/identify', methods=['POST'])
def identify_episodes():
    """Identify episodes in a folder"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        show_name = data.get('show_name')
        season_num = data.get('season')

        logger.info(f"Identifying episodes for {show_name} Season {season_num} in {folder_path}")

        if not folder_path or not show_name or season_num is None:
            logger.error("Missing required parameters")
            return jsonify({'error': 'Missing required parameters'}), 400

        # Search for TV show on TMDB
        logger.info(f"Searching TMDB for: {show_name}")
        show_info = tmdb_service.search_tv_show(show_name)
        if not show_info:
            logger.error(f"TV show not found on TMDB: {show_name}")
            return jsonify({'error': 'TV show not found on TheMovieDB'}), 404

        show_id = show_info['id']

        # Format show name with year: "Show Name (YYYY)"
        tmdb_show_name = show_info['name']
        first_air_date = show_info.get('first_air_date', '')
        if first_air_date and len(first_air_date) >= 4:
            year = first_air_date[:4]
            formatted_show_name = f"{tmdb_show_name} ({year})"
        else:
            formatted_show_name = tmdb_show_name

        logger.info(f"Found show: {formatted_show_name} (ID: {show_id})")

        # Get episode information and images with caching
        logger.info(f"Fetching episode data for Season {season_num} (with caching)")
        episodes, episode_images = tmdb_service.get_season_data_cached(show_id, season_num)

        if not episodes:
            logger.error(f"Season {season_num} not found")
            return jsonify({'error': f'Season {season_num} not found'}), 404

        logger.info(f"Loaded {len(episodes)} episodes with {len(episode_images)} having images")

        # Get all .mkv files in the folder and filter by duration
        mkv_files = sorted(Path(folder_path).glob('*.mkv'))
        logger.info(f"Found {len(mkv_files)} MKV files")

        # Filter out files shorter than 20 minutes (typical TV episode length)
        valid_mkv_files = []
        for mkv_file in mkv_files:
            if video_service.is_minimum_duration(str(mkv_file), min_duration_minutes=20):
                valid_mkv_files.append(mkv_file)

        logger.info(f"Processing {len(valid_mkv_files)} files that meet minimum duration (20 min)")

        # Extract folder name from path for organizing frames
        folder_name = Path(folder_path).name

        # Extract frames from each video file
        video_frames = {}
        for mkv_file in valid_mkv_files:
            logger.info(f"Extracting ALL frames from {mkv_file.name}")
            frame_paths = video_service.extract_frames(str(mkv_file), folder_name)
            video_frames[mkv_file.name] = frame_paths
            logger.info(f"Extracted {len(frame_paths)} frames from {mkv_file.name}")

        # Match videos to episodes
        logger.info("Matching videos to episodes using image comparison")
        matches = image_matcher.match_videos_to_episodes(video_frames, episode_images)

        # Store data for image retrieval
        session_id = f"{folder_path}_{season_num}"
        temp_data[session_id] = {
            'video_frames': video_frames,
            'episode_images': episode_images,
            'matches': matches
        }

        # Convert images to base64 for sending to frontend
        def image_path_to_base64(image_path):
            """Load image from file path and convert to base64"""
            try:
                with Image.open(image_path) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    return base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                logger.error(f"Error converting image to base64: {e}")
                return None

        # Format response
        results = []
        for video_file, match_info in matches.items():
            # Get the matched frame (file path)
            matched_frame_idx = match_info.get('matched_frame_index', 0)
            matched_frame_path = video_frames[video_file][matched_frame_idx] if video_frames[video_file] else None

            # Get the matched episode still (first image from the list)
            matched_episode_num = match_info.get('best_match_episode', 0)
            matched_still_path = None
            if matched_episode_num and matched_episode_num in episode_images:
                # Get first image from the images list (list of file paths)
                image_paths = episode_images[matched_episode_num].get('images', [])
                matched_still_path = image_paths[0] if image_paths else None

            result = {
                'original_name': video_file,
                'episode_number': match_info['episode_number'],
                'episode_name': match_info['episode_name'],
                'confidence': round(match_info['confidence'] * 100, 2),
                'show_name': formatted_show_name,  # Use formatted name with year
                'season': season_num,
                'matched_frame': image_path_to_base64(matched_frame_path) if matched_frame_path else None,
                'matched_still': image_path_to_base64(matched_still_path) if matched_still_path else None,
                'all_scores': match_info.get('all_scores', {}),
                'is_match': match_info['confidence'] >= 0.8  # 80% threshold for "match"
            }
            results.append(result)

        # Sort by episode number
        results.sort(key=lambda x: x['episode_number'])

        # Clean up temporary frame files for this folder
        video_service.cleanup_folder(folder_name)

        logger.info(f"Successfully identified {len(results)} episodes")
        return jsonify({
            'show_name': formatted_show_name,  # Use formatted name with year
            'season': season_num,
            'episodes': results,
            'session_id': session_id
        })

    except Exception as e:
        # Clean up temporary files even on error
        if 'folder_name' in locals():
            video_service.cleanup_folder(folder_name)
        logger.exception(f"Error identifying episodes: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rename', methods=['POST'])
def rename_files():
    """Rename files based on identified episodes"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        episodes = data.get('episodes')

        logger.info(f"Renaming {len(episodes)} files in {folder_path}")

        if not folder_path or not episodes:
            logger.error("Missing required parameters for rename")
            return jsonify({'error': 'Missing required parameters'}), 400

        results = file_service.rename_files(folder_path, episodes)
        logger.info(f"Successfully renamed {len(results)} files")

        return jsonify({'results': results})

    except Exception as e:
        logger.exception(f"Error renaming files: {e}")
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

        logger.info(f"Moving {len(episodes)} files from {folder_path} to {OUTPUT_PATH}/{show_name}/Season {season}")

        if not folder_path or not episodes or not show_name or season is None:
            logger.error("Missing required parameters for move")
            return jsonify({'error': 'Missing required parameters'}), 400

        results = file_service.move_files(
            folder_path, episodes, show_name, season, OUTPUT_PATH
        )
        logger.info(f"Successfully moved {len(results)} files")

        return jsonify({'results': results})

    except Exception as e:
        logger.exception(f"Error moving files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/episode-images', methods=['GET'])
def get_episode_images():
    """Get all TMDB images for all episodes with match scores for a specific video"""
    try:
        session_id = request.args.get('session_id')
        video_file = request.args.get('video_file')

        if not session_id or not video_file:
            logger.error("Missing session_id or video_file parameter")
            return jsonify({'error': 'Missing required parameters'}), 400

        logger.info(f"Retrieving episode images for session {session_id}, video {video_file}")

        if session_id not in temp_data:
            logger.error(f"Session not found: {session_id}")
            return jsonify({'error': 'Session not found'}), 404

        session = temp_data[session_id]
        episode_images = session['episode_images']
        matches = session['matches']
        video_frames = session['video_frames']

        if video_file not in matches:
            logger.error(f"Video file not found in matches: {video_file}")
            return jsonify({'error': 'Video file not found'}), 404

        match_info = matches[video_file]
        all_scores = match_info.get('all_scores', {})
        matched_frame_idx = match_info.get('matched_frame_index', 0)
        matched_frame_path = video_frames[video_file][matched_frame_idx] if video_file in video_frames else None

        # Build response with all episodes and their images
        episodes_data = []
        for episode_num, episode_data in sorted(episode_images.items()):
            image_paths = episode_data.get('images', [])
            score = all_scores.get(episode_num, 0.0)

            # Convert all images for this episode to base64
            episode_images_b64 = []
            for img_path in image_paths:
                img_b64 = image_path_to_base64(img_path)
                if img_b64:
                    episode_images_b64.append(img_b64)

            episodes_data.append({
                'episode_number': episode_data['episode_number'],
                'episode_name': episode_data['name'],
                'images': episode_images_b64,
                'score': round(score * 100, 2),
                'is_match': score >= 0.8
            })

        return jsonify({
            'video_file': video_file,
            'matched_frame': image_path_to_base64(matched_frame_path) if matched_frame_path else None,
            'episodes': episodes_data
        })

    except Exception as e:
        logger.exception(f"Error retrieving episode images: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Retrieve application logs"""
    try:
        lines = request.args.get('lines', default=500, type=int)
        lines = min(lines, 5000)  # Cap at 5000 lines

        logger.info(f"Retrieving last {lines} log lines")

        if not os.path.exists(log_file):
            return jsonify({'logs': 'No logs available yet'})

        # Read the log file
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()

        # Get the last N lines
        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        logs_content = ''.join(recent_lines)

        return jsonify({
            'logs': logs_content,
            'total_lines': len(all_lines),
            'returned_lines': len(recent_lines)
        })

    except Exception as e:
        logger.exception(f"Error retrieving logs: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)
