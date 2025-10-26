from flask import Flask, render_template, jsonify, request, send_file, make_response
import os
import re
import base64
from io import BytesIO
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image
import threading
from services.tmdb_service import TMDBService
from services.video_service import VideoService
from services.image_matcher import ImageMatcher
from services.file_service import FileService
from services.job_manager import JobManager
from services.config_service import ConfigService
from services.folder_watcher import FolderWatcher

app = Flask(__name__)

# Store data temporarily (in production, use Redis or similar)
temp_data = {}

# Initialize configuration service
config_service = ConfigService(config_dir='/app/config')

# Force fixed paths - not configurable
INPUT_PATH = '/app/input'
OUTPUT_PATH = '/app/output'

# Setup logging - force fixed path
log_directory = '/app/logs'
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

# Initialize services with config
tmdb_service = TMDBService(
    api_key=config_service.tmdb_api_key,
    cache_dir='/app/temp/downloads',
    frame_height=config_service.frame_height
)
video_service = VideoService(
    frame_height=config_service.frame_height,
    start_offset=config_service.frame_start_offset,
    end_offset=config_service.frame_end_offset,
    fps=config_service.frame_fps,
    base_temp_dir='/app/temp',
    parallel_enabled=config_service.extraction_parallel_enabled,
    parallel_workers=config_service.extraction_parallel_workers,
    output_format=config_service.extraction_output_format
)
image_matcher = ImageMatcher(
    max_workers=config_service.matcher_max_workers,
    early_stop_threshold=config_service.matcher_early_stop_threshold,
    max_candidates=config_service.matcher_max_candidates,
    gpu_batch_size=config_service.matcher_gpu_batch_size,
    cpu_batch_size=config_service.matcher_cpu_batch_size
)  # Automatically uses GPU if available
file_service = FileService()
job_manager = JobManager(jobs_dir='/app/temp/jobs', video_service=video_service)


# Helper function for image conversion
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

logger.info(f"Application starting with configuration:")
logger.info(f"  FRAME_HEIGHT={config_service.frame_height}")
logger.info(f"  FRAME_FPS={config_service.frame_fps if config_service.frame_fps else 'ALL'}")
logger.info(f"  FRAME_RANGE={config_service.frame_start_offset*100:.0f}%-{config_service.frame_end_offset*100:.0f}%")
logger.info(f"  AUTO_PROCESSING={config_service.auto_processing_enabled}")
logger.info(f"  TMDB_API_KEY={'configured' if config_service.tmdb_api_key else 'not set'}")


# Helper function for auto-processing callback
def auto_process_folder(folder_path, show_name, season):
    """Callback function for folder watcher to start auto-processing"""
    try:
        logger.info(f"Auto-processing triggered for: {show_name} S{season}")

        # Create a job
        job_id = job_manager.create_job(folder_path, show_name, season)

        # Start processing in background thread
        thread = threading.Thread(
            target=process_identification_job,
            args=(job_id, folder_path, show_name, season),
            daemon=True
        )
        thread.start()

        logger.info(f"Auto-processing job {job_id} started for {show_name} Season {season}")

    except Exception as e:
        logger.exception(f"Error in auto-processing: {e}")


# Initialize folder watcher
folder_watcher = FolderWatcher(
    input_path=INPUT_PATH,
    config_service=config_service,
    file_service=file_service,
    process_callback=auto_process_folder
)

# Start folder watcher if auto-processing is enabled
if config_service.auto_processing_enabled:
    folder_watcher.start()
    logger.info("Folder watcher started (auto-processing enabled)")


def process_identification_job(job_id, folder_path, show_name, season_num):
    """Process an identification job in the background"""
    folder_name = None
    try:
        job_manager.set_processing(job_id)
        logger.info(f"[Job {job_id}] Starting identification for {show_name} Season {season_num}")

        # Search for TV show on TMDB
        job_manager.update_progress(job_id, 10, f'Searching TMDB for {show_name}...')
        logger.info(f"[Job {job_id}] Searching TMDB for: {show_name}")
        show_info = tmdb_service.search_tv_show(show_name)
        if not show_info:
            logger.error(f"[Job {job_id}] TV show not found on TMDB: {show_name}")
            job_manager.set_error(job_id, 'TV show not found on TheMovieDB')
            return

        show_id = show_info['id']

        # Format show name with year
        tmdb_show_name = show_info['name']
        first_air_date = show_info.get('first_air_date', '')
        if first_air_date and len(first_air_date) >= 4:
            year = first_air_date[:4]
            formatted_show_name = f"{tmdb_show_name} ({year})"
        else:
            formatted_show_name = tmdb_show_name

        logger.info(f"[Job {job_id}] Found show: {formatted_show_name} (ID: {show_id})")

        # Get episode information and images with caching
        job_manager.update_progress(job_id, 20, f'Fetching episode data for Season {season_num}...')
        logger.info(f"[Job {job_id}] Fetching episode data for Season {season_num}")
        episodes, episode_images = tmdb_service.get_season_data_cached(show_id, season_num)

        if not episodes:
            logger.error(f"[Job {job_id}] Season {season_num} not found")
            job_manager.set_error(job_id, f'Season {season_num} not found')
            return

        logger.info(f"[Job {job_id}] Loaded {len(episodes)} episodes with {len(episode_images)} having images")

        # Get all .mkv files in the folder and categorize by duration
        job_manager.update_progress(job_id, 30, 'Finding video files...')
        mkv_files = sorted(Path(folder_path).glob('*.mkv'))
        logger.info(f"[Job {job_id}] Found {len(mkv_files)} MKV files")

        # Separate files by duration and capture duration info
        valid_mkv_files = []
        extra_files = []
        file_durations = {}  # Store duration for each file

        for mkv_file in mkv_files:
            duration = video_service.get_duration_formatted(str(mkv_file))
            file_durations[mkv_file.name] = duration

            if video_service.is_minimum_duration(str(mkv_file), min_duration_minutes=20):
                valid_mkv_files.append(mkv_file)
            else:
                extra_files.append(mkv_file)

        logger.info(f"[Job {job_id}] Processing {len(valid_mkv_files)} files that meet minimum duration (20 min)")
        logger.info(f"[Job {job_id}] Found {len(extra_files)} extra files (< 20 min)")

        # Extract folder name from path for organizing frames
        folder_name = Path(folder_path).name

        # Extract frames from each video file using job-specific directory
        video_frames = {}
        total_files = len(valid_mkv_files)
        for idx, mkv_file in enumerate(valid_mkv_files):
            progress = 30 + int((idx / total_files) * 40)  # 30-70%
            job_manager.update_progress(job_id, progress, f'Extracting frames from {mkv_file.name} ({idx+1}/{total_files})...')
            logger.info(f"[Job {job_id}] Extracting frames from {mkv_file.name}")
            frame_paths = video_service.extract_frames(str(mkv_file), folder_name=folder_name, job_id=job_id)
            video_frames[mkv_file.name] = frame_paths
            logger.info(f"[Job {job_id}] Extracted {len(frame_paths)} frames from {mkv_file.name}")

        # Match videos to episodes
        job_manager.update_progress(job_id, 75, 'Matching videos to episodes...')
        logger.info(f"[Job {job_id}] Matching videos to episodes using image comparison")
        matches = image_matcher.match_videos_to_episodes(video_frames, episode_images)

        # Store data for image retrieval
        session_id = f"{folder_path}_{season_num}"
        temp_data[session_id] = {
            'video_frames': video_frames,
            'episode_images': episode_images,
            'matches': matches
        }

        # Format response
        job_manager.update_progress(job_id, 90, 'Formatting results...')
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

            # Get all video frames for this video
            all_video_frames = video_frames.get(video_file, [])
            all_image_frame_indices = match_info.get('all_image_frame_indices', {})
            all_image_scores = match_info.get('all_image_scores', {})

            # Build comprehensive episode match data with all images (store paths, not base64)
            all_episodes_data = []
            for ep_num, ep_data in sorted(episode_images.items()):
                ep_image_paths = ep_data.get('images', [])
                ep_score = match_info.get('all_scores', {}).get(ep_num, 0.0)
                ep_image_scores = all_image_scores.get(ep_num, [])
                ep_frame_indices = all_image_frame_indices.get(ep_num, [])

                # Store paths to TMDB stills and matched video frames
                ep_matched_frame_paths = []
                for idx in range(len(ep_image_paths)):
                    # Matched video frame for this specific TMDB still
                    if idx < len(ep_frame_indices) and ep_frame_indices[idx] < len(all_video_frames):
                        frame_path = all_video_frames[ep_frame_indices[idx]]
                        ep_matched_frame_paths.append(frame_path if frame_path else None)
                    else:
                        ep_matched_frame_paths.append(None)

                all_episodes_data.append({
                    'episode_number': ep_data['episode_number'],
                    'episode_name': ep_data['name'],
                    'image_paths': ep_image_paths,  # Store paths instead of base64
                    'matched_frame_paths': ep_matched_frame_paths,  # Store paths instead of base64
                    'score': round(ep_score * 100, 2),
                    'image_scores': [round(s * 100, 2) for s in ep_image_scores],
                    'is_match': ep_score >= 0.8
                })

            result = {
                'original_name': video_file,
                'episode_number': match_info['episode_number'],
                'episode_name': match_info['episode_name'],
                'confidence': round(match_info['confidence'] * 100, 2),
                'show_name': formatted_show_name,
                'season': season_num,
                'matched_frame_path': matched_frame_path,  # Store path instead of base64
                'matched_still_path': matched_still_path,  # Store path instead of base64
                'all_scores': match_info.get('all_scores', {}),
                'is_match': match_info['confidence'] >= 0.8,
                'is_extra': False,
                'keep': True,  # Default to keeping all files
                'duration': file_durations.get(video_file, 'Unknown'),
                'all_episodes_data': all_episodes_data  # Store all episode comparison data with paths
            }
            results.append(result)

        # Add extra files (< 20 minutes) to results
        for extra_file in extra_files:
            result = {
                'original_name': extra_file.name,
                'episode_number': None,
                'episode_name': extra_file.stem,  # Use filename without extension
                'confidence': 0,
                'show_name': formatted_show_name,
                'season': season_num,
                'matched_frame': None,
                'matched_still': None,
                'all_scores': {},
                'is_match': False,
                'is_extra': True,
                'keep': True,  # Default to keeping all files
                'duration': file_durations.get(extra_file.name, 'Unknown')
            }
            results.append(result)

        # Sort by episode number (extras will be at the end with None)
        results.sort(key=lambda x: (x['episode_number'] is None, x['episode_number'] if x['episode_number'] is not None else 0))

        # Keep frames with the job - they'll be cleaned up when job is deleted
        # (No longer call cleanup_folder here)

        # Mark job as completed
        job_manager.set_completed(job_id, {
            'show_name': formatted_show_name,
            'season': season_num,
            'episodes': results,
            'session_id': session_id
        }, session_id)

        logger.info(f"[Job {job_id}] Successfully identified {len(results)} episodes")

    except Exception as e:
        # Frames will be cleaned up when job is deleted
        logger.exception(f"[Job {job_id}] Error identifying episodes: {e}")
        job_manager.set_error(job_id, str(e))


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/folders/<path:folder_name>/identify', methods=['POST'])
def identify_folder_pattern(folder_name):
    """Rename a folder to match the pattern: {show_name} s{season}d{disk}"""
    try:
        data = request.json
        show_name = data.get('show_name')
        season = data.get('season')
        disk = data.get('disk')

        if not show_name or season is None or disk is None:
            return jsonify({'error': 'Show name, season, and disk are required'}), 400

        # Validate inputs
        try:
            season = int(season)
            disk = int(disk)
        except ValueError:
            return jsonify({'error': 'Season and disk must be numbers'}), 400

        # Format the new folder name
        new_name = f"{show_name} s{season}d{disk}"

        old_path = os.path.join(INPUT_PATH, folder_name)
        new_path = os.path.join(INPUT_PATH, new_name)

        logger.info(f"Identifying folder {old_path} as {new_name}")

        if not os.path.exists(old_path):
            return jsonify({'error': 'Folder not found'}), 404

        if os.path.exists(new_path):
            return jsonify({'error': f'A folder named "{new_name}" already exists'}), 409

        # Check if folder is being processed
        if job_manager.is_folder_processing(old_path):
            return jsonify({'error': 'Cannot rename folder while it is being processed'}), 409

        # Rename the folder
        os.rename(old_path, new_path)
        logger.info(f"Successfully identified folder as {new_name}")

        return jsonify({
            'success': True,
            'old_name': folder_name,
            'new_name': new_name,
            'show_name': show_name,
            'season': season,
            'disk': disk
        })

    except Exception as e:
        logger.exception(f"Error identifying folder: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/folders/<path:folder_name>/rename', methods=['POST'])
def rename_folder(folder_name):
    """Rename a folder in the input directory"""
    try:
        data = request.json
        new_name = data.get('new_name')

        if not new_name:
            return jsonify({'error': 'New name is required'}), 400

        old_path = os.path.join(INPUT_PATH, folder_name)
        new_path = os.path.join(INPUT_PATH, new_name)

        logger.info(f"Renaming folder {old_path} to {new_path}")

        if not os.path.exists(old_path):
            return jsonify({'error': 'Folder not found'}), 404

        if os.path.exists(new_path):
            return jsonify({'error': 'A folder with that name already exists'}), 409

        # Check if folder is being processed
        if job_manager.is_folder_processing(old_path):
            return jsonify({'error': 'Cannot rename folder while it is being processed'}), 409

        # Rename the folder
        os.rename(old_path, new_path)
        logger.info(f"Successfully renamed folder to {new_name}")

        return jsonify({
            'success': True,
            'old_name': folder_name,
            'new_name': new_name
        })

    except Exception as e:
        logger.exception(f"Error renaming folder: {e}")
        return jsonify({'error': str(e)}), 500


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
                # Count .mkv files
                mkv_files = list(Path(item_path).glob('*.mkv'))

                # Get job status for this folder
                is_processing = job_manager.is_folder_processing(item_path)
                job_status = job_manager.get_folder_job_status(item_path)

                # Parse folder name
                parsed = file_service.parse_folder_name(item)

                if parsed:
                    # Folder matches the pattern
                    folders.append({
                        'name': item,
                        'path': item_path,
                        'show_name': parsed['show_name'],
                        'season': parsed['season'],
                        'disk': parsed['disk'],
                        'file_count': len(mkv_files),
                        'is_processing': is_processing,
                        'job_status': job_status if job_status else None,
                        'matches_pattern': True
                    })
                else:
                    # Folder doesn't match pattern - still list it
                    folders.append({
                        'name': item,
                        'path': item_path,
                        'show_name': None,
                        'season': None,
                        'disk': None,
                        'file_count': len(mkv_files),
                        'is_processing': is_processing,
                        'job_status': job_status if job_status else None,
                        'matches_pattern': False
                    })

        # Sort folders alphabetically by name
        folders.sort(key=lambda x: x['name'].lower())

        logger.info(f"Found {len(folders)} valid folders")
        return jsonify({'folders': folders})
    except Exception as e:
        logger.exception(f"Error listing folders: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/identify', methods=['POST'])
def identify_episodes():
    """Create a new identification job and process it in the background"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        show_name = data.get('show_name')
        season_num = data.get('season')

        logger.info(f"Creating identification job for {show_name} Season {season_num} in {folder_path}")

        if not folder_path or not show_name or season_num is None:
            logger.error("Missing required parameters")
            return jsonify({'error': 'Missing required parameters'}), 400

        # Check if folder is already being processed
        if job_manager.is_folder_processing(folder_path):
            logger.warning(f"Folder {folder_path} is already being processed")
            return jsonify({'error': 'This folder is already being identified. Please wait for the current job to complete.'}), 409

        # Create a job
        job_id = job_manager.create_job(folder_path, show_name, season_num)

        # Start processing in background thread
        thread = threading.Thread(
            target=process_identification_job,
            args=(job_id, folder_path, show_name, season_num),
            daemon=True
        )
        thread.start()

        logger.info(f"Started background job {job_id}")
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Job created and processing started'
        }), 202

    except Exception as e:
        logger.exception(f"Error creating identification job: {e}")
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

        # Mark the job as moved
        all_jobs = job_manager.get_all_jobs()
        for job in all_jobs:
            if job.get('folder_path') == folder_path:
                job_manager.mark_as_moved(job['job_id'])
                logger.info(f"Marked job {job['job_id']} as moved")
                break

        return jsonify({'results': results})

    except Exception as e:
        logger.exception(f"Error moving files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-and-move', methods=['POST'])
def save_and_move_files():
    """Combined endpoint: Rename non-extras and move all files"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        episodes = data.get('episodes')
        show_name = data.get('show_name')
        season = data.get('season')

        logger.info(f"Save & Move: Processing {len(episodes)} files from {folder_path}")

        if not folder_path or not episodes or not show_name or season is None:
            logger.error("Missing required parameters for save-and-move")
            return jsonify({'error': 'Missing required parameters'}), 400

        # Find the job for this folder
        job_id = None
        all_jobs = job_manager.get_all_jobs()
        for job in all_jobs:
            if job.get('folder_path') == folder_path:
                job_id = job['job_id']
                break

        # Check if already moved
        if job_id:
            job = job_manager.get_job(job_id)
            if job and job.get('moved'):
                return jsonify({'error': 'Files have already been moved'}), 409

            # Set status to moving
            job_manager.set_moving(job_id)

        try:
            # Step 1: Rename files (only non-extras that are marked as keep)
            files_to_rename = [ep for ep in episodes if not ep.get('is_extra', False)]

            if files_to_rename:
                logger.info(f"Renaming {len(files_to_rename)} non-extra files")
                rename_results = file_service.rename_files(folder_path, files_to_rename)

                # Update episode names with renamed filenames
                for result in rename_results:
                    if result.get('success'):
                        for ep in episodes:
                            if ep['original_name'] == result['original']:
                                ep['original_name'] = result['new']
                                break

            # Step 2: Move all files (renamed episodes + extras)
            logger.info(f"Moving {len(episodes)} files to {OUTPUT_PATH}/{show_name}/Season {season}")
            move_results = file_service.move_files(
                folder_path, episodes, show_name, season, OUTPUT_PATH
            )

            # Mark the job as moved and set back to completed status
            if job_id:
                job_manager.mark_as_moved(job_id)
                job_manager.update_job(
                    job_id,
                    status='completed',
                    progress_message='Completed - Files Moved'
                )

            logger.info(f"Successfully completed save & move for {len(episodes)} files")
            return jsonify({
                'success': True,
                'rename_results': rename_results if files_to_rename else [],
                'move_results': move_results
            })

        except Exception as inner_error:
            # Reset job status back to completed on error
            if job_id:
                job_manager.update_job(
                    job_id,
                    status='completed',
                    progress_message='Completed'
                )
            raise inner_error

    except Exception as e:
        logger.exception(f"Error in save-and-move: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/episode-images', methods=['GET'])
def get_episode_images():
    """Get all TMDB images for all episodes with match scores for a specific video"""
    try:
        session_id = request.args.get('session_id')
        video_file = request.args.get('video_file')
        job_id = request.args.get('job_id')  # Optional: get from job instead

        if not session_id or not video_file:
            logger.error("Missing session_id or video_file parameter")
            return jsonify({'error': 'Missing required parameters'}), 400

        logger.info(f"Retrieving episode images for session {session_id}, video {video_file}")

        # First, try to get data from the stored job (persists after frame cleanup)
        if job_id:
            job = job_manager.get_job(job_id)
            if job and job.get('results'):
                # Find the result for this video file
                for result in job['results'].get('episodes', []):
                    if result.get('original_name') == video_file:
                        # Return the stored all_episodes_data, converting paths to base64
                        if 'all_episodes_data' in result:
                            logger.info(f"Using stored episode data from job {job_id}")

                            # Convert paths to base64 for display
                            episodes_with_images = []
                            for ep_data in result['all_episodes_data']:
                                ep_copy = ep_data.copy()

                                # Convert TMDB image paths to base64
                                if ep_data.get('image_paths'):
                                    ep_copy['images'] = [image_path_to_base64(path) for path in ep_data['image_paths']]

                                # Convert matched frame paths to base64
                                if ep_data.get('matched_frame_paths'):
                                    ep_copy['matched_frames'] = [
                                        image_path_to_base64(path) if path else None
                                        for path in ep_data['matched_frame_paths']
                                    ]

                                episodes_with_images.append(ep_copy)

                            # Convert matched_frame_path to base64
                            matched_frame_b64 = None
                            if result.get('matched_frame_path'):
                                matched_frame_b64 = image_path_to_base64(result['matched_frame_path'])

                            return jsonify({
                                'video_file': video_file,
                                'matched_frame': matched_frame_b64,
                                'episodes': episodes_with_images
                            })

        # Fall back to temp session data (if frames haven't been cleaned up yet)
        if session_id not in temp_data:
            logger.error(f"Session not found and no job data available: {session_id}")
            return jsonify({'error': 'Session not found and frames have been cleaned up'}), 404

        session = temp_data[session_id]
        episode_images = session['episode_images']
        matches = session['matches']
        video_frames = session['video_frames']

        if video_file not in matches:
            logger.error(f"Video file not found in matches: {video_file}")
            return jsonify({'error': 'Video file not found'}), 404

        match_info = matches[video_file]
        all_scores = match_info.get('all_scores', {})
        all_image_scores = match_info.get('all_image_scores', {})
        all_image_frame_indices = match_info.get('all_image_frame_indices', {})
        matched_frame_idx = match_info.get('matched_frame_index', 0)
        matched_frame_path = video_frames[video_file][matched_frame_idx] if video_file in video_frames else None

        # Get all video frames for this video
        all_video_frames = video_frames.get(video_file, [])

        # Build response with all episodes and their images
        episodes_data = []
        for episode_num, episode_data in sorted(episode_images.items()):
            image_paths = episode_data.get('images', [])
            score = all_scores.get(episode_num, 0.0)
            image_scores = all_image_scores.get(episode_num, [])
            image_frame_indices = all_image_frame_indices.get(episode_num, [])

            # Convert all images for this episode to base64
            episode_images_b64 = []
            matched_frames_b64 = []
            for idx, img_path in enumerate(image_paths):
                img_b64 = image_path_to_base64(img_path)
                if img_b64:
                    episode_images_b64.append(img_b64)

                    # Get the matched video frame for this specific TMDB still
                    if idx < len(image_frame_indices) and image_frame_indices[idx] < len(all_video_frames):
                        matched_frame_path = all_video_frames[image_frame_indices[idx]]
                        matched_frame_b64 = image_path_to_base64(matched_frame_path) if matched_frame_path else None
                        matched_frames_b64.append(matched_frame_b64)
                    else:
                        matched_frames_b64.append(None)

            episodes_data.append({
                'episode_number': episode_data['episode_number'],
                'episode_name': episode_data['name'],
                'images': episode_images_b64,
                'matched_frames': matched_frames_b64,  # Per-image matched frames
                'score': round(score * 100, 2),
                'image_scores': [round(s * 100, 2) for s in image_scores],
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


@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get all jobs"""
    try:
        jobs = job_manager.get_all_jobs()
        return jsonify({'jobs': jobs})
    except Exception as e:
        logger.exception(f"Error getting jobs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get a specific job by ID, converting image paths to base64 for display"""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        # Convert image paths to base64 for display
        if job.get('results') and job['results'].get('episodes'):
            for episode in job['results']['episodes']:
                # Convert matched frame and still paths to base64
                if episode.get('matched_frame_path'):
                    episode['matched_frame'] = image_path_to_base64(episode['matched_frame_path'])
                if episode.get('matched_still_path'):
                    episode['matched_still'] = image_path_to_base64(episode['matched_still_path'])

                # Convert all_episodes_data paths to base64
                if episode.get('all_episodes_data'):
                    for ep_data in episode['all_episodes_data']:
                        # Convert TMDB image paths
                        if ep_data.get('image_paths'):
                            ep_data['images'] = [image_path_to_base64(path) for path in ep_data['image_paths']]

                        # Convert matched frame paths
                        if ep_data.get('matched_frame_paths'):
                            ep_data['matched_frames'] = [
                                image_path_to_base64(path) if path else None
                                for path in ep_data['matched_frame_paths']
                            ]

        return jsonify(job)
    except Exception as e:
        logger.exception(f"Error getting job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job"""
    try:
        success = job_manager.delete_job(job_id)
        if success:
            return jsonify({'success': True, 'message': 'Job deleted'})
        else:
            return jsonify({'error': 'Job not found'}), 404
    except Exception as e:
        logger.exception(f"Error deleting job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system information including GPU and hardware acceleration status"""
    try:
        import platform
        import psutil

        # Try to import PyTorch for GPU detection
        torch_info = {}

        try:
            import torch
            torch_info = {
                'available': True,
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
            }
            if torch.cuda.is_available():
                torch_info['cuda_version'] = torch.version.cuda
                torch_info['device_count'] = torch.cuda.device_count()
                torch_info['devices'] = [
                    {
                        'index': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total_gb': round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                    }
                    for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            torch_info = {'available': False, 'error': 'PyTorch not installed'}
        except Exception as e:
            torch_info = {'available': False, 'error': str(e)}

        # Get image matcher GPU status
        gpu_acceleration = {
            'enabled': image_matcher.use_gpu,
            'backend': 'pytorch' if image_matcher.use_gpu else None
        }

        # Get video service hardware acceleration and GPU scaling status
        video_acceleration = {
            'hwaccel_type': video_service.hwaccel_type,
            'gpu_scaling_available': video_service.gpu_scaling_available
        }

        # System info
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_total_gb': round(psutil.virtual_memory().total / 1024**3, 2),
            'memory_available_gb': round(psutil.virtual_memory().available / 1024**3, 2),
        }

        # Configuration
        config_info = {
            'input_path': INPUT_PATH,
            'output_path': OUTPUT_PATH,
            'frame_height': config_service.frame_height,
            'frame_fps': config_service.frame_fps,
            'frame_start_offset': config_service.frame_start_offset,
            'frame_end_offset': config_service.frame_end_offset,
            'tmdb_api_key_configured': bool(config_service.tmdb_api_key),
            'auto_processing_enabled': config_service.auto_processing_enabled,
            'auto_processing_delay': config_service.auto_processing_delay
        }

        return jsonify({
            'system': system_info,
            'gpu_acceleration': gpu_acceleration,
            'video_acceleration': video_acceleration,
            'torch': torch_info,
            'configuration': config_info
        })

    except Exception as e:
        logger.exception(f"Error getting system info: {e}")
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


@app.route('/api/logs', methods=['DELETE'])
def clear_logs():
    """Clear application logs"""
    try:
        logger.info("Clearing application logs")

        if os.path.exists(log_file):
            # Clear the log file by opening it in write mode
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')

        logger.info("Application logs cleared")
        return jsonify({'success': True, 'message': 'Logs cleared successfully'})

    except Exception as e:
        logger.exception(f"Error clearing logs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/<path:folder_path>/<path:filename>')
def stream_video(folder_path, filename):
    """Stream video file with support for range requests"""
    try:
        # Decode the folder path and construct full path
        full_folder_path = folder_path.replace('|', os.sep)
        video_path = os.path.join(full_folder_path, filename)

        logger.info(f"Streaming video request: {video_path}")

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return jsonify({'error': 'Video file not found'}), 404

        # Determine MIME type based on file extension
        file_ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.mkv': 'video/x-matroska',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.webm': 'video/webm',
            '.m4v': 'video/x-m4v'
        }
        mime_type = mime_types.get(file_ext, 'video/x-matroska')

        # Get file size
        file_size = os.path.getsize(video_path)

        # Check if this is a range request
        range_header = request.headers.get('Range', None)

        if range_header:
            # Parse range header
            byte_range = range_header.strip().split('=')[1]
            start, end = byte_range.split('-')
            start = int(start) if start else 0
            end = int(end) if end else file_size - 1

            # Ensure valid range
            if start >= file_size or end >= file_size:
                return jsonify({'error': 'Invalid range'}), 416

            length = end - start + 1

            # Read the requested chunk
            with open(video_path, 'rb') as f:
                f.seek(start)
                chunk = f.read(length)

            # Return partial content
            response = make_response(chunk)
            response.headers['Content-Type'] = mime_type
            response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Length'] = str(length)
            response.status_code = 206  # Partial Content

            logger.info(f"Streaming partial content: {start}-{end}/{file_size} bytes")
            return response

        else:
            # Return full file
            logger.info(f"Streaming full file: {file_size} bytes, mime: {mime_type}")
            return send_file(video_path, mimetype=mime_type)

    except Exception as e:
        logger.exception(f"Error streaming video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current application settings"""
    try:
        settings = config_service.get_all()
        return jsonify({'settings': settings})
    except Exception as e:
        logger.exception(f"Error getting settings: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No settings provided'}), 400

        # Update configuration
        updated_config = config_service.update(data)

        # Reinitialize services that depend on config
        global tmdb_service, video_service, image_matcher

        tmdb_service = TMDBService(
            api_key=config_service.tmdb_api_key,
            cache_dir='/app/temp/downloads',
            frame_height=config_service.frame_height
        )
        video_service = VideoService(
            frame_height=config_service.frame_height,
            start_offset=config_service.frame_start_offset,
            end_offset=config_service.frame_end_offset,
            fps=config_service.frame_fps,
            base_temp_dir='/app/temp',
            parallel_enabled=config_service.extraction_parallel_enabled,
            parallel_workers=config_service.extraction_parallel_workers,
            output_format=config_service.extraction_output_format
        )

        # Reinitialize ImageMatcher if matching settings changed
        if any(k.startswith('matcher_') for k in data.keys()):
            image_matcher = ImageMatcher(
                max_workers=config_service.matcher_max_workers,
                early_stop_threshold=config_service.matcher_early_stop_threshold,
                max_candidates=config_service.matcher_max_candidates,
                gpu_batch_size=config_service.matcher_gpu_batch_size,
                cpu_batch_size=config_service.matcher_cpu_batch_size
            )
            logger.info("ImageMatcher reinitialized with new settings")

        # Restart folder watcher if auto-processing settings changed
        if 'auto_processing_enabled' in data or 'auto_processing_delay' in data:
            if config_service.auto_processing_enabled:
                folder_watcher.stop()
                folder_watcher.start()
                logger.info("Folder watcher restarted with new settings")
            else:
                folder_watcher.stop()
                logger.info("Folder watcher stopped (auto-processing disabled)")

        logger.info(f"Settings updated: {list(data.keys())}")

        return jsonify({
            'success': True,
            'settings': updated_config,
            'message': 'Settings updated successfully'
        })

    except Exception as e:
        logger.exception(f"Error updating settings: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)
