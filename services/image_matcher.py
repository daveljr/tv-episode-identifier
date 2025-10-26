import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import imagehash
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyTorch for GPU acceleration
TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = None
TORCH_CUDA_ERROR = None
try:
    import torch
    import torchvision.transforms.functional as TF
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
    else:
        TORCH_CUDA_ERROR = "PyTorch imported successfully but CUDA not available (torch.cuda.is_available() returned False)"
except ImportError as e:
    TORCH_IMPORT_ERROR = str(e)


class ImageMatcher:
    """Service for matching video frames to episode stills"""

    def __init__(self, max_workers=None, early_stop_threshold=0.95,
                 max_candidates=5000, gpu_batch_size=500, cpu_batch_size=50):
        """
        Initialize ImageMatcher with parallel processing support

        Args:
            max_workers: Number of parallel workers (default: CPU count)
            early_stop_threshold: Stop searching when this confidence is reached (default: 0.95)
            max_candidates: Maximum number of candidate frames to compare per episode still (default: 5000)
                           Higher values = more thorough but slower. With GPU, 5000+ is reasonable.
                           With CPU, 200-500 is more practical.
            gpu_batch_size: Number of frames to process in a single GPU batch (default: 500)
                           Larger batches = better GPU utilization but more memory
            cpu_batch_size: Number of frames to process in a single CPU batch (default: 50)
        """
        self._episode_still_cache = {}  # Cache preprocessed episode stills
        self.max_workers = max_workers  # None = use CPU count
        self.early_stop_threshold = early_stop_threshold
        self.max_candidates = max_candidates
        self.gpu_batch_size = gpu_batch_size
        self.cpu_batch_size = cpu_batch_size

        # GPU configuration - automatically use GPU if PyTorch with CUDA is available
        self.use_gpu = TORCH_AVAILABLE
        self.device = None
        self.gpu_lock = threading.Lock()  # Lock for GPU operations to prevent conflicts

        # Detailed GPU detection logging
        logger.info("=" * 60)
        logger.info("GPU Detection Report:")
        logger.info("-" * 60)

        # PyTorch status
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"✓ PyTorch with CUDA: AVAILABLE")
            logger.info(f"  └─ CUDA Version: {torch.version.cuda}")
            logger.info(f"  └─ GPU Count: {gpu_count}")
            logger.info(f"  └─ GPU 0: {gpu_name}")
            logger.info(f"")
            logger.info(f"🚀 GPU ACCELERATION ENABLED")
            logger.info(f"   Device: {gpu_name}")
            logger.info(f"   Expected speedup: 6-8x faster than CPU")
        elif TORCH_IMPORT_ERROR:
            logger.info(f"✗ PyTorch: NOT INSTALLED")
            logger.info(f"  └─ Error: {TORCH_IMPORT_ERROR}")
        elif TORCH_CUDA_ERROR:
            logger.info(f"✗ PyTorch: INSTALLED (CPU-only)")
            logger.info(f"  └─ Reason: {TORCH_CUDA_ERROR}")

        if not self.use_gpu:
            logger.info(f"")
            logger.info(f"⚠️  GPU ACCELERATION DISABLED - USING CPU MODE")
            logger.info(f"   Reason: PyTorch with CUDA not available")
            logger.info(f"   To enable GPU acceleration:")
            logger.info(f"   1. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            logger.info(f"   2. Restart the application")

        logger.info("=" * 60)

        logger.info(f"ImageMatcher initialized:")
        logger.info(f"  max_workers={max_workers}")
        logger.info(f"  early_stop_threshold={early_stop_threshold}")
        logger.info(f"  max_candidates={max_candidates}")
        logger.info(f"  gpu_batch_size={gpu_batch_size}")
        logger.info(f"  cpu_batch_size={cpu_batch_size}")
        logger.info(f"  use_gpu={self.use_gpu}")

    def match_videos_to_episodes(self, video_frames, episode_images):
        """
        Match videos to episodes based on frame similarity
        Now supports parallel processing of multiple videos

        Args:
            video_frames: dict of {video_filename: [frame_paths]} - list of file paths to video frames
            episode_images: dict of {episode_number: {'images': [image_paths], 'name': str, 'episode_number': int}}

        Returns:
            dict of {video_filename: {'episode_number': int, 'episode_name': str, 'confidence': float,
                                      'matched_frame_index': int, 'best_match_episode': int,
                                      'all_scores': dict}}
        """
        matches = {}

        logger.info(f"Matching {len(video_frames)} videos to {len(episode_images)} episodes")

        # Preprocess all episode stills once (cache them with perceptual hashes)
        # Now handles multiple images per episode
        preprocessed_stills = {}
        episode_hashes = {}
        logger.info("Preprocessing episode stills and computing perceptual hashes...")
        for episode_num, episode_data in episode_images.items():
            # Process all images for this episode
            images = episode_data.get('images', [episode_data.get('image')])  # Support old and new format
            if not isinstance(images, list):
                images = [images]

            preprocessed_stills[episode_num] = [self._preprocess_image(img) for img in images]
            # Compute perceptual hashes for all images
            episode_hashes[episode_num] = [imagehash.dhash(self._load_image(img), hash_size=8) for img in images]

        # Process videos in parallel using ThreadPoolExecutor
        # This allows multiple videos to be matched simultaneously
        if len(video_frames) > 1:
            logger.info(f"Processing {len(video_frames)} videos in parallel...")
            with ThreadPoolExecutor(max_workers=min(len(video_frames), 8)) as executor:
                # Submit all video matching tasks
                futures = {}
                for video_file, frames in video_frames.items():
                    future = executor.submit(
                        self._match_single_video,
                        video_file, frames, episode_images, preprocessed_stills, episode_hashes
                    )
                    futures[future] = video_file

                # Collect results as they complete
                for future in as_completed(futures):
                    video_file = futures[future]
                    try:
                        result = future.result()
                        matches[video_file] = result
                        logger.info(f"{video_file} matched to Episode {result['episode_number']}: {result['episode_name']} (confidence: {result['confidence']:.2%})")
                    except Exception as e:
                        logger.exception(f"Error matching {video_file}: {e}")
                        matches[video_file] = {
                            'episode_number': 0,
                            'episode_name': 'Unknown',
                            'confidence': 0.0,
                            'matched_frame_index': 0,
                            'best_match_episode': 0,
                            'all_scores': {}
                        }
        else:
            # Single video - process directly
            for video_file, frames in video_frames.items():
                result = self._match_single_video(
                    video_file, frames, episode_images, preprocessed_stills, episode_hashes
                )
                matches[video_file] = result
                logger.info(f"{video_file} matched to Episode {result['episode_number']}: {result['episode_name']} (confidence: {result['confidence']:.2%})")

        logger.info(f"Matching complete: {len(matches)} videos processed")
        return matches

    def _match_single_video(self, video_file, frames, episode_images, preprocessed_stills, episode_hashes):
        """
        Match a single video to episodes
        Extracted to support parallel processing of multiple videos

        Returns:
            dict: Match result for this video
        """
        logger.info(f"Processing video: {video_file} with {len(frames)} frames")

        # Use parallel processing for large frame counts
        if len(frames) > 50:
            result = self._match_video_parallel(
                video_file, frames, episode_images, preprocessed_stills, episode_hashes
            )
        else:
            result = self._match_video_sequential(
                video_file, frames, episode_images, preprocessed_stills, episode_hashes
            )

        best_match = result['best_match']
        best_score = result['best_score']
        best_frame_index = result['best_frame_index']
        all_episode_scores = result['all_episode_scores']
        all_image_scores = result.get('all_image_scores', {})
        all_image_frame_indices = result.get('all_image_frame_indices', {})

        if best_match:
            return {
                'episode_number': int(best_match['episode_number']),
                'episode_name': best_match['name'],
                'confidence': float(best_score),
                'matched_frame_index': int(best_frame_index),
                'best_match_episode': int(best_match['episode_number']),
                'all_scores': all_episode_scores,
                'all_image_scores': all_image_scores,
                'all_image_frame_indices': all_image_frame_indices
            }
        else:
            logger.warning(f"{video_file} - No match found")
            return {
                'episode_number': 0,
                'episode_name': 'Unknown',
                'confidence': 0.0,
                'matched_frame_index': 0,
                'best_match_episode': 0,
                'all_scores': {},
                'all_image_scores': {},
                'all_image_frame_indices': {}
            }

    def _match_video_sequential(self, video_file, frame_paths, episode_images, preprocessed_stills, episode_hashes):
        """
        Sequential matching for smaller frame counts (< 50 frames)
        Uses perceptual hash filtering and early stopping

        Args:
            frame_paths: List of file paths to video frames (strings)
        """
        best_match = None
        best_score = 0
        best_frame_index = 0
        all_episode_scores = {}
        all_image_scores = {}  # Dict of {episode_num: [score1, score2, ...]}
        all_image_frame_indices = {}  # Dict of {episode_num: [frame_idx1, frame_idx2, ...]}

        # First pass: Use perceptual hashing to filter candidate frames (fast)
        logger.debug(f"Computing perceptual hashes for {len(frame_paths)} frames...")
        frame_hashes = []
        for frame_path in frame_paths:
            # Load image and compute hash
            with Image.open(frame_path) as img:
                frame_hash = imagehash.dhash(img, hash_size=8)
                frame_hashes.append(frame_hash)

        # Compare each episode (now with multiple images per episode)
        for episode_num, episode_data in episode_images.items():
            episode_hash_list = episode_hashes[episode_num]
            preprocessed_still_list = preprocessed_stills[episode_num]

            # For each episode still, find the best matching frame
            episode_scores_per_still = []

            for episode_hash, preprocessed_still in zip(episode_hash_list, preprocessed_still_list):
                # Find top candidate frames using hash distance (very fast)
                hash_distances = [(abs(fh - episode_hash), idx) for idx, fh in enumerate(frame_hashes)]
                hash_distances.sort()

                # Use configurable max_candidates (default: 5000)
                num_candidates = min(self.max_candidates, len(frame_paths))
                candidate_frame_paths = [frame_paths[idx] for dist, idx in hash_distances[:num_candidates]]
                candidate_indices = [idx for dist, idx in hash_distances[:num_candidates]]

                # Detailed comparison on candidate frames only
                scores = []
                for frame_path, frame_idx in zip(candidate_frame_paths, candidate_indices):
                    # Load frame image on-demand
                    with Image.open(frame_path) as frame_img:
                        score = self._compare_images(frame_img, preprocessed_still)
                        scores.append((score, frame_idx))

                        # Early stopping if we found a near-perfect match
                        if score >= self.early_stop_threshold:
                            logger.info(f"Early stop: Found {score:.2%} match for episode {episode_num}")
                            break

                max_score, max_frame_idx = max(scores, key=lambda x: x[0]) if scores else (0, 0)
                episode_scores_per_still.append((max_score, max_frame_idx))

            # Average the scores across all stills for this episode
            # Use weighted average that prioritizes high-confidence matches (80%+)
            if episode_scores_per_still:
                # Calculate weighted average
                # Matches >= 80% get weight of 2.0
                # Matches < 80% get weight of 1.0
                weighted_sum = 0
                total_weight = 0
                for score, _ in episode_scores_per_still:
                    weight = 2.0 if score >= 0.80 else 1.0
                    weighted_sum += score * weight
                    total_weight += weight

                avg_score = weighted_sum / total_weight if total_weight > 0 else 0
                # Use the frame that had the highest individual score
                best_still_score, best_still_frame_idx = max(episode_scores_per_still, key=lambda x: x[0])
                logger.debug(f"Episode {episode_num}: Averaged {len(episode_scores_per_still)} stills, weighted_avg={avg_score:.2%}, best={best_still_score:.2%}")
            else:
                avg_score = 0
                best_still_frame_idx = 0

            all_episode_scores[episode_num] = float(avg_score)
            # Store individual image scores for this episode
            all_image_scores[episode_num] = [float(score) for score, _ in episode_scores_per_still]
            # Store the frame indices that matched each TMDB still
            all_image_frame_indices[episode_num] = [int(frame_idx) for _, frame_idx in episode_scores_per_still]

            if avg_score > best_score:
                best_score = avg_score
                best_match = episode_data
                best_frame_index = best_still_frame_idx

        return {
            'best_match': best_match,
            'best_score': best_score,
            'best_frame_index': best_frame_index,
            'all_episode_scores': all_episode_scores,
            'all_image_scores': all_image_scores,
            'all_image_frame_indices': all_image_frame_indices
        }

    def _match_video_parallel(self, video_file, frame_paths, episode_images, preprocessed_stills, episode_hashes):
        """
        Parallel matching for large frame counts (50+ frames)
        Uses ThreadPoolExecutor for parallel processing

        Args:
            frame_paths: List of file paths to video frames (strings)

        Note: We use threads instead of processes because:
        - File paths are simple strings (no pickling issues)
        - Image comparison is I/O bound (disk reads) and numpy-bound (releases GIL)
        - Threads work well for this use case
        """
        logger.info(f"Using parallel threading for {len(frame_paths)} frames...")

        best_match = None
        best_score = 0
        best_frame_index = 0
        all_episode_scores = {}
        all_image_scores = {}  # Dict of {episode_num: [score1, score2, ...]}
        all_image_frame_indices = {}  # Dict of {episode_num: [frame_idx1, frame_idx2, ...]}

        # OPTIMIZATION: Compute hashes AND preprocess frames in one pass
        # This avoids loading each frame twice (once for hash, once for comparison)
        logger.info("Computing perceptual hashes and preprocessing frames (optimized single-pass)...")
        frame_hashes = []
        frame_preprocessed = {}  # Cache: {frame_idx: (arr, gray)}

        hash_batch_size = 1000  # Process 1000 frames at a time
        for i in range(0, len(frame_paths), hash_batch_size):
            batch_paths = frame_paths[i:i+hash_batch_size]

            for j, frame_path in enumerate(batch_paths):
                frame_idx = i + j
                try:
                    with Image.open(frame_path) as img:
                        # Compute hash
                        frame_hash = imagehash.dhash(img, hash_size=8)
                        frame_hashes.append(frame_hash)

                        # Preprocess and cache (reuse img that's already loaded)
                        arr, gray = self._preprocess_image(img)
                        frame_preprocessed[frame_idx] = (arr, gray)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_path}: {e}")
                    frame_hashes.append(imagehash.hex_to_hash('0' * 16))  # Dummy hash
                    frame_preprocessed[frame_idx] = None

            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(frame_paths)} frames...")

        # Process each episode (now with multiple images per episode)
        for episode_num, episode_data in episode_images.items():
            episode_hash_list = episode_hashes[episode_num]
            preprocessed_still_list = preprocessed_stills[episode_num]

            # Compare against all stills for this episode
            episode_scores_per_still = []

            for episode_hash, preprocessed_still in zip(episode_hash_list, preprocessed_still_list):
                # Filter frames using perceptual hashing
                hash_distances = [(abs(fh - episode_hash), idx) for idx, fh in enumerate(frame_hashes)]
                hash_distances.sort()

                # Use configurable max_candidates (default: 5000 for GPU, smaller for CPU)
                num_candidates = min(self.max_candidates, len(frame_paths))

                candidate_frame_paths = [frame_paths[idx] for dist, idx in hash_distances[:num_candidates]]
                candidate_indices = [idx for dist, idx in hash_distances[:num_candidates]]

                logger.info(f"Episode {episode_num}: Comparing {num_candidates} candidate frames (filtered from {len(frame_paths)})")

                # Use appropriate batch size based on GPU/CPU mode
                batch_size = self.gpu_batch_size if self.use_gpu else self.cpu_batch_size
                max_score = 0
                max_frame_idx = 0

                for i in range(0, len(candidate_frame_paths), batch_size):
                    batch_frame_paths = candidate_frame_paths[i:i+batch_size]
                    batch_indices = candidate_indices[i:i+batch_size]

                    # OPTIMIZATION: Use cached preprocessed frames instead of reloading
                    batch_preprocessed = []
                    for idx in batch_indices:
                        preprocessed = frame_preprocessed.get(idx)
                        if preprocessed is None:
                            logger.warning(f"Frame {idx} not in cache, skipping")
                        batch_preprocessed.append(preprocessed)

                    # GPU optimization: Process entire batch on GPU at once
                    if self.use_gpu:
                        results = self._compare_batch_gpu_preprocessed(batch_preprocessed, preprocessed_still)
                    else:
                        # CPU: Parallel processing for this batch using threads
                        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                            # Process frames in parallel using cached preprocessed data
                            futures = []
                            for preprocessed_frame in batch_preprocessed:
                                if preprocessed_frame is None:
                                    futures.append(executor.submit(lambda: 0.0))
                                else:
                                    future = executor.submit(self._compare_preprocessed, preprocessed_frame, preprocessed_still)
                                    futures.append(future)

                            # Collect results
                            results = [future.result() for future in futures]

                    # Find best in this batch
                    for score, frame_idx in zip(results, batch_indices):
                        if score > max_score:
                            max_score = score
                            max_frame_idx = frame_idx

                        # Early stopping if we found excellent match
                        if max_score >= self.early_stop_threshold:
                            logger.info(f"Early stop: Found {max_score:.2%} match for episode {episode_num}")
                            break

                    if max_score >= self.early_stop_threshold:
                        break

                episode_scores_per_still.append((max_score, max_frame_idx))

            # Average the scores across all stills for this episode
            # Use weighted average that prioritizes high-confidence matches (80%+)
            if episode_scores_per_still:
                # Calculate weighted average
                # Matches >= 80% get weight of 2.0
                # Matches < 80% get weight of 1.0
                weighted_sum = 0
                total_weight = 0
                for score, _ in episode_scores_per_still:
                    weight = 2.0 if score >= 0.80 else 1.0
                    weighted_sum += score * weight
                    total_weight += weight

                avg_score = weighted_sum / total_weight if total_weight > 0 else 0
                # Use the frame that had the highest individual score
                best_still_score, best_still_frame_idx = max(episode_scores_per_still, key=lambda x: x[0])
                logger.debug(f"Episode {episode_num}: Averaged {len(episode_scores_per_still)} stills, weighted_avg={avg_score:.2%}, best={best_still_score:.2%}")
            else:
                avg_score = 0
                best_still_frame_idx = 0

            all_episode_scores[episode_num] = float(avg_score)
            # Store individual image scores for this episode
            all_image_scores[episode_num] = [float(score) for score, _ in episode_scores_per_still]
            # Store the frame indices that matched each TMDB still
            all_image_frame_indices[episode_num] = [int(frame_idx) for _, frame_idx in episode_scores_per_still]

            if avg_score > best_score:
                best_score = avg_score
                best_match = episode_data
                best_frame_index = best_still_frame_idx

        return {
            'best_match': best_match,
            'best_score': best_score,
            'best_frame_index': best_frame_index,
            'all_episode_scores': all_episode_scores,
            'all_image_scores': all_image_scores,
            'all_image_frame_indices': all_image_frame_indices
        }

    def _load_image(self, img_path):
        """
        Load image from file path
        """
        if isinstance(img_path, (str, Path)):
            img = Image.open(img_path)
        else:
            img = img_path  # Assume already a PIL Image

        return img
        

    def _preprocess_image(self, img):
        """
        Preprocess image for faster comparison
        Supports both PIL Image objects and file paths (str/Path)
        Returns tuple of (resized_array, gray_array)
        """
        # Load image if it's a file path
        if isinstance(img, (str, Path)):
            img = Image.open(img)

        # Use smaller size for faster processing
        target_size = (320, 180)  # Reduced from 640x360 for 4x speed boost
        img_resized = img.resize(target_size, Image.Resampling.BILINEAR)  # BILINEAR is faster than LANCZOS

        arr = np.array(img_resized)

        # Convert to grayscale
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        return arr, gray

    def _compare_image_from_path(self, img1_path, img2_preprocessed):
        """
        Compare an image from a file path with a preprocessed image

        Args:
            img1_path: File path to image (string or Path)
            img2_preprocessed: Tuple of (arr, gray) from _preprocess_image

        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            with Image.open(img1_path) as img1:
                return self._compare_images(img1, img2_preprocessed)
        except Exception as e:
            logger.error(f"Error loading image from {img1_path}: {e}")
            return 0.0

    def _compare_images(self, img1, img2_preprocessed):
        """
        Compare two images and return a similarity score between 0 and 1
        Uses SSIM for structural similarity (optimized version)

        Args:
            img1: PIL Image (video frame)
            img2_preprocessed: Tuple of (arr, gray) from _preprocess_image
        """
        try:
            # Use GPU-accelerated comparison if available
            if self.use_gpu:
                return self._compare_images_gpu_torch(img1, img2_preprocessed)

            # Fallback to CPU comparison
            arr1, gray1 = self._preprocess_image(img1)
            arr2, gray2 = img2_preprocessed

            # Calculate structural similarity (this is the most important metric)
            ssim_score = ssim(gray1, gray2, data_range=255)

            # Only calculate histogram for close matches (optimization)
            if ssim_score > 0.5:
                hist_score = self._compare_histograms_fast(arr1, arr2)
                # Combine scores (weighted average)
                combined_score = (ssim_score * 0.7) + (hist_score * 0.3)
            else:
                # Skip histogram calculation for poor matches
                combined_score = ssim_score * 0.7

            return max(0, combined_score)  # Ensure non-negative

        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            return 0.0

    def _compare_preprocessed(self, frame_preprocessed, episode_still_preprocessed):
        """
        Compare two preprocessed images (CPU mode)

        Args:
            frame_preprocessed: Tuple of (arr, gray) from _preprocess_image
            episode_still_preprocessed: Tuple of (arr, gray) from _preprocess_image

        Returns:
            float: Similarity score
        """
        try:
            arr1, gray1 = frame_preprocessed
            arr2, gray2 = episode_still_preprocessed

            # Calculate structural similarity
            ssim_score = ssim(gray1, gray2, data_range=255)

            # Only calculate histogram for close matches
            if ssim_score > 0.5:
                hist_score = self._compare_histograms_fast(arr1, arr2)
                combined_score = (ssim_score * 0.7) + (hist_score * 0.3)
            else:
                combined_score = ssim_score * 0.7

            return max(0, combined_score)

        except Exception as e:
            logger.error(f"Error comparing preprocessed images: {e}")
            return 0.0

    def _compare_batch_gpu_preprocessed(self, batch_preprocessed, episode_still_preprocessed):
        """
        GPU-optimized TRUE batch comparison - uses vectorized tensor operations
        Much faster than processing one-by-one

        Args:
            batch_preprocessed: List of tuples [(arr, gray), ...] from _preprocess_image
            episode_still_preprocessed: Tuple of (arr, gray) from _preprocess_image

        Returns:
            list of float: Similarity scores for each frame
        """
        try:
            arr2, gray2 = episode_still_preprocessed

            # Filter out None entries
            valid_frames = [(i, fp) for i, fp in enumerate(batch_preprocessed) if fp is not None]
            if not valid_frames:
                return [0.0] * len(batch_preprocessed)

            batch_results = [0.0] * len(batch_preprocessed)

            # ADVANCED OPTIMIZATION: Stack all grayscale frames into a single tensor for vectorized SSIM
            # This processes all frames simultaneously on GPU instead of one-by-one
            valid_grays = []
            valid_arrs = []
            valid_indices = []

            for idx, frame_data in valid_frames:
                arr1, gray1 = frame_data
                valid_grays.append(gray1)
                valid_arrs.append(arr1)
                valid_indices.append(idx)

            if not valid_grays:
                return batch_results

            # Single GPU lock for entire batch
            with self.gpu_lock:
                # Stack all grayscale frames into a batch tensor [batch_size, height, width]
                batch_gray_np = np.stack(valid_grays, axis=0)
                batch_gray_t = torch.from_numpy(batch_gray_np).float().to(self.device)

                # Episode still as tensor
                gray2_t = torch.from_numpy(gray2).float().to(self.device)

                # Vectorized SSIM computation for entire batch
                ssim_scores = self._ssim_torch_batch(batch_gray_t, gray2_t)

                # Process results
                for i, (idx, ssim_score) in enumerate(zip(valid_indices, ssim_scores)):
                    # Only calculate histogram for close matches
                    if ssim_score > 0.5:
                        arr1_t = torch.from_numpy(valid_arrs[i]).to(self.device)
                        arr2_t = torch.from_numpy(arr2).to(self.device)
                        hist_score = self._compare_histograms_gpu_torch(arr1_t, arr2_t)
                        combined_score = (ssim_score * 0.7) + (hist_score * 0.3)
                        del arr1_t, arr2_t
                    else:
                        combined_score = ssim_score * 0.7

                    batch_results[idx] = max(0, combined_score)

                # Clean up tensors
                del batch_gray_t, gray2_t

            return batch_results

        except Exception as e:
            logger.error(f"Error in batch GPU comparison (preprocessed): {e}, falling back")
            # Fallback to CPU comparison
            return [self._compare_preprocessed(fp, episode_still_preprocessed) if fp else 0.0
                    for fp in batch_preprocessed]

    def _compare_batch_gpu(self, frame_paths, episode_still_preprocessed):
        """
        GPU-optimized batch comparison - processes entire batch on GPU at once
        Much faster than individual comparisons due to reduced GPU lock contention

        Args:
            frame_paths: List of file paths to video frames
            episode_still_preprocessed: Preprocessed episode still

        Returns:
            list of float: Similarity scores for each frame
        """
        try:
            # Load all frames at once
            frames_data = []
            for frame_path in frame_paths:
                try:
                    with Image.open(frame_path) as img:
                        arr, gray = self._preprocess_image(img)
                        frames_data.append((arr, gray))
                except Exception as e:
                    logger.error(f"Error loading {frame_path}: {e}")
                    frames_data.append(None)

            arr2, gray2 = episode_still_preprocessed
            results = []

            # Single GPU lock for entire batch
            with self.gpu_lock:
                # Process each frame with GPU
                for frame_data in frames_data:
                    if frame_data is None:
                        results.append(0.0)
                        continue

                    arr1, gray1 = frame_data

                    # Convert to torch tensors and move to GPU
                    gray1_t = torch.from_numpy(gray1).float().to(self.device)
                    gray2_t = torch.from_numpy(gray2).float().to(self.device)

                    # Calculate SSIM using GPU
                    ssim_score = self._ssim_torch(gray1_t, gray2_t)

                    # Only calculate histogram for close matches
                    if ssim_score > 0.5:
                        arr1_t = torch.from_numpy(arr1).to(self.device)
                        arr2_t = torch.from_numpy(arr2).to(self.device)
                        hist_score = self._compare_histograms_gpu_torch(arr1_t, arr2_t)
                        combined_score = (ssim_score * 0.7) + (hist_score * 0.3)
                        del arr1_t, arr2_t
                    else:
                        combined_score = ssim_score * 0.7

                    results.append(max(0, combined_score))

                    # Clean up
                    del gray1_t, gray2_t

            return results

        except Exception as e:
            logger.error(f"Error in batch GPU comparison: {e}, falling back to individual comparison")
            # Fallback to individual comparison
            return [self._compare_image_from_path(fp, episode_still_preprocessed) for fp in frame_paths]

    def _compare_images_gpu_torch(self, img1, img2_preprocessed):
        """
        GPU-accelerated image comparison using PyTorch
        Significantly faster for batch operations
        Uses lock to prevent concurrent GPU access conflicts
        """
        try:
            # Preprocess the video frame
            arr1, gray1 = self._preprocess_image(img1)
            arr2, gray2 = img2_preprocessed

            # Use lock for GPU operations when multiple threads are active
            with self.gpu_lock:
                # Convert to torch tensors and move to GPU
                gray1_t = torch.from_numpy(gray1).float().to(self.device)
                gray2_t = torch.from_numpy(gray2).float().to(self.device)

                # Calculate SSIM using GPU (faster for large batches)
                ssim_score = self._ssim_torch(gray1_t, gray2_t)

                # Only calculate histogram for close matches
                if ssim_score > 0.5:
                    # Move color images to GPU for histogram calculation
                    arr1_t = torch.from_numpy(arr1).to(self.device)
                    arr2_t = torch.from_numpy(arr2).to(self.device)
                    hist_score = self._compare_histograms_gpu_torch(arr1_t, arr2_t)
                    combined_score = (ssim_score * 0.7) + (hist_score * 0.3)
                else:
                    combined_score = ssim_score * 0.7

                # Clean up GPU memory
                del gray1_t, gray2_t
                if ssim_score > 0.5:
                    del arr1_t, arr2_t

            return max(0, combined_score)

        except Exception as e:
            logger.error(f"Error in GPU comparison (torch): {e}, falling back to CPU")
            # Fallback to CPU
            arr1, gray1 = self._preprocess_image(img1)
            arr2, gray2 = img2_preprocessed
            ssim_score = ssim(gray1, gray2, data_range=255)
            if ssim_score > 0.5:
                hist_score = self._compare_histograms_fast(arr1, arr2)
                return max(0, (ssim_score * 0.7) + (hist_score * 0.3))
            return max(0, ssim_score * 0.7)

    def _ssim_torch(self, img1, img2):
        """
        Fast SSIM calculation using PyTorch on GPU
        Based on the Wang et al. SSIM formula
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = img1.mean()
        mu2 = img2.mean()
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        ssim_score = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                     ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(ssim_score.cpu().item())

    def _ssim_torch_batch(self, batch_imgs, target_img):
        """
        Vectorized batch SSIM calculation using PyTorch on GPU
        Processes multiple images simultaneously for maximum GPU utilization

        Args:
            batch_imgs: Tensor of shape [batch_size, height, width]
            target_img: Tensor of shape [height, width]

        Returns:
            list of float: SSIM scores for each image in batch
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Compute statistics for target image once
        mu2 = target_img.mean()
        mu2_sq = mu2 ** 2
        sigma2_sq = ((target_img - mu2) ** 2).mean()

        # Compute statistics for all batch images simultaneously
        # Use dim=(1,2) to compute mean across height and width for each image
        mu1 = batch_imgs.mean(dim=(1, 2))  # Shape: [batch_size]

        # Expand target stats to match batch
        mu2_expanded = mu2.expand(mu1.shape[0])  # Shape: [batch_size]

        # Vectorized computations
        mu1_sq = mu1 ** 2
        mu1_mu2 = mu1 * mu2_expanded

        # Compute variance and covariance for each image in batch
        # Center the images
        centered_batch = batch_imgs - mu1.view(-1, 1, 1)
        centered_target = target_img - mu2

        # Variance of batch images
        sigma1_sq = (centered_batch ** 2).mean(dim=(1, 2))

        # Covariance between each batch image and target
        sigma12 = (centered_batch * centered_target.unsqueeze(0)).mean(dim=(1, 2))

        # Vectorized SSIM calculation for all images at once
        sigma2_sq_expanded = torch.full_like(sigma1_sq, sigma2_sq.item())

        ssim_scores = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq_expanded + C2))

        # Convert to Python list
        return ssim_scores.cpu().tolist()

    def _compare_histograms_gpu_torch(self, arr1, arr2):
        """
        GPU-accelerated histogram comparison using PyTorch
        """
        try:
            scores = []
            for channel in range(3):
                # Calculate histograms on GPU
                hist1 = torch.histc(arr1[:, :, channel].float(), bins=64, min=0, max=255)
                hist2 = torch.histc(arr2[:, :, channel].float(), bins=64, min=0, max=255)

                # Normalize
                hist1 = hist1 / hist1.sum()
                hist2 = hist2 / hist2.sum()

                # Correlation
                mean1 = hist1.mean()
                mean2 = hist2.mean()
                std1 = hist1.std()
                std2 = hist2.std()

                if std1 > 0 and std2 > 0:
                    correlation = ((hist1 - mean1) * (hist2 - mean2)).sum() / (std1 * std2 * len(hist1))
                    scores.append(float(correlation.cpu().item()))
                else:
                    scores.append(0.0)

            return np.mean(scores)

        except Exception as e:
            logger.error(f"Error in GPU histogram comparison: {e}")
            return 0.0

    def _compare_histograms_fast(self, arr1, arr2):
        """
        Fast histogram comparison using reduced bins
        """
        try:
            # Use fewer bins for faster computation (64 instead of 256)
            hist1 = [cv2.calcHist([arr1], [i], None, [64], [0, 256]) for i in range(3)]
            hist2 = [cv2.calcHist([arr2], [i], None, [64], [0, 256]) for i in range(3)]

            # Normalize and compare in one step
            scores = []
            for h1, h2 in zip(hist1, hist2):
                h1 = cv2.normalize(h1, h1).flatten()
                h2 = cv2.normalize(h2, h2).flatten()
                score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                scores.append(score)

            return np.mean(scores)

        except Exception as e:
            logger.error(f"Error comparing histograms: {e}")
            return 0.0
