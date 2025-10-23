import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import imagehash
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import GPU libraries (optional)
# Store import results and exceptions for detailed logging later
CUPY_AVAILABLE = False
CUPY_IMPORT_ERROR = None
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError as e:
    CUPY_IMPORT_ERROR = str(e)

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

    def __init__(self, max_workers=None, early_stop_threshold=0.95):
        """
        Initialize ImageMatcher with parallel processing support

        Args:
            max_workers: Number of parallel workers (default: CPU count)
            early_stop_threshold: Stop searching when this confidence is reached (default: 0.95)
        """
        self._episode_still_cache = {}  # Cache preprocessed episode stills
        self.max_workers = max_workers  # None = use CPU count
        self.early_stop_threshold = early_stop_threshold

        # GPU configuration - automatically use GPU if available
        self.use_gpu = CUPY_AVAILABLE or TORCH_AVAILABLE
        self.gpu_backend = None

        # Detailed GPU detection logging
        logger.info("=" * 60)
        logger.info("GPU Detection Report:")
        logger.info("-" * 60)

        # PyTorch status
        if TORCH_AVAILABLE:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"✓ PyTorch: AVAILABLE")
            logger.info(f"  └─ CUDA Version: {torch.version.cuda}")
            logger.info(f"  └─ GPU Count: {gpu_count}")
            logger.info(f"  └─ GPU 0: {gpu_name}")
        elif TORCH_IMPORT_ERROR:
            logger.info(f"✗ PyTorch: NOT INSTALLED")
            logger.info(f"  └─ Error: {TORCH_IMPORT_ERROR}")
        elif TORCH_CUDA_ERROR:
            logger.info(f"✗ PyTorch: INSTALLED (CPU-only)")
            logger.info(f"  └─ Reason: {TORCH_CUDA_ERROR}")

        # CuPy status
        if CUPY_AVAILABLE:
            logger.info(f"✓ CuPy: AVAILABLE")
        elif CUPY_IMPORT_ERROR:
            logger.info(f"✗ CuPy: NOT INSTALLED")
            logger.info(f"  └─ Error: {CUPY_IMPORT_ERROR}")

        logger.info("-" * 60)

        # Final GPU configuration
        if self.use_gpu:
            if TORCH_AVAILABLE:
                self.gpu_backend = 'torch'
                self.device = torch.device('cuda:0')
                logger.info(f"🚀 GPU ACCELERATION ENABLED")
                logger.info(f"   Backend: PyTorch")
                logger.info(f"   Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"   Expected speedup: 2-8x faster")
            elif CUPY_AVAILABLE:
                self.gpu_backend = 'cupy'
                logger.info(f"🚀 GPU ACCELERATION ENABLED")
                logger.info(f"   Backend: CuPy")
                logger.info(f"   Expected speedup: 2-8x faster")
        else:
            if not use_gpu:
                logger.info(f"⚠️  GPU ACCELERATION DISABLED")
                logger.info(f"   Reason: GPU_MATCHING_ENABLED=false in configuration")
            else:
                logger.info(f"⚠️  GPU ACCELERATION DISABLED - USING CPU MODE")
                logger.info(f"   Reason: No GPU libraries available")
                logger.info(f"   To enable GPU acceleration:")
                logger.info(f"   1. Install PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                logger.info(f"   2. Or install CuPy: pip install cupy-cuda11x")
                logger.info(f"   3. Restart the application")

        logger.info("=" * 60)

        logger.info(f"ImageMatcher initialized: max_workers={max_workers}, early_stop_threshold={early_stop_threshold}, use_gpu={self.use_gpu}, backend={self.gpu_backend}")

    def match_videos_to_episodes(self, video_frames, episode_images):
        """
        Match videos to episodes based on frame similarity

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

        for video_file, frames in video_frames.items():
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

            if best_match:
                logger.info(f"{video_file} matched to Episode {best_match['episode_number']}: {best_match['name']} (confidence: {best_score:.2%})")
                matches[video_file] = {
                    'episode_number': int(best_match['episode_number']),
                    'episode_name': best_match['name'],
                    'confidence': float(best_score),
                    'matched_frame_index': int(best_frame_index),
                    'best_match_episode': int(best_match['episode_number']),
                    'all_scores': all_episode_scores
                }
            else:
                logger.warning(f"{video_file} - No match found")
                # No match found, assign based on filename order as fallback
                matches[video_file] = {
                    'episode_number': 0,
                    'episode_name': 'Unknown',
                    'confidence': 0.0,
                    'matched_frame_index': 0,
                    'best_match_episode': 0,
                    'all_scores': {}
                }

        logger.info(f"Matching complete: {len(matches)} videos processed")
        return matches

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

                # Only compare top 20% of frames based on hash similarity (or min 10 frames)
                num_candidates = max(10, len(frame_paths) // 5)
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

            if avg_score > best_score:
                best_score = avg_score
                best_match = episode_data
                best_frame_index = best_still_frame_idx

        return {
            'best_match': best_match,
            'best_score': best_score,
            'best_frame_index': best_frame_index,
            'all_episode_scores': all_episode_scores
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

        # Compute perceptual hashes for frames in batches to avoid loading all into memory
        logger.debug("Computing perceptual hashes for frames in batches...")
        frame_hashes = []
        hash_batch_size = 1000  # Compute 1000 hashes at a time
        for i in range(0, len(frame_paths), hash_batch_size):
            batch_paths = frame_paths[i:i+hash_batch_size]
            batch_hashes = []
            for frame_path in batch_paths:
                with Image.open(frame_path) as img:
                    frame_hash = imagehash.dhash(img, hash_size=8)
                    batch_hashes.append(frame_hash)
            frame_hashes.extend(batch_hashes)

            # Free memory after each batch
            del batch_paths
            del batch_hashes
            if i % 10000 == 0 and i > 0:
                logger.debug(f"Computed {i}/{len(frame_paths)} frame hashes...")

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

                # For large frame counts, be more selective (top 10% or max 200 frames)
                if len(frame_paths) > 500:
                    num_candidates = min(200, max(50, len(frame_paths) // 10))
                else:
                    num_candidates = max(20, len(frame_paths) // 5)

                candidate_frame_paths = [frame_paths[idx] for dist, idx in hash_distances[:num_candidates]]
                candidate_indices = [idx for dist, idx in hash_distances[:num_candidates]]

                logger.debug(f"Episode {episode_num}: Comparing {num_candidates} candidate frames (filtered from {len(frame_paths)})")

                # Process frames in parallel batches
                batch_size = 50  # Process 50 frames at a time to manage memory
                max_score = 0
                max_frame_idx = 0

                for i in range(0, len(candidate_frame_paths), batch_size):
                    batch_frame_paths = candidate_frame_paths[i:i+batch_size]
                    batch_indices = candidate_indices[i:i+batch_size]

                    # Parallel processing for this batch using threads (not processes)
                    # ThreadPoolExecutor works better with file paths (no pickling issues)
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Process frames in parallel using threads
                        futures = []
                        for frame_path in batch_frame_paths:
                            # Load image on-demand in thread
                            future = executor.submit(self._compare_image_from_path, frame_path, preprocessed_still)
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

            if avg_score > best_score:
                best_score = avg_score
                best_match = episode_data
                best_frame_index = best_still_frame_idx

        return {
            'best_match': best_match,
            'best_score': best_score,
            'best_frame_index': best_frame_index,
            'all_episode_scores': all_episode_scores
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
            if self.use_gpu and self.gpu_backend == 'torch':
                return self._compare_images_gpu_torch(img1, img2_preprocessed)
            elif self.use_gpu and self.gpu_backend == 'cupy':
                return self._compare_images_gpu_cupy(img1, img2_preprocessed)

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

    def _compare_images_gpu_torch(self, img1, img2_preprocessed):
        """
        GPU-accelerated image comparison using PyTorch
        Significantly faster for batch operations
        """
        try:
            # Preprocess the video frame
            arr1, gray1 = self._preprocess_image(img1)
            arr2, gray2 = img2_preprocessed

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

    def _compare_images_gpu_cupy(self, img1, img2_preprocessed):
        """
        GPU-accelerated image comparison using CuPy
        Alternative to PyTorch for systems with CuPy installed
        """
        try:
            # Preprocess the video frame
            arr1, gray1 = self._preprocess_image(img1)
            arr2, gray2 = img2_preprocessed

            # Convert to CuPy arrays (move to GPU)
            gray1_gpu = cp.asarray(gray1)
            gray2_gpu = cp.asarray(gray2)

            # Calculate SSIM on GPU
            ssim_score = self._ssim_cupy(gray1_gpu, gray2_gpu)

            if ssim_score > 0.5:
                arr1_gpu = cp.asarray(arr1)
                arr2_gpu = cp.asarray(arr2)
                hist_score = self._compare_histograms_gpu_cupy(arr1_gpu, arr2_gpu)
                combined_score = (ssim_score * 0.7) + (hist_score * 0.3)
            else:
                combined_score = ssim_score * 0.7

            return max(0, combined_score)

        except Exception as e:
            logger.error(f"Error in GPU comparison (cupy): {e}, falling back to CPU")
            # Fallback to CPU
            arr1, gray1 = self._preprocess_image(img1)
            arr2, gray2 = img2_preprocessed
            ssim_score = ssim(gray1, gray2, data_range=255)
            if ssim_score > 0.5:
                hist_score = self._compare_histograms_fast(arr1, arr2)
                return max(0, (ssim_score * 0.7) + (hist_score * 0.3))
            return max(0, ssim_score * 0.7)

    def _ssim_cupy(self, img1, img2):
        """
        Fast SSIM calculation using CuPy on GPU
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = float(cp.mean(img1))
        mu2 = float(cp.mean(img2))
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = float(cp.mean((img1 - mu1) ** 2))
        sigma2_sq = float(cp.mean((img2 - mu2) ** 2))
        sigma12 = float(cp.mean((img1 - mu1) * (img2 - mu2)))

        ssim_score = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                     ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_score

    def _compare_histograms_gpu_cupy(self, arr1, arr2):
        """
        GPU-accelerated histogram comparison using CuPy
        """
        try:
            scores = []
            for channel in range(3):
                # Calculate histograms on GPU
                hist1, _ = cp.histogram(arr1[:, :, channel].ravel(), bins=64, range=(0, 256))
                hist2, _ = cp.histogram(arr2[:, :, channel].ravel(), bins=64, range=(0, 256))

                # Normalize
                hist1 = hist1.astype(cp.float32) / cp.sum(hist1)
                hist2 = hist2.astype(cp.float32) / cp.sum(hist2)

                # Correlation
                mean1 = cp.mean(hist1)
                mean2 = cp.mean(hist2)
                std1 = cp.std(hist1)
                std2 = cp.std(hist2)

                if std1 > 0 and std2 > 0:
                    correlation = cp.sum((hist1 - mean1) * (hist2 - mean2)) / (std1 * std2 * len(hist1))
                    scores.append(float(correlation))
                else:
                    scores.append(0.0)

            return np.mean(scores)

        except Exception as e:
            logger.error(f"Error in GPU histogram comparison (cupy): {e}")
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
