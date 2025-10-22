import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2


class ImageMatcher:
    """Service for matching video frames to episode stills"""

    def match_videos_to_episodes(self, video_frames, episode_images):
        """
        Match videos to episodes based on frame similarity

        Args:
            video_frames: dict of {video_filename: [PIL Images]}
            episode_images: dict of {episode_number: {'image': PIL Image, 'name': str, 'episode_number': int}}

        Returns:
            dict of {video_filename: {'episode_number': int, 'episode_name': str, 'confidence': float,
                                      'matched_frame_index': int, 'best_match_episode': int,
                                      'all_scores': dict}}
        """
        matches = {}

        for video_file, frames in video_frames.items():
            best_match = None
            best_score = 0
            best_frame_index = 0
            all_episode_scores = {}

            # Compare each frame against all episode stills
            for episode_num, episode_data in episode_images.items():
                episode_still = episode_data['image']

                # Calculate similarity for each frame
                scores = []
                for frame_idx, frame in enumerate(frames):
                    score = self._compare_images(frame, episode_still)
                    scores.append((score, frame_idx))

                # Use the maximum score among all frames
                max_score, max_frame_idx = max(scores, key=lambda x: x[0]) if scores else (0, 0)
                all_episode_scores[episode_num] = max_score

                if max_score > best_score:
                    best_score = max_score
                    best_match = episode_data
                    best_frame_index = max_frame_idx

            if best_match:
                matches[video_file] = {
                    'episode_number': best_match['episode_number'],
                    'episode_name': best_match['name'],
                    'confidence': best_score,
                    'matched_frame_index': best_frame_index,
                    'best_match_episode': best_match['episode_number'],
                    'all_scores': all_episode_scores
                }
            else:
                # No match found, assign based on filename order as fallback
                matches[video_file] = {
                    'episode_number': 0,
                    'episode_name': 'Unknown',
                    'confidence': 0.0,
                    'matched_frame_index': 0,
                    'best_match_episode': 0,
                    'all_scores': {}
                }

        return matches

    def _compare_images(self, img1, img2):
        """
        Compare two images and return a similarity score between 0 and 1
        Uses a combination of histogram comparison and structural similarity
        """
        try:
            # Resize images to the same size for comparison
            target_size = (640, 360)  # Standard size for comparison
            img1_resized = img1.resize(target_size, Image.Resampling.LANCZOS)
            img2_resized = img2.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy arrays
            arr1 = np.array(img1_resized)
            arr2 = np.array(img2_resized)

            # Convert to grayscale for SSIM
            if len(arr1.shape) == 3:
                gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = arr1

            if len(arr2.shape) == 3:
                gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = arr2

            # Calculate structural similarity
            ssim_score = ssim(gray1, gray2)

            # Calculate histogram similarity for color images
            hist_score = 0
            if len(arr1.shape) == 3 and len(arr2.shape) == 3:
                hist_score = self._compare_histograms(arr1, arr2)

            # Combine scores (weighted average)
            combined_score = (ssim_score * 0.7) + (hist_score * 0.3)

            return max(0, combined_score)  # Ensure non-negative

        except Exception as e:
            print(f"Error comparing images: {e}")
            return 0.0

    def _compare_histograms(self, img1, img2):
        """Compare color histograms of two images"""
        try:
            # Calculate histograms for each channel
            hist1 = [cv2.calcHist([img1], [i], None, [256], [0, 256]) for i in range(3)]
            hist2 = [cv2.calcHist([img2], [i], None, [256], [0, 256]) for i in range(3)]

            # Normalize histograms
            hist1 = [cv2.normalize(h, h).flatten() for h in hist1]
            hist2 = [cv2.normalize(h, h).flatten() for h in hist2]

            # Compare histograms using correlation
            scores = []
            for h1, h2 in zip(hist1, hist2):
                score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                scores.append(score)

            # Return average score across all channels
            return np.mean(scores)

        except Exception as e:
            print(f"Error comparing histograms: {e}")
            return 0.0
