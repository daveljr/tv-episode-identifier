import subprocess
import cv2
import numpy as np
from PIL import Image
import tempfile
import os


class VideoService:
    """Service for extracting frames from video files"""

    def extract_frames(self, video_path, num_frames=5):
        """
        Extract frames from a video file at different timestamps
        Returns a list of PIL Images
        """
        frames = []

        try:
            # Get video duration using ffprobe
            duration = self._get_video_duration(video_path)
            if not duration:
                print(f"Could not determine duration for {video_path}")
                return frames

            # Extract frames at evenly spaced intervals
            # Skip the very beginning and end to avoid black screens or credits
            start_offset = duration * 0.15  # Start at 15% into the video
            end_offset = duration * 0.85    # End at 85% into the video
            usable_duration = end_offset - start_offset

            timestamps = [start_offset + (usable_duration / (num_frames + 1)) * (i + 1)
                         for i in range(num_frames)]

            for timestamp in timestamps:
                frame = self._extract_frame_at_timestamp(video_path, timestamp)
                if frame is not None:
                    frames.append(frame)

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")

        return frames

    def _get_video_duration(self, video_path):
        """Get video duration in seconds using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())

        except Exception as e:
            print(f"Error getting video duration: {e}")

        return None

    def _extract_frame_at_timestamp(self, video_path, timestamp):
        """Extract a single frame at the given timestamp"""
        try:
            # Create temporary file for the frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name

            # Use ffmpeg to extract frame
            cmd = [
                'ffmpeg',
                '-ss', str(timestamp),
                '-i', video_path,
                '-frames:v', '1',
                '-q:v', '2',
                '-y',
                tmp_path
            ]

            result = subprocess.run(cmd, capture_output=True, stderr=subprocess.DEVNULL)

            if result.returncode == 0 and os.path.exists(tmp_path):
                # Load image with PIL
                image = Image.open(tmp_path)
                image.load()  # Force load before deleting temp file
                os.unlink(tmp_path)
                return image

            # Clean up if failed
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        except Exception as e:
            print(f"Error extracting frame at {timestamp}s: {e}")

        return None
