import subprocess
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoService:
    """
    Service for extracting frames from video files
    - ALWAYS uses hardware acceleration with automatic CPU fallback
    - ALWAYS extracts ALL frames from videos
    - ALWAYS stores frames on filesystem
    """

    def __init__(self, frame_height=480, start_offset=0.05, end_offset=0.95, fps=None, base_temp_dir='/app/temp',
                 parallel_enabled=True, parallel_workers=4, output_format='jpg'):
        """
        Initialize VideoService with optimized settings

        Args:
            frame_height: Height in pixels for extracted frames (default: 480, width auto-calculated)
            start_offset: Percentage into video to start extraction (default: 0.05 = 5%)
            end_offset: Percentage into video to end extraction (default: 0.95 = 95%)
            fps: Target frames per second for extraction (default: None = extract all frames)
                 Recommended: 1-2 fps for episode matching (reduces frames by 92-98%)
            base_temp_dir: Base directory for temporary files (default: /app/temp)
            parallel_enabled: Enable parallel frame extraction (default: True)
            parallel_workers: Number of parallel FFmpeg processes (default: 4)
            output_format: Output format - 'jpg' or 'png' (default: 'jpg')
                          PNG is faster for extraction but larger files
        """
        self.frame_height = frame_height
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.fps = fps
        self.base_temp_dir = Path(base_temp_dir)
        self.parallel_enabled = parallel_enabled
        self.parallel_workers = parallel_workers
        self.output_format = output_format.lower()
        self.hwaccel_type = None
        self.gpu_scaling_available = False  # Track if GPU scaling (scale_cuda) works

        # Detect hardware acceleration and GPU scaling capability
        self._detect_hardware_acceleration()

        # Create base temp directory
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("VideoService Configuration:")
        logger.info(f"  Hardware Acceleration: {self.hwaccel_type.upper() if self.hwaccel_type else 'CPU (No GPU)'}")
        if self.gpu_scaling_available:
            logger.info(f"  GPU Scaling: ENABLED (scale_cuda)")
        logger.info(f"  Parallel Extraction: {'ENABLED' if self.parallel_enabled else 'DISABLED'} ({self.parallel_workers} workers)")
        logger.info(f"  Output Format: {self.output_format.upper()}")
        if self.fps:
            logger.info(f"  Frame Extraction: {self.fps} FPS (selective)")
        else:
            logger.info(f"  Frame Extraction: ALL FRAMES (not selective)")
        logger.info(f"  Frame Range: {self.start_offset*100:.1f}% - {self.end_offset*100:.1f}% of video")
        logger.info(f"  Frame Size: height={self.frame_height}px, width=auto (aspect ratio preserved)")
        logger.info(f"  Storage: Filesystem ({self.base_temp_dir})")
        logger.info("=" * 70)

    def _detect_hardware_acceleration(self):
        """
        Detect best available hardware acceleration with smart fallback
        Priority order: CUDA (NVIDIA) -> QSV (Intel) -> VAAPI -> others -> CPU

        CUDA/CUVID works with driver 550.x but has compatibility issues with 572.16+
        This method tests CUDA and gracefully falls back if it fails

        Returns:
            str: Hardware acceleration type or None for CPU
        """
        # Check for NVIDIA GPU and driver version
        nvidia_gpu_present = False
        driver_version = None

        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                driver_version = result.stdout.strip()
                nvidia_gpu_present = True
                logger.debug(f"NVIDIA GPU detected with driver version: {driver_version}")
        except Exception:
            logger.debug("No NVIDIA GPU detected (nvidia-smi not found or failed)")

        # Test hardware acceleration methods
        # Try CUDA first if NVIDIA GPU is present
        test_methods = []

        if nvidia_gpu_present:
            # Try CUDA with actual h264 video (more reliable test than testsrc)
            test_methods.append(('cuda', ['-hwaccel', 'cuda']))

        # Add other acceleration methods
        test_methods.extend([
            ('qsv', ['-hwaccel', 'qsv']),
            ('vaapi', ['-hwaccel', 'vaapi']),
            ('dxva2', ['-hwaccel', 'dxva2']),
            ('videotoolbox', ['-hwaccel', 'videotoolbox']),
        ])

        for method_name, hwaccel_flags in test_methods:
            try:
                # Use real H.264 video for more accurate testing (especially for CUDA/CUVID)
                # Fall back to testsrc if test video doesn't exist
                test_video = '/app/hwaccel_test.mp4'
                use_test_video = Path(test_video).exists()

                if use_test_video:
                    # Test with real H.264 video - triggers actual decoder usage
                    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error'] + hwaccel_flags + [
                        '-i', test_video,
                        '-frames:v', '1',
                        '-f', 'null', '-'
                    ]
                else:
                    # Fallback to testsrc if test video not found
                    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error'] + hwaccel_flags + [
                        '-f', 'lavfi', '-i', 'testsrc=duration=0.1:size=256x256:rate=1',
                        '-f', 'null', '-'
                    ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                # Check for success: exit code 0 AND no CUDA errors in stderr
                # FFmpeg may return 0 even if hwaccel failed and fell back to CPU
                has_cuda_error = 'CUDA_ERROR' in result.stderr or 'Failed setup for format cuda' in result.stderr
                test_passed = result.returncode == 0 and not has_cuda_error

                if test_passed:
                    if method_name == 'cuda':
                        logger.info(f"✓ NVIDIA CUDA hardware acceleration detected (driver {driver_version})")
                        logger.info("  Using GPU-accelerated video decoding")

                        # Test GPU scaling (scale_cuda) if CUDA works
                        self._test_gpu_scaling(test_video if use_test_video else None)
                    else:
                        logger.info(f"✓ Hardware acceleration detected: {method_name.upper()}")
                    logger.debug(f"Test command: {' '.join(cmd)}")
                    self.hwaccel_type = method_name
                    return
                else:
                    if method_name == 'cuda':
                        # CUDA failed - likely driver 572.16+ compatibility issue
                        logger.info(f"ℹ️  NVIDIA GPU detected (driver {driver_version}) but CUDA hwaccel failed")
                        logger.info("   This is expected with driver 572.16+ - trying fallback methods")
                        logger.debug(f"CUDA test error: {result.stderr[:200]}")
                    else:
                        logger.debug(f"Hardware acceleration test failed for {method_name}: {result.stderr[:200]}")
            except Exception as e:
                logger.debug(f"Exception testing {method_name}: {e}")
                continue

        logger.info("ℹ️  Using CPU decoding (fast and reliable)")
        self.hwaccel_type = None
        return

    def _test_gpu_scaling(self, test_video_path=None):
        """
        Test if GPU scaling (scale_cuda) works with CUDA hardware acceleration
        Requires -hwaccel_output_format cuda to keep frames in GPU memory

        Args:
            test_video_path: Path to test video, or None to use testsrc
        """
        try:
            if test_video_path and Path(test_video_path).exists():
                # Test with real H.264 video using GPU scaling
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error',
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda',  # Keep frames in GPU memory
                    '-i', test_video_path,
                    '-vf', 'scale_cuda=-1:480,hwdownload,format=nv12',  # nv12 is CUDA-compatible
                    '-frames:v', '1',
                    '-f', 'null', '-'
                ]
            else:
                # Fallback to testsrc
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error',
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda',
                    '-f', 'lavfi', '-i', 'testsrc=duration=0.1:size=640x480:rate=1',
                    '-vf', 'scale_cuda=-1:480,hwdownload,format=nv12',  # nv12 is CUDA-compatible
                    '-f', 'null', '-'
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            # Check for success: no CUDA errors and successful execution
            has_error = (
                'CUDA_ERROR' in result.stderr or
                'Failed setup for format cuda' in result.stderr or
                'scale_cuda' in result.stderr  # Filter initialization errors
            )

            if result.returncode == 0 and not has_error:
                self.gpu_scaling_available = True
                logger.info("  ✓ GPU scaling (scale_cuda) available - will use for frame extraction")
                logger.debug(f"GPU scaling test command: {' '.join(cmd)}")
            else:
                self.gpu_scaling_available = False
                logger.info("  ℹ️  GPU scaling unavailable - using CPU scaling (still fast)")
                logger.debug(f"GPU scaling test failed: {result.stderr[:200]}")

        except Exception as e:
            self.gpu_scaling_available = False
            logger.debug(f"GPU scaling test exception: {e}")

    def extract_frames(self, video_path, folder_name=None, job_id=None):
        """
        Extract ALL frames from a video file
        Stores frames on filesystem in job-specific or folder-specific directory
        Uses parallel extraction if enabled for better GPU utilization

        Args:
            video_path: Path to video file
            folder_name: Folder name to use for organizing frames (legacy, for cleanup)
            job_id: Job ID for job-specific frame storage (preferred)

        Returns:
            List[str]: List of frame file paths
        """
        video_stem = Path(video_path).stem

        # Use job-specific directory if job_id provided, otherwise use folder_name
        if job_id:
            frames_dir = self.base_temp_dir / 'jobs' / job_id / 'frames' / video_stem
        else:
            frames_dir = self.base_temp_dir / folder_name / video_stem

        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get video duration
            duration = self._get_video_duration(video_path)
            if not duration:
                logger.warning(f"Could not determine duration for {video_path}")
                return []

            # Calculate time range
            start_time = duration * self.start_offset
            end_time = duration * self.end_offset
            extract_duration = end_time - start_time

            logger.info(f"Extracting frames from: {Path(video_path).name}")
            logger.info(f"  Time range: {start_time:.2f}s - {end_time:.2f}s ({extract_duration:.2f}s total)")
            logger.info(f"  Output: {frames_dir}")

            # Use parallel extraction if enabled and duration is long enough
            if self.parallel_enabled and extract_duration > 60:
                frame_files = self._extract_frames_parallel(
                    video_path, frames_dir, start_time, end_time
                )
            else:
                frame_files = self._extract_frames_sequential(
                    video_path, frames_dir, start_time, end_time
                )

            if frame_files:
                logger.info(f"✓ Extracted {len(frame_files)} frames → {frames_dir}")
            else:
                logger.warning(f"No frames extracted from {Path(video_path).name}")

            return frame_files

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return []

    def _extract_frames_sequential(self, video_path, frames_dir, start_time, end_time):
        """
        Sequential frame extraction (original method)

        Args:
            video_path: Path to video file
            frames_dir: Directory to store frames
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List[str]: List of frame file paths
        """
        # Determine file extension based on output format
        ext = 'png' if self.output_format == 'png' else 'jpg'
        output_pattern = str(frames_dir / f'frame%08d.{ext}')

        # Build and execute ffmpeg command
        cmd = self._build_ffmpeg_command(video_path, output_pattern, start_time, end_time)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg extraction failed for {Path(video_path).name}")
            logger.error(f"Error: {result.stderr[-1000:]}")  # Last 1000 chars
            return []

        # Get all extracted frames
        frame_files = sorted(frames_dir.glob(f'frame*.{ext}'))
        return [str(f) for f in frame_files]

    def _extract_frames_parallel(self, video_path, frames_dir, start_time, end_time):
        """
        Parallel frame extraction for better GPU utilization
        Splits video into chunks and processes them simultaneously

        Args:
            video_path: Path to video file
            frames_dir: Directory to store frames
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List[str]: List of frame file paths
        """
        import concurrent.futures

        duration = end_time - start_time
        num_workers = min(self.parallel_workers, int(duration / 30))  # At least 30s per chunk
        num_workers = max(1, num_workers)  # At least 1 worker

        logger.info(f"Using parallel extraction with {num_workers} workers")

        # Calculate chunk boundaries
        chunk_duration = duration / num_workers
        chunks = []
        for i in range(num_workers):
            chunk_start = start_time + (i * chunk_duration)
            chunk_end = start_time + ((i + 1) * chunk_duration)
            chunks.append((i, chunk_start, chunk_end))

        # Extract chunks in parallel
        ext = 'png' if self.output_format == 'png' else 'jpg'

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for chunk_id, chunk_start, chunk_end in chunks:
                # Each chunk uses a unique output pattern
                chunk_pattern = str(frames_dir / f'chunk{chunk_id}_%08d.{ext}')
                future = executor.submit(
                    self._extract_chunk,
                    video_path, chunk_pattern, chunk_start, chunk_end, chunk_id
                )
                futures.append(future)

            # Wait for all chunks to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.extend(result)
                except Exception as e:
                    logger.error(f"Error in parallel chunk extraction: {e}")

        # Rename files to sequential numbering
        results.sort()  # Sort by filename
        final_files = []
        for idx, temp_file in enumerate(results, start=1):
            temp_path = Path(temp_file)
            final_name = f'frame{idx:08d}.{ext}'
            final_path = frames_dir / final_name
            temp_path.rename(final_path)
            final_files.append(str(final_path))

        return final_files

    def _extract_chunk(self, video_path, output_pattern, start_time, end_time, chunk_id):
        """
        Extract a single chunk of frames

        Args:
            video_path: Path to video file
            output_pattern: Output file pattern
            start_time: Start time in seconds
            end_time: End time in seconds
            chunk_id: Chunk identifier

        Returns:
            List[str]: List of extracted frame paths
        """
        try:
            cmd = self._build_ffmpeg_command(video_path, output_pattern, start_time, end_time)

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg chunk {chunk_id} extraction failed")
                logger.error(f"Error: {result.stderr[-500:]}")
                return []

            # Get extracted frames for this chunk
            ext = 'png' if self.output_format == 'png' else 'jpg'
            frames_dir = Path(output_pattern).parent
            frame_files = sorted(frames_dir.glob(f'chunk{chunk_id}_*.{ext}'))

            logger.debug(f"Chunk {chunk_id}: Extracted {len(frame_files)} frames")

            return [str(f) for f in frame_files]

        except Exception as e:
            logger.error(f"Error extracting chunk {chunk_id}: {e}")
            return []

    def _build_ffmpeg_command(self, video_path, output_pattern, start_time, end_time):
        """
        Build optimized ffmpeg command with hardware acceleration

        Returns:
            list: FFmpeg command arguments
        """
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']

        # Add hardware acceleration (if available)
        if self.hwaccel_type == 'cuda':
            cmd.extend(['-hwaccel', 'cuda'])
            # Add GPU output format if GPU scaling is available
            if self.gpu_scaling_available:
                cmd.extend(['-hwaccel_output_format', 'cuda'])
        elif self.hwaccel_type == 'qsv':
            cmd.extend(['-hwaccel', 'qsv'])
        elif self.hwaccel_type == 'vaapi':
            cmd.extend(['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128'])
        elif self.hwaccel_type == 'videotoolbox':
            cmd.extend(['-hwaccel', 'videotoolbox'])
        elif self.hwaccel_type == 'dxva2':
            cmd.extend(['-hwaccel', 'dxva2'])

        # Input file with time range
        cmd.extend([
            '-ss', str(start_time),            # Seek to start time
            '-i', video_path,                  # Input file
            '-to', str(end_time - start_time), # Duration from start
        ])

        # Video filter: scale to height, auto width, maintain aspect ratio
        vf_filters = []

        # Add FPS filter first (if specified) to reduce frames before scaling
        if self.fps:
            vf_filters.append(f'fps={self.fps}')

        if self.gpu_scaling_available:
            # GPU-accelerated scaling (much faster for large videos)
            vf_filters.append(f'scale_cuda=-1:{self.frame_height}')  # GPU scaling
            vf_filters.append('hwdownload')                           # Transfer from GPU to CPU
            vf_filters.append('format=nv12')                          # CUDA-compatible pixel format
        else:
            # CPU scaling (works everywhere)
            vf_filters.append(f'scale=-1:{self.frame_height}')
            vf_filters.append('format=yuv420p')

        cmd.extend(['-vf', ','.join(vf_filters)])

        # Output settings based on format
        if self.output_format == 'png':
            # PNG: Lossless but larger files, faster encoding
            cmd.extend([
                '-compression_level', '1',  # Fast compression (0-9, lower = faster)
                '-f', 'image2',             # Image output format
                '-y',                       # Overwrite files without asking
                output_pattern
            ])
        else:
            # JPEG: Lossy but smaller files, slower encoding
            cmd.extend([
                '-q:v', '2',       # Quality (2 is high, range 1-31, lower is better)
                '-f', 'image2',    # Image output format
                '-y',              # Overwrite files without asking
                output_pattern
            ])

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        return cmd

    def _get_video_codec(self, video_path):
        """
        Detect the video codec used in a file

        Args:
            video_path: Path to video file

        Returns:
            str: Codec name (e.g., 'h264', 'hevc', 'vp9'), or None if detection fails
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                codec = result.stdout.strip().lower()
                logger.debug(f"Detected codec for {Path(video_path).name}: {codec}")
                return codec

        except Exception as e:
            logger.debug(f"Error detecting codec for {video_path}: {e}")

        return None

    def _get_video_duration(self, video_path):
        """
        Get video duration in seconds using ffprobe

        Args:
            video_path: Path to video file

        Returns:
            float: Duration in seconds, or None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())

        except Exception as e:
            logger.error(f"Error getting video duration for {video_path}: {e}")

        return None

    def get_duration_formatted(self, video_path):
        """
        Get video duration formatted as HH:MM:SS

        Args:
            video_path: Path to video file

        Returns:
            str: Duration formatted as HH:MM:SS, or None if failed
        """
        duration = self._get_video_duration(video_path)
        if duration is None:
            return None

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def is_minimum_duration(self, video_path, min_duration_minutes=20):
        """
        Check if video meets minimum duration requirement

        Args:
            video_path: Path to video file
            min_duration_minutes: Minimum duration in minutes (default 20 for TV episodes)

        Returns:
            bool: True if video is at least min_duration_minutes long
        """
        duration = self._get_video_duration(video_path)
        if duration is None:
            logger.warning(f"Could not determine duration for {Path(video_path).name}, skipping")
            return False

        duration_minutes = duration / 60
        meets_requirement = duration_minutes >= min_duration_minutes

        if not meets_requirement:
            logger.info(f"Skipping {Path(video_path).name}: {duration_minutes:.1f} min < {min_duration_minutes} min")

        return meets_requirement

    def cleanup_folder(self, folder_name):
        """
        Clean up extracted frames for a specific folder

        Args:
            folder_name: Folder name to clean up (e.g., "Show Name s01d1")
        """
        import shutil

        folder_path = self.base_temp_dir / folder_name
        if folder_path.exists():
            try:
                shutil.rmtree(folder_path)
                logger.info(f"✓ Cleaned up frames for: {folder_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {folder_path}: {e}")
        else:
            logger.debug(f"No cleanup needed for {folder_name} (doesn't exist)")

    def cleanup_job_frames(self, job_id):
        """
        Clean up extracted frames for a specific job

        Args:
            job_id: Job ID to clean up frames for
        """
        import shutil

        frames_path = self.base_temp_dir / 'jobs' / job_id / 'frames'
        if frames_path.exists():
            try:
                shutil.rmtree(frames_path)
                logger.info(f"✓ Cleaned up job frames for: {job_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup job frames {frames_path}: {e}")
        else:
            logger.debug(f"No cleanup needed for job {job_id} (doesn't exist)")
