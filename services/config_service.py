import json
import os
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)


class ConfigService:
    """
    Service for managing application configuration
    - Stores configuration in JSON file
    - Provides thread-safe access to settings
    - Falls back to environment variables for initial defaults
    """

    def __init__(self, config_dir='/app/config', config_file='settings.json'):
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / config_file
        self.lock = threading.Lock()
        self._config = None

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load or create config
        self._load_or_create_config()

    def _get_defaults(self):
        """Get default configuration from environment variables"""
        # Parse FPS from environment (allow None for "extract all frames")
        fps_env = os.environ.get('FRAME_FPS', '1')
        fps = None if fps_env.lower() in ('none', '', 'null', 'all') else float(fps_env)

        return {
            'tmdb_api_key': os.environ.get('TMDB_API_KEY', ''),
            'frame_height': int(os.environ.get('FRAME_HEIGHT', '480')),
            'frame_start_offset': float(os.environ.get('FRAME_START_OFFSET', '0.0')),
            'frame_end_offset': float(os.environ.get('FRAME_END_OFFSET', '1.0')),
            'frame_fps': fps,  # Default: 1 fps (recommended for episode matching)
            'auto_processing_enabled': False,
            'auto_processing_delay': 30,  # seconds

            # ImageMatcher configuration
            'matcher_max_candidates': 5000,
            'matcher_gpu_batch_size': 500,
            'matcher_cpu_batch_size': 50,
            'matcher_early_stop_threshold': 0.95,
            'matcher_max_workers': None,  # None = use CPU count

            # Frame extraction performance
            'extraction_parallel_enabled': True,  # Enable parallel extraction
            'extraction_parallel_workers': 4,  # Number of parallel FFmpeg processes
            'extraction_output_format': 'jpg'  # jpg or png (png is faster extraction, jpg smaller files)
        }

    def _load_or_create_config(self):
        """Load config from file or create default"""
        with self.lock:
            if self.config_file.exists():
                try:
                    with open(self.config_file, 'r') as f:
                        self._config = json.load(f)
                    logger.info(f"Loaded configuration from {self.config_file}")
                except Exception as e:
                    logger.error(f"Error loading config file: {e}")
                    self._config = self._get_defaults()
                    self._save_config()
            else:
                logger.info("Config file not found, creating default configuration")
                self._config = self._get_defaults()
                self._save_config()

    def _save_config(self):
        """Save current config to file (caller must hold lock)"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")

    def get_all(self):
        """Get all configuration settings"""
        with self.lock:
            return self._config.copy()

    def get(self, key, default=None):
        """Get a specific configuration value"""
        with self.lock:
            return self._config.get(key, default)

    def update(self, updates):
        """
        Update configuration settings

        Args:
            updates: Dictionary of settings to update

        Returns:
            dict: Updated configuration
        """
        with self.lock:
            self._config.update(updates)
            self._save_config()
            return self._config.copy()

    def set(self, key, value):
        """Set a specific configuration value"""
        with self.lock:
            self._config[key] = value
            self._save_config()

    # Convenience properties for commonly used settings
    @property
    def tmdb_api_key(self):
        return self.get('tmdb_api_key', '')

    @property
    def frame_height(self):
        return self.get('frame_height', 480)

    @property
    def frame_start_offset(self):
        return self.get('frame_start_offset', 0.0)

    @property
    def frame_end_offset(self):
        return self.get('frame_end_offset', 1.0)

    @property
    def auto_processing_enabled(self):
        return self.get('auto_processing_enabled', False)

    @property
    def auto_processing_delay(self):
        return self.get('auto_processing_delay', 30)

    @property
    def frame_fps(self):
        return self.get('frame_fps', 1)

    # ImageMatcher configuration properties
    @property
    def matcher_max_candidates(self):
        return self.get('matcher_max_candidates', 5000)

    @property
    def matcher_gpu_batch_size(self):
        return self.get('matcher_gpu_batch_size', 500)

    @property
    def matcher_cpu_batch_size(self):
        return self.get('matcher_cpu_batch_size', 50)

    @property
    def matcher_early_stop_threshold(self):
        return self.get('matcher_early_stop_threshold', 0.95)

    @property
    def matcher_max_workers(self):
        return self.get('matcher_max_workers', None)

    # Frame extraction performance properties
    @property
    def extraction_parallel_enabled(self):
        return self.get('extraction_parallel_enabled', True)

    @property
    def extraction_parallel_workers(self):
        return self.get('extraction_parallel_workers', 4)

    @property
    def extraction_output_format(self):
        return self.get('extraction_output_format', 'jpg')
