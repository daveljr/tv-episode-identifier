import os
import time
import threading
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FolderWatcher:
    """
    Service for watching input folder and auto-processing new folders
    - Monitors input directory for new folders
    - Triggers processing after configured delay
    - Respects folder naming pattern: {show_name} s{season}d{disk}
    """

    def __init__(self, input_path, config_service, file_service, process_callback):
        """
        Initialize folder watcher

        Args:
            input_path: Path to watch for new folders
            config_service: ConfigService instance for settings
            file_service: FileService instance for parsing folder names
            process_callback: Function to call to start processing (folder_path, show_name, season)
        """
        self.input_path = input_path
        self.config_service = config_service
        self.file_service = file_service
        self.process_callback = process_callback

        self.watcher_thread = None
        self.stop_event = threading.Event()
        self.known_folders = set()
        self.pending_folders = {}  # folder_path: (show_name, season, detected_time)

        logger.info(f"FolderWatcher initialized for: {input_path}")

    def start(self):
        """Start the folder watcher in a background thread"""
        if self.watcher_thread and self.watcher_thread.is_alive():
            logger.warning("Folder watcher is already running")
            return

        logger.info("Starting folder watcher")
        self.stop_event.clear()
        self.watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watcher_thread.start()

    def stop(self):
        """Stop the folder watcher"""
        if not self.watcher_thread or not self.watcher_thread.is_alive():
            logger.warning("Folder watcher is not running")
            return

        logger.info("Stopping folder watcher")
        self.stop_event.set()
        if self.watcher_thread:
            self.watcher_thread.join(timeout=5)

    def _watch_loop(self):
        """Main watching loop"""
        logger.info("Folder watcher loop started")

        # Initialize known folders
        self._scan_existing_folders()

        while not self.stop_event.is_set():
            try:
                # Only watch if auto-processing is enabled
                if not self.config_service.auto_processing_enabled:
                    time.sleep(5)
                    continue

                # Scan for new folders
                self._scan_for_new_folders()

                # Process pending folders that have passed the delay
                self._process_pending_folders()

                # Sleep before next scan
                time.sleep(2)

            except Exception as e:
                logger.exception(f"Error in folder watcher loop: {e}")
                time.sleep(5)

        logger.info("Folder watcher loop stopped")

    def _scan_existing_folders(self):
        """Scan and remember existing folders to avoid processing them on startup"""
        try:
            if not os.path.exists(self.input_path):
                return

            for item in os.listdir(self.input_path):
                item_path = os.path.join(self.input_path, item)
                if os.path.isdir(item_path):
                    self.known_folders.add(item_path)

            logger.info(f"Scanned {len(self.known_folders)} existing folders")

        except Exception as e:
            logger.error(f"Error scanning existing folders: {e}")

    def _scan_for_new_folders(self):
        """Scan for new folders that appeared since last check"""
        try:
            if not os.path.exists(self.input_path):
                return

            current_folders = set()
            for item in os.listdir(self.input_path):
                item_path = os.path.join(self.input_path, item)
                if os.path.isdir(item_path):
                    current_folders.add(item_path)

            # Find new folders
            new_folders = current_folders - self.known_folders

            for folder_path in new_folders:
                folder_name = os.path.basename(folder_path)

                # Parse folder name to check if it matches pattern
                parsed = self.file_service.parse_folder_name(folder_name)

                if parsed:
                    show_name = parsed['show_name']
                    season = parsed['season']

                    # Add to pending folders with detection time
                    self.pending_folders[folder_path] = (show_name, season, time.time())

                    logger.info(f"New folder detected: {folder_name} -> {show_name} S{season}")
                else:
                    logger.info(f"New folder detected but doesn't match pattern: {folder_name}")

                # Add to known folders regardless
                self.known_folders.add(folder_path)

        except Exception as e:
            logger.error(f"Error scanning for new folders: {e}")

    def _process_pending_folders(self):
        """Process folders that have passed the configured delay"""
        try:
            delay = self.config_service.auto_processing_delay
            current_time = time.time()

            folders_to_process = []

            # Find folders ready to process
            for folder_path, (show_name, season, detected_time) in list(self.pending_folders.items()):
                if current_time - detected_time >= delay:
                    folders_to_process.append((folder_path, show_name, season))

            # Process folders
            for folder_path, show_name, season in folders_to_process:
                try:
                    logger.info(f"Auto-processing folder: {os.path.basename(folder_path)}")

                    # Call the processing callback
                    self.process_callback(folder_path, show_name, season)

                    # Remove from pending
                    del self.pending_folders[folder_path]

                except Exception as e:
                    logger.exception(f"Error auto-processing folder {folder_path}: {e}")
                    # Remove from pending even on error to avoid retry loop
                    del self.pending_folders[folder_path]

        except Exception as e:
            logger.error(f"Error processing pending folders: {e}")

    def get_status(self):
        """Get current watcher status"""
        return {
            'running': self.watcher_thread and self.watcher_thread.is_alive(),
            'enabled': self.config_service.auto_processing_enabled,
            'known_folders_count': len(self.known_folders),
            'pending_folders_count': len(self.pending_folders),
            'delay': self.config_service.auto_processing_delay
        }
