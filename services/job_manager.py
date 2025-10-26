import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class JobManager:
    """Manages background jobs for episode identification with persistent storage"""

    def __init__(self, jobs_dir='/app/temp/jobs', video_service=None):
        self.jobs_dir = jobs_dir
        self.jobs = {}
        self.lock = threading.Lock()
        self.video_service = video_service  # Optional: for cleaning up frames

        # Create jobs directory if it doesn't exist
        os.makedirs(jobs_dir, exist_ok=True)

        # Load existing jobs from disk
        self._load_jobs_from_disk()

    def _load_jobs_from_disk(self):
        """Load all jobs from disk on startup"""
        try:
            for job_file in Path(self.jobs_dir).glob('*.json'):
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                        job_id = job_data['job_id']

                        # Mark any 'processing' jobs as 'error' since they were interrupted
                        if job_data['status'] == 'processing':
                            job_data['status'] = 'error'
                            job_data['error'] = 'Job interrupted by server restart'
                            self._save_job_to_disk(job_data)

                        self.jobs[job_id] = job_data
                        logger.info(f"Loaded job {job_id} from disk with status: {job_data['status']}")
                except Exception as e:
                    logger.error(f"Error loading job file {job_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading jobs from disk: {e}")

    def _save_job_to_disk(self, job_data):
        """Save a job to disk"""
        try:
            job_file = os.path.join(self.jobs_dir, f"{job_data['job_id']}.json")
            with open(job_file, 'w') as f:
                json.dump(job_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving job {job_data['job_id']} to disk: {e}")

    def create_job(self, folder_path, show_name, season):
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())

        job_data = {
            'job_id': job_id,
            'folder_path': folder_path,
            'show_name': show_name,
            'season': season,
            'status': 'queued',  # queued, processing, completed, error
            'progress': 0,  # 0-100
            'progress_message': 'Queued',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'results': None,
            'error': None,
            'session_id': None,
            'moved': False  # Track if files have been moved
        }

        with self.lock:
            self.jobs[job_id] = job_data
            self._save_job_to_disk(job_data)

        logger.info(f"Created job {job_id} for {show_name} Season {season}")
        return job_id

    def update_job(self, job_id, **kwargs):
        """Update job data"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)
                self._save_job_to_disk(self.jobs[job_id])

    def get_job(self, job_id):
        """Get job data by ID"""
        with self.lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self):
        """Get all jobs, sorted by creation time (newest first)"""
        with self.lock:
            jobs_list = list(self.jobs.values())
            jobs_list.sort(key=lambda x: x['created_at'], reverse=True)
            return jobs_list

    def delete_job(self, job_id):
        """Delete a job and its data from disk, including extracted frames"""
        with self.lock:
            if job_id in self.jobs:
                # Delete from memory
                del self.jobs[job_id]

                # Delete extracted frames if video_service is available
                if self.video_service:
                    try:
                        self.video_service.cleanup_job_frames(job_id)
                    except Exception as e:
                        logger.warning(f"Error cleaning up frames for job {job_id}: {e}")

                # Delete job JSON file from disk
                try:
                    job_file = os.path.join(self.jobs_dir, f"{job_id}.json")
                    if os.path.exists(job_file):
                        os.remove(job_file)
                    logger.info(f"Deleted job {job_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting job file for {job_id}: {e}")
                    return False
        return False

    def set_processing(self, job_id):
        """Mark job as processing"""
        self.update_job(
            job_id,
            status='processing',
            started_at=datetime.now().isoformat(),
            progress=5,
            progress_message='Starting identification...'
        )

    def set_completed(self, job_id, results, session_id):
        """Mark job as completed with results"""
        self.update_job(
            job_id,
            status='completed',
            completed_at=datetime.now().isoformat(),
            progress=100,
            progress_message='Completed',
            results=results,
            session_id=session_id
        )

    def set_error(self, job_id, error_message):
        """Mark job as error"""
        self.update_job(
            job_id,
            status='error',
            completed_at=datetime.now().isoformat(),
            progress_message='Error',
            error=error_message
        )

    def update_progress(self, job_id, progress, message):
        """Update job progress"""
        self.update_job(
            job_id,
            progress=min(max(progress, 0), 100),  # Clamp between 0-100
            progress_message=message
        )

    def mark_as_moved(self, job_id):
        """Mark job as having files moved"""
        self.update_job(job_id, moved=True)
        logger.info(f"Job {job_id} marked as moved")

    def set_moving(self, job_id):
        """Mark job as currently moving files"""
        self.update_job(
            job_id,
            status='moving',
            progress_message='Moving files...'
        )
        logger.info(f"Job {job_id} set to moving status")

    def is_folder_processing(self, folder_path):
        """Check if a folder is currently being processed or queued"""
        with self.lock:
            for job in self.jobs.values():
                if job['folder_path'] == folder_path and job['status'] in ['queued', 'processing', 'moving']:
                    return True
            return False

    def get_folder_job_status(self, folder_path):
        """Get the status and progress of the most recent job for a folder"""
        with self.lock:
            folder_jobs = [job for job in self.jobs.values() if job['folder_path'] == folder_path]
            if not folder_jobs:
                return None
            # Sort by creation time and get most recent
            folder_jobs.sort(key=lambda x: x['created_at'], reverse=True)
            most_recent = folder_jobs[0]
            return {
                'status': most_recent['status'],
                'progress': most_recent.get('progress', 0)
            }
