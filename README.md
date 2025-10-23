# TV Episode Identifier

A Docker-based web application that automatically identifies and renames TV episodes ripped from discs using TheMovieDB API and image matching.

## Features

- Web-based interface for managing ripped TV episodes
- Automatic episode identification using TheMovieDB API
- Image comparison between video frames and episode stills
- **GPU-accelerated image matching** with PyTorch/CUDA (2-8x faster)
- **Hardware-accelerated frame extraction** with automatic CPU fallback (FFmpeg hwaccel)
- **Intelligent caching system** for TMDB data (7-day expiration)
- Weighted confidence scoring with preview of all episode matches
- Batch rename functionality
- Rename and move files to organized directory structure
- Built-in log viewer for monitoring and troubleshooting
- Configurable frame extraction settings via environment variables
- Comprehensive logging system with rotating log files

## Prerequisites

- Docker and Docker Compose installed
- TheMovieDB API key (free - get one at https://www.themoviedb.org/settings/api)
- Ripped TV episodes in folders following the naming convention: `{TV Show Name} s{Season #}d{Disk #}`
  - Example: `The Office s01d1`

## Folder Structure

Your ripped episodes should be organized as follows:

```
/app/input/
├── The Office s01d1/
│   ├── title_1.mkv
│   ├── title_2.mkv
│   └── title_3.mkv
├── Breaking Bad s01d1/
│   ├── title_1.mkv
│   └── title_2.mkv
└── ...
```

After processing, files will be moved to:

```
/app/output/
├── The Office (2005)/
│   └── Season 01/
│       ├── The Office s01e01.mkv
│       ├── The Office s01e02.mkv
│       └── The Office s01e03.mkv
└── ...
```

## Installation

1. Clone or download this repository

2. Create a `.env` file (optional - for custom paths):

```bash
TMDB_API_KEY=your_api_key_here
INPUT_PATH=/path/to/your/ripped/episodes
OUTPUT_PATH=/path/to/your/shows/library
```

3. The included `docker-compose.yml` is pre-configured with:
   - GPU support enabled by default (automatic CPU fallback)
   - Hardware-accelerated frame extraction (automatic detection)
   - Intelligent TMDB caching system
   - All required environment variables

4. Update paths in `docker-compose.yml`:
   - Replace `./test-data/input` with your ripped episodes path
   - Replace `./test-data/output` with your TV shows library path
   - Or use the `.env` file to set `INPUT_PATH` and `OUTPUT_PATH`

5. Build and run the container:

```bash
docker-compose up -d
```

6. Access the web interface at `http://localhost:5000`

## Usage

1. **View Folders**: The main page displays all folders in your ripper directory that match the naming convention

2. **Identify Episodes**:
   - Click the "Identify" button next to a folder
   - The app will:
     - Search TheMovieDB for the TV show
     - Download episode stills for the season
     - Extract frames from each video file
     - Compare frames to stills using image matching
     - Display results with confidence scores and match indicators

3. **Review Results**:
   - Check the identified episodes, names, and confidence scores
   - Green (80%+): High confidence match - very reliable
   - Yellow (60-79%): Medium confidence match - review recommended
   - Red (<60%): Low confidence match - manual verification needed
   - **Match Badge**: Shows "Match" (green) for 80%+ confidence, "No Match" (red) for below 80%

4. **Manual Editing**:
   - Edit episode numbers and names directly in the table
   - Changes are saved automatically and will be used when renaming/moving files
   - Useful for correcting misidentified episodes

5. **Preview Images**:
   - Click "Show Images" for any episode to see side-by-side comparison
   - **Video Frame**: The extracted frame from your MKV file
   - **Episode Still**: The official still from TheMovieDB
   - Visual comparison helps verify match accuracy

6. **Rename Files**:
   - Click "Rename" to rename files in place using the format: `{TV Show Name} s{Season}e{Episode}.mkv`
   - Uses the current episode number and name (including any manual edits)

7. **Move Files**:
   - Click "Move" to move files (with their current names) to `/app/output/{TV Show Name}/Season {Season}/`
   - Files keep their current names, so rename first if desired
   - Creates the destination directory structure automatically

8. **View Logs**:
   - Click the "View Logs" button in the header to open the log viewer
   - View application logs in real-time for monitoring and troubleshooting
   - Select number of log lines to display (100 to 5000)
   - Copy logs to clipboard for sharing or bug reports
   - Logs include detailed information about:
     - Frame extraction progress
     - TMDB API requests and responses
     - Image matching results
     - File operations (rename/move)
     - Errors and warnings

## How It Works

### Episode Identification

1. **Folder Parsing**: Extracts TV show name and season number from folder name
2. **API Query**: Searches TheMovieDB for the show and retrieves episode metadata
3. **Smart Caching**: Checks local cache for TMDB data (7-day expiration) to avoid repeated API calls
4. **Image Download**: Downloads official episode still images from TheMovieDB (cached locally)
5. **Frame Extraction**: Extracts **ALL frames** from each video using hardware-accelerated FFmpeg
   - Automatic GPU detection (CUDA, QSV, VAAPI, VideoToolbox, DXVA2)
   - Falls back to CPU if no GPU available
   - Frames scaled during extraction (default: 480 height, auto width)
   - Stored in `/app/temp/{folder_name}/{video_name}/`
6. **Image Matching**: Compares extracted frames to episode stills using:
   - Perceptual hashing for pre-filtering (reduces candidates by 90%)
   - Structural Similarity Index (SSIM) - 70% weight
   - Color histogram correlation - 30% weight
   - GPU-accelerated with PyTorch/CUDA when available
7. **Confidence Scoring**: Weighted average favoring high-confidence matches (80%+)
8. **Cleanup**: Automatically removes temporary frames after processing

### Matching Algorithm

The image matcher uses a combination of techniques:

- **SSIM (Structural Similarity)**: Compares image structure and luminance
- **Histogram Comparison**: Compares color distribution across RGB channels
- **Multi-frame Matching**: Tests multiple frames from each video to find the best match
- **Weighted Scoring**: Combines SSIM (70%) and histogram (30%) for final confidence score

## Configuration

### Environment Variables

**Required:**
- `TMDB_API_KEY`: Your TheMovieDB API key (get one at https://www.themoviedb.org/settings/api)

**Paths:**
- `INPUT_PATH`: Path to ripped episodes (default: `/app/input`)
- `OUTPUT_PATH`: Path to organized shows library (default: `/app/output`)
- `LOG_PATH`: Path to store log files (default: `/app/logs`)

**Frame Extraction Settings:**
- `FRAME_HEIGHT`: Height in pixels for extracted frames (default: `480`)
  - Width is auto-calculated to preserve aspect ratio
  - Lower values = faster processing, less disk space
  - Higher values = better visual quality (minimal accuracy improvement)
  - Recommended: `240` (fast), `360` (balanced), `480` (high quality)

- `FRAME_START_OFFSET`: Percentage into video to start extraction (default: `0.0`)
  - `0.05` = start at 5% to skip intros/logos
  - Range: `0.0` to `1.0`

- `FRAME_END_OFFSET`: Percentage into video to end extraction (default: `1.0`)
  - `0.95` = end at 95% to skip credits
  - Range: `0.0` to `1.0`

**Performance Notes:**
- **GPU acceleration is AUTOMATIC** for both frame extraction and image matching
  - Frame extraction: Tries CUDA → QSV → VAAPI → DXVA2 → VideoToolbox → CPU (in order)
  - Image matching: Automatically uses PyTorch/CUDA or CuPy if available
  - No configuration needed - completely automatic!
  - 2-8x faster with NVIDIA GPUs, graceful CPU fallback
- **ALL frames are extracted** from videos (no selective sampling)
  - Perceptual hashing pre-filters candidates (~90% reduction)
  - Enables maximum accuracy while maintaining fast matching
- **TMDB data is cached** locally for 7 days
  - Avoids repeated API calls for the same show/season
  - Stores in `/app/temp/downloads/{show_id}_{season}/`
- **Temporary frames are automatically cleaned up** after processing

### Example Configurations

```yaml
# Default - Balanced performance and quality
environment:
  - FRAME_HEIGHT=480
  - FRAME_START_OFFSET=0.05
  - FRAME_END_OFFSET=0.95

# Fast mode - Lower resolution for speed
environment:
  - FRAME_HEIGHT=240
  - FRAME_START_OFFSET=0.1
  - FRAME_END_OFFSET=0.9

# High quality - Better visual quality
environment:
  - FRAME_HEIGHT=480
  - FRAME_START_OFFSET=0.05
  - FRAME_END_OFFSET=0.95
```

**Note:** GPU acceleration is automatic - no configuration needed! The app will detect and use available GPUs for both frame extraction and image matching, with automatic CPU fallback if no GPU is found.

### GPU Setup

**Docker Users (Recommended):**

GPU support is **AUTOMATIC** - no configuration needed! Just run the container.

The app will automatically:
- ✅ Detect and use PyTorch/CUDA or CuPy for GPU-accelerated image matching
- ✅ Detect and use hardware acceleration for frame extraction (CUDA/QSV/VAAPI/DXVA2/VideoToolbox)
- ✅ Fall back to CPU if no GPU libraries are found
- ✅ Use all available GPUs for maximum performance
- ✅ Work perfectly on CPU-only systems (no errors, just slower)

**Troubleshooting Docker GPU Support:**

If you get an error about NVIDIA runtime when starting the container, you have two options:

**Option 1: Install NVIDIA Container Toolkit** (enables GPU)

For Windows with Docker Desktop:
- See [GPU-SETUP.md](GPU-SETUP.md) for detailed Windows setup instructions

For Linux:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Option 2: Run in CPU mode**

Comment out the `deploy` section in `docker-compose.yml`:
```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

The app works perfectly on CPU - just without GPU acceleration.

### Performance Comparison

**Hardware-Accelerated Frame Extraction:**

| Hardware | Time per Video | Notes |
|----------|---------------|-------|
| NVIDIA GPU (CUDA) | 2-3 seconds | Best performance |
| Intel GPU (QSV) | 3-5 seconds | Good performance |
| AMD GPU (VAAPI) | 3-5 seconds | Good performance |
| CPU only | 8-15 seconds | Still reasonable |

**GPU-Accelerated Image Matching:**

| Scenario | CPU Time | GPU Time | Speedup |
|----------|----------|----------|---------|
| 10 videos (typical episode count) | ~2-3 min | ~30-45 sec | 4-5x |
| Single video match | ~15-20 sec | ~5-8 sec | 2-3x |

**Overall Impact:**
- **With GPU**: ~45 seconds for 10 episodes (extraction + matching)
- **CPU only**: ~3-4 minutes for 10 episodes (still very usable!)

**Recommendation:** GPU acceleration is great but not required. The app is well-optimized for CPU-only operation.

## Troubleshooting

### No folders appearing

- Check that your folders follow the naming convention: `{Show Name} s{Season}d{Disk}`
- Verify the volume mount in `docker-compose.yml` is correct
- Ensure the container has read permissions on the mounted directory

### Low confidence scores

- Episode stills may not be available for all episodes on TheMovieDB
- Video quality or encoding may affect frame extraction
- **Solution**: Try adjusting `FRAME_START_OFFSET` and `FRAME_END_OFFSET` to capture different scenes
- **Solution**: Increase `FRAME_HEIGHT` for better visual quality (minimal accuracy improvement)
- Only matches with 80%+ confidence are considered reliable
- Manually review and correct matches below 80%
- Use the "Preview All Matches" button to see all episode comparisons

### API errors

- Verify your TheMovieDB API key is correct
- Check that you have an active internet connection
- TheMovieDB may have rate limits - wait a few minutes and try again

### Frame extraction failures

- Ensure videos are valid MKV files
- Check that ffmpeg can read the video files
- Verify the container has read permissions on the video files
- Check logs for hardware acceleration detection messages
- If GPU acceleration fails, the app will automatically fall back to CPU
- Ensure `/app/temp` volume has sufficient disk space for extracted frames
- Use the log viewer to see detailed error messages

### Checking logs

- Click "View Logs" in the web interface for real-time log viewing
- Logs are stored in `./logs/app.log` (mounted from container)
- Log files rotate automatically at 10MB (keeps 5 backup files)
- Use logs to troubleshoot issues with:
  - TMDB API connectivity
  - Frame extraction failures
  - Image matching performance
  - File operation errors

## Notes

- Processing time depends on the number of episodes and video file sizes
- **First run per show/season**: Downloads TMDB data and caches it locally for 7 days
- **Subsequent runs**: Uses cached data - much faster!
- Confidence scores are estimates - always review results before renaming
- Original files are preserved when using "Rename" (no data loss)
- "Move" will move files with current names, leaving the original folder empty
- Temporary extracted frames are automatically cleaned up after processing
- Hardware acceleration is automatic - no configuration needed
- GPU matching provides 2-8x speedup but CPU-only mode is still fast

## Credits

- Built with Flask, OpenCV, scikit-image, and Pillow
- Episode metadata from [TheMovieDB](https://www.themoviedb.org/)
- Designed for use with [AutomaticRippingMachine](https://github.com/automatic-ripping-machine/automatic-ripping-machine)

## License

MIT License - feel free to use and modify as needed
