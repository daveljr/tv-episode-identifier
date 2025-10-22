# TV Episode Identifier

A Docker-based web application that automatically identifies and renames TV episodes ripped from discs using TheMovieDB API and image matching.

## Features

- Web-based interface for managing ripped TV episodes
- Automatic episode identification using TheMovieDB API
- Image comparison between video frames and episode stills
- Confidence scoring for episode matches
- Batch rename functionality
- Rename and move files to organized directory structure

## Prerequisites

- Docker and Docker Compose installed
- TheMovieDB API key (free - get one at https://www.themoviedb.org/settings/api)
- Ripped TV episodes in folders following the naming convention: `{TV Show Name} s{Season #}d{Disk #}`
  - Example: `The Office s01d1`

## Folder Structure

Your ripped episodes should be organized as follows:

```
/mnt/ripper/
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
/mnt/shows/
├── The Office/
│   └── Season 01/
│       ├── The Office s01e01.mkv
│       ├── The Office s01e02.mkv
│       └── The Office s01e03.mkv
└── ...
```

## Installation

1. Clone or download this repository

2. Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  tv-identifier:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - /path/to/your/ripped/episodes:/mnt/ripper
      - /path/to/your/shows/library:/mnt/shows
    environment:
      - TMDB_API_KEY=your_api_key_here
      - RIPPER_PATH=/mnt/ripper
      - SHOWS_PATH=/mnt/shows
    restart: unless-stopped
```

3. Update the `docker-compose.yml` file:
   - Replace `/path/to/your/ripped/episodes` with your actual path to ripped episodes
   - Replace `/path/to/your/shows/library` with your actual path to your organized TV shows
   - Replace `your_api_key_here` with your TheMovieDB API key

4. Build and run the container:

```bash
docker-compose up -d
```

5. Access the web interface at `http://localhost:5000`

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
   - Green (70%+): High confidence match
   - Yellow (40-69%): Medium confidence match
   - Red (<40%): Low confidence match - review carefully
   - **Match Badge**: Shows "Match" (green) for 40%+ confidence, "No Match" (red) for below 40%

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
   - Click "Move" to move files (with their current names) to `/mnt/shows/{TV Show Name}/Season {Season}/`
   - Files keep their current names, so rename first if desired
   - Creates the destination directory structure automatically

## How It Works

### Episode Identification

1. **Folder Parsing**: Extracts TV show name and season number from folder name
2. **API Query**: Searches TheMovieDB for the show and retrieves episode metadata
3. **Image Download**: Downloads official episode still images from TheMovieDB
4. **Frame Extraction**: Extracts 5 frames from each video file at evenly-spaced intervals (skipping intro/credits)
5. **Image Matching**: Compares extracted frames to episode stills using:
   - Structural Similarity Index (SSIM) - 70% weight
   - Color histogram correlation - 30% weight
6. **Confidence Scoring**: Returns the best match with a confidence percentage

### Matching Algorithm

The image matcher uses a combination of techniques:

- **SSIM (Structural Similarity)**: Compares image structure and luminance
- **Histogram Comparison**: Compares color distribution across RGB channels
- **Multi-frame Matching**: Tests multiple frames from each video to find the best match
- **Weighted Scoring**: Combines SSIM (70%) and histogram (30%) for final confidence score

## Configuration

Environment variables:

- `TMDB_API_KEY`: Your TheMovieDB API key (required)
- `RIPPER_PATH`: Path to ripped episodes (default: `/mnt/ripper`)
- `SHOWS_PATH`: Path to organized shows library (default: `/mnt/shows`)

## Troubleshooting

### No folders appearing

- Check that your folders follow the naming convention: `{Show Name} s{Season}d{Disk}`
- Verify the volume mount in `docker-compose.yml` is correct
- Ensure the container has read permissions on the mounted directory

### Low confidence scores

- Episode stills may not be available for all episodes on TheMovieDB
- Video quality or encoding may affect frame extraction
- Consider manually reviewing matches with <70% confidence

### API errors

- Verify your TheMovieDB API key is correct
- Check that you have an active internet connection
- TheMovieDB may have rate limits - wait a few minutes and try again

### Frame extraction failures

- Ensure videos are valid MKV files
- Check that ffmpeg can read the video files
- Verify the container has read permissions on the video files

## Notes

- Processing time depends on the number of episodes and video file sizes
- First run may be slower as images are downloaded
- Confidence scores are estimates - always review results before renaming
- Original files are preserved when using "Rename" (no data loss)
- "Rename and Move" will move files, leaving the original folder empty

## Credits

- Built with Flask, OpenCV, scikit-image, and Pillow
- Episode metadata from [TheMovieDB](https://www.themoviedb.org/)
- Designed for use with [AutomaticRippingMachine](https://github.com/automatic-ripping-machine/automatic-ripping-machine)

## License

MIT License - feel free to use and modify as needed
