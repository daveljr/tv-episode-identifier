# Recent Changes

## Updates Made

### 1. Renamed "Rename and Move" Button to "Move"
- Changed button text from "Rename and Move" to just "Move"
- Updated functionality to move files without renaming them
- Files now retain their current filenames when moved to `/mnt/shows/`

### 2. Added Manual Editing Capability
- Episode numbers and names are now editable directly in the results table
- Click on any episode number or name field to edit
- Changes are automatically saved and used during rename/move operations
- Useful for correcting misidentified episodes before processing

### 3. Added Image Previews
- New "Show Images" button for each episode in the results
- Displays side-by-side comparison of:
  - **Video Frame**: Extracted frame from the MKV file that was used for matching
  - **Episode Still**: Official still image from TheMovieDB
- Toggle-able to keep the interface clean
- Helps verify match accuracy visually

### 4. Added Match/No Match Indicators
- Color-coded badges showing match status:
  - **Green "Match" badge**: Confidence >= 40%
  - **Red "No Match" badge**: Confidence < 40%
- Makes it easier to spot low-confidence matches that need review
- Displayed in a new "Match" column in the results table

### 5. Enhanced Backend
- Updated image matcher to track which frame matched best
- Returns base64-encoded images to frontend for preview
- Added session storage for image data
- New `/api/move` endpoint (replaces `/api/rename-and-move`)

## API Changes

### Modified Endpoints

**`/api/identify` (POST)**
- Now returns additional fields:
  - `matched_frame`: Base64-encoded image of the matching video frame
  - `matched_still`: Base64-encoded image of the TMDB episode still
  - `is_match`: Boolean indicating if confidence is above threshold
  - `all_scores`: Dictionary of all episode similarity scores

**`/api/move` (POST)** - New endpoint
- Moves files without renaming them
- Replaces the old `/api/rename-and-move` endpoint
- Parameters: `folder_path`, `episodes`, `show_name`, `season`

### Unchanged Endpoints

**`/api/folders` (GET)**
- Lists all folders in ripper directory (unchanged)

**`/api/rename` (POST)**
- Renames files in place (unchanged)

## Workflow Changes

### Old Workflow
1. Identify episodes
2. Review results
3. Choose: "Rename" OR "Rename and Move"

### New Workflow
1. Identify episodes
2. Review results with image previews
3. Manually edit any incorrect episode numbers/names
4. Click "Show Images" to verify matches visually
5. Choose: "Rename" (in place) OR "Move" (to /mnt/shows without renaming)

### Typical Use Case
1. Identify episodes
2. Review and manually correct any errors
3. Click "Rename" to apply proper naming
4. Click "Move" to transfer to final location

## Benefits

- **More Control**: Manual editing allows fixing any identification errors
- **Visual Verification**: Image previews let you confirm matches before committing
- **Clearer Workflow**: Separate "Rename" and "Move" operations are more intuitive
- **Better Feedback**: Match/No Match badges quickly show which episodes need attention
- **Flexibility**: Can rename without moving, or move without renaming
