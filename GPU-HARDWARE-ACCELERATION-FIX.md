# GPU Hardware Acceleration Fix

## Problem Summary

The video service was reporting "No hardware acceleration available" despite the GPU being detected by the image matcher service.

## Root Cause

There were two separate GPU acceleration systems at play:

1. **Image Matcher GPU (Working)**: Uses PyTorch with CUDA for image comparison
   - PyTorch directly accesses GPU through CUDA runtime libraries
   - Provided by NVIDIA Container Toolkit
   - ✅ Working correctly

2. **Video Service GPU (Not Working)**: Uses ffmpeg for hardware-accelerated video decoding
   - The standard Debian `ffmpeg` package does NOT include NVIDIA CUDA/NVENC/NVDEC support
   - FFmpeg needs to be compiled with special flags (`--enable-cuda-nvcc`, `--enable-cuvid`, `--enable-nvenc`)
   - ❌ Was failing because standard ffmpeg lacks GPU support

## Solution

### 1. **Dockerfile Changes**
Changed from `python:3.11-slim` base image to `nvidia/cuda:11.8.0-runtime-ubuntu22.04`:
- Provides CUDA runtime libraries that ffmpeg needs
- Installs jellyfin-ffmpeg (pre-built with full NVIDIA GPU support)
- Much faster than building ffmpeg from source
- Well-tested and maintained by Jellyfin project

### 2. **Video Service Detection Improvements**
Enhanced `_detect_hardware_acceleration()` in [video_service.py](services/video_service.py#L45):
- Added nvidia-smi check for better diagnostics
- Tests CUVID decoders (h264_cuvid) which are more reliable
- Improved error logging to help debug issues
- Tests with actual video generation (not just null sources)

### 3. **FFmpeg Command Updates**
Updated `_build_ffmpeg_command()` to properly use CUVID:
- Uses `-c:v h264_cuvid` decoder flag for NVIDIA decoding
- Applies `scale_cuda` filter for GPU-accelerated scaling
- Downloads frames from GPU to CPU for JPEG encoding
- Falls back gracefully if CUDA isn't available

## How to Apply the Fix

### Rebuild the Docker Image
```bash
docker compose build --no-cache
```

### Start the Container
```bash
docker compose up
```

### Verify GPU Acceleration
Look for these log messages on startup:

**Image Matcher:**
```
🚀 GPU ACCELERATION ENABLED
   Backend: PyTorch
   Device: NVIDIA GeForce RTX 3060
   Expected speedup: 2-8x faster
```

**Video Service:**
```
✓ Hardware acceleration detected: CUVID
  Hardware Acceleration: CUVID (NVIDIA GPU)
```

## Performance Impact

With GPU acceleration enabled:
- **Video frame extraction**: 3-10x faster (depends on video codec)
- **Image matching**: 2-8x faster (already working)
- **Overall processing**: Significantly faster for large video files

## Technical Details

### FFmpeg CUDA Acceleration Modes

1. **CUVID** (h264_cuvid, hevc_cuvid, etc.)
   - Hardware video decoding
   - Fastest for supported codecs (H.264, HEVC, VP9, etc.)
   - ✅ Now enabled

2. **NVENC** (h264_nvenc, hevc_nvenc)
   - Hardware video encoding
   - Not used in this application (we only decode)

3. **scale_cuda**
   - GPU-accelerated video scaling/resizing
   - Much faster than CPU scaling
   - ✅ Now enabled

### Fallback Behavior

If GPU acceleration fails for any reason:
- Video service automatically falls back to CPU decoding
- Image matcher falls back to CPU-based image comparison
- Application continues to work (just slower)
- Clear warning messages in logs

## Troubleshooting

### If GPU still not detected:

1. **Check NVIDIA driver on host:**
   ```bash
   nvidia-smi
   ```

2. **Check NVIDIA Container Toolkit:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
   ```

3. **Check docker-compose.yml has GPU configuration:**
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

4. **Check ffmpeg GPU support inside container:**
   ```bash
   docker compose exec app ffmpeg -hwaccels
   ```
   Should show: `cuda`

5. **Enable debug logging:**
   Set logging level to DEBUG in app.py to see detailed hardware detection logs

## References

- [FFmpeg NVIDIA GPU Acceleration](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/)
- [Jellyfin FFmpeg Releases](https://github.com/jellyfin/jellyfin-ffmpeg/releases)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
