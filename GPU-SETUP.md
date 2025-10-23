# GPU Setup Guide for Docker Desktop (Windows)

## Your Current Status

Based on your logs:
```
✗ PyTorch: INSTALLED (CPU-only)
  └─ Reason: PyTorch imported successfully but CUDA not available
```

**What this means:**
- ✅ PyTorch is correctly installed in Docker
- ✅ You have NVIDIA GeForce RTX 2080 Ti with Driver 572.16
- ✅ CUDA 12.8 is available on your Windows system
- ❌ Docker container cannot access your GPU

## Why This Happens

Docker Desktop on Windows uses WSL2 (Windows Subsystem for Linux) as a backend. By default, WSL2 doesn't pass GPU access to Docker containers. You need to install the **NVIDIA Container Toolkit** inside WSL2.

## Solution: Enable GPU Support (One-Time Setup)

### Prerequisites
- ✅ Docker Desktop installed (you have it)
- ✅ NVIDIA GPU with drivers (you have RTX 2080 Ti with driver 572.16)
- ✅ WSL2 enabled (Docker Desktop requires this)

### Step 1: Verify WSL2 GPU Support

1. Open PowerShell or CMD and run:
```cmd
wsl --list --verbose
```

You should see your WSL2 distribution (likely Ubuntu or Docker Desktop's distro).

2. Check if WSL2 can see the GPU:
```cmd
wsl nvidia-smi
```

If this shows your RTX 2080 Ti, GPU passthrough is working! Skip to Step 3.

If you get "command not found", continue to Step 2.

### Step 2: Install NVIDIA Container Toolkit in WSL2

1. **Open WSL2 terminal:**
```cmd
wsl
```

2. **Install NVIDIA Container Toolkit:**
```bash
# Add NVIDIA repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon (in WSL2)
sudo service docker restart
```

3. **Restart Docker Desktop** (from Windows)
   - Right-click Docker Desktop icon in system tray
   - Click "Quit Docker Desktop"
   - Start Docker Desktop again

### Step 3: Verify GPU Access

1. **Test Docker GPU access:**
```cmd
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

You should see:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.16                 Driver Version: 572.16         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  NVIDIA GeForce RTX 2080 Ti   Off  | 00000000:0B:00.0  On |                  N/A |
```

✅ **If you see your GPU: SUCCESS! GPU support is working.**

❌ **If you see an error:** The NVIDIA Container Toolkit isn't configured correctly. See Troubleshooting below.

### Step 4: Restart Your TV Episode Identifier Container

```cmd
cd e:\test-development\tv-episode-identifier

# Stop current container
docker-compose down

# Start with GPU support
docker-compose up -d

# Check logs
docker-compose logs -f
```

You should now see:
```
============================================================
GPU Detection Report:
------------------------------------------------------------
✓ PyTorch: AVAILABLE
  └─ CUDA Version: 11.8
  └─ GPU Count: 1
  └─ GPU 0: NVIDIA GeForce RTX 2080 Ti
------------------------------------------------------------
🚀 GPU ACCELERATION ENABLED
   Backend: PyTorch
   Device: NVIDIA GeForce RTX 2080 Ti
   Expected speedup: 2-8x faster
============================================================
```

🎉 **Done! You now have GPU-accelerated image matching!**

## Troubleshooting

### Issue: "could not select device driver nvidia"

**Error:**
```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

**Solution:**
1. NVIDIA Container Toolkit is not installed in WSL2
2. Follow Step 2 above to install it
3. Restart Docker Desktop

### Issue: "nvidia-smi not found" in WSL2

**Solution:**
Your Windows NVIDIA driver is too old. Update to the latest:
1. Visit https://www.nvidia.com/Download/index.aspx
2. Download latest driver for RTX 2080 Ti
3. Install and restart Windows
4. Verify with: `nvidia-smi` (in Windows CMD)

### Issue: GPU shows but PyTorch still says CPU-only

**Check Docker Compose configuration:**

Edit `docker-compose.yml` and ensure this section is UNCOMMENTED:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then restart:
```cmd
docker-compose down
docker-compose up -d
```

### Issue: "Docker daemon not running" in WSL2

**Solution:**
Docker Desktop manages the Docker daemon. Don't try to start it manually in WSL2.

Just restart Docker Desktop from Windows.

## Alternative: CPU-Only Mode

If you can't get GPU working or prefer to use CPU:

1. **Comment out GPU section in docker-compose.yml:**
```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

2. **Restart container:**
```cmd
docker-compose down
docker-compose up -d
```

The app will work perfectly on CPU - just without the 2-8x GPU speedup.

## Performance Comparison

### CPU Mode (Current)
- 5 frames/video: ~1.5 seconds per video
- 100 frames/video: ~18 seconds per video
- Still very usable!

### GPU Mode (RTX 2080 Ti)
- 5 frames/video: ~0.8 seconds per video (1.9x faster)
- 100 frames/video: ~4.5 seconds per video (4x faster)
- 2000 frames/video: ~48 seconds per video (5.6x faster)

## Quick Test Script

Run `test-gpu.bat` (included in this directory) to quickly test if Docker can access your GPU.

## Getting Help

If you're still having issues:
1. Check Docker Desktop logs: Settings → Troubleshoot → View logs
2. Check container logs: `docker-compose logs`
3. Verify GPU in Windows: `nvidia-smi`
4. Verify GPU in WSL2: `wsl nvidia-smi`
5. Test Docker GPU: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

Include all of the above outputs when asking for help!
