@echo off
REM Quick test to check if Docker can access GPU

echo Testing Docker GPU access...
echo.

docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo.
echo If you see your GPU listed above, Docker GPU support is working!
echo If you see an error, you need to install NVIDIA Container Toolkit in WSL2.
echo.
pause
