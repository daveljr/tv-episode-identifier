FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install NVIDIA video codec SDK libraries
# These provide libnvcuvid.so for hardware video decoding
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    libnvidia-decode-530 \
#    libnvidia-encode-530 \
#    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support for GPU-accelerated image matching
# Using CUDA 11.8 for maximum compatibility
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application files
COPY . .

# Copy hardware acceleration test video (23KB H.264 video for detection testing)
COPY hwaccel_test.mp4 /app/hwaccel_test.mp4

# Create mount points
RUN mkdir -p /app/input /app/output /app/logs /app/temp

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
