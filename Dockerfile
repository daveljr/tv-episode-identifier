FROM python:3.11-slim

# Install ffmpeg for video frame extraction
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create mount points
RUN mkdir -p /mnt/ripper /mnt/shows

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
