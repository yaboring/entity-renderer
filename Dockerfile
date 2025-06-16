FROM python:3.10-slim

# Install minimal system dependencies for OpenGL (headless or hardware)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libglu1-mesa-dev \
    libgl1-mesa-dri libosmesa6 \
    libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir moderngl Pillow pyrr flask

# App setup
WORKDIR /app
COPY render.py .

# Run the app
CMD ["python", "render.py"]