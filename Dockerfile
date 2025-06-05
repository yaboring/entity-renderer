FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    libxrender1 libsm6 libxext6

RUN pip3 install moderngl moderngl-window Pillow pyrr flask

WORKDIR /app
COPY render.py .

CMD ["python3", "render.py"]
