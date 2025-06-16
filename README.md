
## Headless 3D renderer with HTTP streaming

### Not hooked up to Coordinator yet

#### Local version
    docker build -t yaboring-renderer ./local
    docker run -p 5000:5000 --name yaboring-renderer yaboring-renderer

#### Paperspace "ML-in-a-box" version

    sudo docker build -t yaboring-renderer .
    sudo docker run --gpus all -p 5000:5000 yaboring-renderer
