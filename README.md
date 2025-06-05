
## Headless 3D renderer with HTTP streaming

#### Not hooked up to Coordinator yet, so move files into a Paperspace machine using their "ML-in-a-box" OS Template, then:

```
sudo docker build -t yaboring-renderer .

sudo docker run --gpus all -p 5000:5000 yaboring-renderer
```

