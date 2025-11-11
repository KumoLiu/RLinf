docker run -it --entrypoint bash \
   --shm-size 100g \
   --name rlinf \
   --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -v $HOME/.Xauthority:/isaac-sim/.Xauthority \
   -e DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /home/yunliu/Workspace/Code/:/code \
   -v /mnt/hdd:/yunl \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -e NVIDIA_DRIVER_CAPABILITIES=all \
   rlinf/rlinf:agentic-rlinf0.1-behavior

# docker run -it --entrypoint bash \
#    --shm-size 100g \
#    --name rlinf \
#    --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
#    -e "PRIVACY_CONSENT=Y" \
#    -v $HOME/.Xauthority:/isaac-sim/.Xauthority \
#    -e DISPLAY \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -v /localhome/local-yunl/:/colosuss \
#    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
#    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
#    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
#    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
#    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
#    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
#    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
#    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
#    -e NVIDIA_DRIVER_CAPABILITIES=all \
#    rlinf/rlinf:agentic-rlinf0.1-behavior
