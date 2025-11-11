xhost +local:
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
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   rlinf/rlinf:agentic-rlinf0.1-behavior

https://github.com/isaac-sim/IsaacSim/blob/47d886f2858d1ceed556b21c88927aa67bc81c12/tools/docker/Dockerfile#L30

-v /home/nvidia/workspace/yunl/:/yunl \

docker run -it  --shm-size 100g \
   --name rlinf \
   --runtime=nvidia \
   --gpus all \
   -e "ACCEPT_EULA=Y" \
   --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -v $HOME/.Xauthority:/isaac-sim/.Xauthority \
   -e DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /home/yunliu/Workspace/Code/:/code \
   -v /mnt/hdd:/yunl \
   -v /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   rlinf/rlinf:agentic-rlinf0.1-behavior /bin/bash
