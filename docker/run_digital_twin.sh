#!/bin/bash
# Start the Digital_Twin container (run from the repo root). Same mounts as the
# geer container (run_my_docker.sh), so tool paths are identical in both:
#   repo -> /home, captures -> /media. Runs as the host user, so outputs in
#   data/ are not root-owned. HF_HOME (Hugging Face token/cache, for SAM3) is
#   kept on the host. X11 as in run_my_docker.sh, for the COLMAP GUI (xpra).
PROJECT_DIR="$(pwd)"
DATASET_DIR="/home/deos/s.jois/EXECO/Simulator_Validation/"
HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"
mkdir -p "$HF_DIR"
docker rm -f Digital_Twin 2>/dev/null

echo "mount projects: $PROJECT_DIR --> Digital_Twin:/home"
echo "mount datasets: $DATASET_DIR --> Digital_Twin:/media"
echo "mount HF cache: $HF_DIR --> Digital_Twin:/workspace/hf"

docker run --name Digital_Twin -t -d --gpus 'all,"capabilities=compute,utility,graphics"' --ipc=host --net=host \
    --user "$(id -u):$(id -g)" \
    --volume="$DATASET_DIR:/media:rw" \
    --volume="$PROJECT_DIR:/home:rw" \
    --volume="$HF_DIR:/workspace/hf:rw" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY --env=QT_X11_NO_MITSHM=1 \
    gutwrap/digital_twin:latest

# docker exec -it Digital_Twin /bin/bash
