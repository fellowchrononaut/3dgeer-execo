#!/bin/bash
# Build the Digital_Twin dataset-preparation image (run from the repo root).
docker build -t gutwrap/digital_twin:latest -f docker/Dockerfile_digital_twin docker
