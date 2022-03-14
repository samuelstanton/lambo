#!/usr/bin/env bash

cd /home/sam/Code/bo-protein-context
export DOCKER_BUILDKIT=1
docker build . -f bo-protein/Dockerfile --tag samuelstanton/bo-protein:py3.8_cuda11 --build-arg BUILDKIT_INLINE_CACHE=1
docker push samuelstanton/bo-protein:py3.8_cuda11
docker system prune -f
