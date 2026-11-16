# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OPENFAL_BUILD_CUDA=1 \
    OPENFAL_PROFILE=dual_5060ti_32gb

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/openfal
COPY . /workspace/openfal

RUN pip3 install --no-cache-dir --break-system-packages -e ".[torch,dev]"

EXPOSE 8000
CMD ["openfal", "serve", "--host", "0.0.0.0", "--port", "8000"]
