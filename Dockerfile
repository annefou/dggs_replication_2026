# DGGS Benchmark Replication Environment
# 
# This Dockerfile provides a reproducible environment for replicating
# the benchmarks from Law & Ardo (2024):
# "Using a discrete global grid system for a scalable, interoperable, 
# and reproducible system of land-use mapping"
# https://doi.org/10.1080/20964471.2024.2429847
#
# Build:  docker build -t dggs-benchmark-replication .
# Run:    docker run -it --rm -v $(pwd)/results:/app/results dggs-benchmark-replication
#
# Environment variables (passed via docker run -e):
#   VECTOR_LAYERS     Comma-separated list (e.g., "5,10,20,50,100")
#   RASTER_LAYERS     Comma-separated list (e.g., "10,50,100,500")
#   H3_RESOLUTION     H3 resolution for raster benchmark (default: 9)
#   RANDOM_SEED       Random seed for reproducibility (default: 42)
#
# Author: Anne Fouilloux
# Date: 2026-01-20

FROM python:3.11-slim-bookworm

# Set labels
LABEL maintainer="Anne Fouilloux <annef@simula.no>"
LABEL description="Reproducible environment for DGGS benchmark replication"
LABEL paper.doi="10.1080/20964471.2024.2429847"

# Create app directory
WORKDIR /app

# Install system dependencies including GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt

# Copy config.env for documentation
COPY config.env /app/config.env

# Install GDAL Python bindings matching system version, then other packages
RUN GDAL_VERSION=$(gdal-config --version) && \
    echo "System GDAL version: $GDAL_VERSION" && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "GDAL==$GDAL_VERSION" && \
    pip install --no-cache-dir -r requirements.txt

# Copy our replication script
COPY run_replication.py /app/
COPY README.md /app/

# Create directories for results
RUN mkdir -p /app/results /app/data/vector /app/data/raster

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV PYTHONUNBUFFERED=1

# Default configuration (can be overridden via docker run -e)
ENV VECTOR_LAYERS="5,10,20,50,100"
ENV RASTER_LAYERS="10,50,100,500"
ENV H3_RESOLUTION="9"
ENV RANDOM_SEED="42"

# Default command
CMD ["python", "run_replication.py", "--all", "--output", "results"]
