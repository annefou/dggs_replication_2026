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
# All configuration is read from config.env - single source of truth!
#
# Author: Anne Fouilloux
# Date: 2026-01-17

FROM python:3.11-slim-bookworm

# Set labels
LABEL maintainer="Anne Fouilloux <annef@simula.no>"
LABEL description="Reproducible environment for DGGS benchmark replication"
LABEL paper.doi="10.1080/20964471.2024.2429847"

# Create app directory first
WORKDIR /app

# Copy config.env FIRST - this is our single source of truth
COPY config.env /app/config.env

# Install system dependencies for geospatial libraries
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
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone the original benchmark repository using version from config.env
RUN set -a && . /app/config.env && set +a && \
    echo "================================================" && \
    echo "Cloning original benchmark repository:" && \
    echo "  Repo:    ${DGGS_BENCHMARKS_REPO}" && \
    echo "  Version: ${DGGS_BENCHMARKS_VERSION}" && \
    echo "================================================" && \
    git clone --depth 1 --branch ${DGGS_BENCHMARKS_VERSION} \
        ${DGGS_BENCHMARKS_REPO}.git \
        /app/original_benchmarks && \
    cd /app/original_benchmarks && \
    echo "Cloned version: $(git describe --tags --always)" && \
    echo "" && \
    echo "=== Repository Contents ===" && \
    ls -la && \
    echo "" && \
    echo "=== Python Files ===" && \
    find . -name "*.py" -type f | head -20 || \
    echo "Warning: Could not clone original repo"

# Copy our replication scripts
COPY run_replication.py /app/
COPY README.md /app/

# Create directories for results
RUN mkdir -p /app/results /app/data/vector /app/data/raster

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV PYTHONUNBUFFERED=1

# Copy entrypoint script (loads config.env without overwriting -e vars)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "run_replication.py", "--all"]
