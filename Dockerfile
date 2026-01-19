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
# Uses mamba for clean GDAL/geospatial dependency management
#
# Author: Anne Fouilloux
# Date: 2026-01-17

FROM mambaorg/micromamba:1.5-bookworm-slim

# Set labels
LABEL maintainer="Anne Fouilloux <annef@simula.no>"
LABEL description="Reproducible environment for DGGS benchmark replication"
LABEL paper.doi="10.1080/20964471.2024.2429847"

# Switch to root for system setup
USER root

# Install git (needed to clone benchmark repo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy config.env FIRST - this is our single source of truth
COPY --chown=$MAMBA_USER:$MAMBA_USER config.env /app/config.env

# Copy environment file for mamba
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /app/environment.yml

# Switch back to micromamba user
USER $MAMBA_USER

# Create the conda environment with all dependencies
# This handles GDAL and all geospatial libs cleanly!
RUN micromamba install -y -n base -f /app/environment.yml && \
    micromamba clean --all --yes

# Activate the environment by default
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Clone the original benchmark repository using version from config.env
USER root
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
    find . -name "*.py" -type f | head -20 && \
    chown -R $MAMBA_USER:$MAMBA_USER /app/original_benchmarks || \
    echo "Warning: Could not clone original repo"

# Copy our replication scripts
COPY --chown=$MAMBA_USER:$MAMBA_USER run_replication.py /app/
COPY --chown=$MAMBA_USER:$MAMBA_USER run_original_benchmarks.py /app/
COPY --chown=$MAMBA_USER:$MAMBA_USER README.md /app/

# Create directories for results
RUN mkdir -p /app/results /app/data/vector /app/data/raster && \
    chown -R $MAMBA_USER:$MAMBA_USER /app/results /app/data

# Copy entrypoint script
COPY --chown=$MAMBA_USER:$MAMBA_USER entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch back to mamba user for running
USER $MAMBA_USER

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/entrypoint.sh"]
CMD ["python", "run_replication.py", "--all"]
