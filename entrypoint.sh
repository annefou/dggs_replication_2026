#!/bin/bash
# Entrypoint script for DGGS Benchmark Replication
#
# This script loads config.env but does NOT overwrite variables
# that are already set (e.g., from docker run -e VECTOR_LAYERS=...)
#
# Priority:
#   1. Environment variables passed via docker run -e (highest)
#   2. Values from config.env (fallback)

# Load config.env, only setting variables that aren't already defined
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ "$key" =~ ^#.*$ ]] && continue
    [[ -z "$key" ]] && continue
    # Remove quotes from value
    value="${value%\"}"
    value="${value#\"}"
    # Only set if not already defined (allows docker run -e to override)
    if [ -z "${!key}" ]; then
        export "$key=$value"
    fi
done < /app/config.env

# Execute the command passed to docker run
exec "$@"
