# DGGS Benchmark Replication - Makefile
#
# Usage:
#   make help          Show available commands
#   make docker-build  Build Docker image
#   make docker-run    Run replication in Docker
#   make local-setup   Setup local Python environment
#   make run           Run replication locally
#   make clean         Clean generated files

.PHONY: help docker-build docker-run local-setup run clean test

# Load configuration from config.env
include config.env
export

# Default target
help:
	@echo "DGGS Benchmark Replication Study"
	@echo "================================="
	@echo ""
	@echo "Configuration (from config.env):"
	@echo "  Original repo:    $(DGGS_BENCHMARKS_REPO)"
	@echo "  Original version: $(DGGS_BENCHMARKS_VERSION)"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run complete replication in Docker"
	@echo "  make docker-shell   Open shell in Docker container"
	@echo ""
	@echo "Local commands:"
	@echo "  make local-setup    Create Python venv and install deps"
	@echo "  make run            Run complete replication locally"
	@echo "  make run-vector     Run vector benchmark only"
	@echo "  make run-raster     Run raster benchmark only"
	@echo "  make generate-data  Generate synthetic data only"
	@echo "  make plot           Generate plots from existing results"
	@echo ""
	@echo "Utility commands:"
	@echo "  make clean          Remove generated files"
	@echo "  make test           Run quick test (small dataset)"
	@echo "  make show-config    Show current configuration"
	@echo ""

# Show configuration
show-config:
	@echo "Current configuration (from config.env):"
	@echo "  DGGS_BENCHMARKS_REPO:    $(DGGS_BENCHMARKS_REPO)"
	@echo "  DGGS_BENCHMARKS_VERSION: $(DGGS_BENCHMARKS_VERSION)"
	@echo "  PYTHON_VERSION:          $(PYTHON_VERSION)"
	@echo "  H3_RESOLUTION:           $(H3_RESOLUTION)"
	@echo "  RANDOM_SEED:             $(RANDOM_SEED)"

# Docker commands
DOCKER_IMAGE = dggs-benchmark-replication
DOCKER_TAG = latest

docker-build:
	docker build \
		--build-arg DGGS_BENCHMARKS_VERSION=$(DGGS_BENCHMARKS_VERSION) \
		--build-arg DGGS_BENCHMARKS_REPO=$(DGGS_BENCHMARKS_REPO) \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: docker-build
	docker run -it --rm \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/data:/app/data \
		-e DGGS_BENCHMARKS_VERSION=$(DGGS_BENCHMARKS_VERSION) \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		python run_replication.py --all

docker-shell: docker-build
	docker run -it --rm \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		bash

# Local Python environment
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

local-setup: $(VENV)/bin/activate
	@echo "Virtual environment ready. Activate with: source $(VENV)/bin/activate"

# Run commands (local)
run: $(VENV)/bin/activate
	$(PYTHON) run_replication.py --all

run-vector: $(VENV)/bin/activate
	$(PYTHON) run_replication.py --generate-data --vector

run-raster: $(VENV)/bin/activate
	$(PYTHON) run_replication.py --generate-data --raster

generate-data: $(VENV)/bin/activate
	$(PYTHON) run_replication.py --generate-data

plot: $(VENV)/bin/activate
	$(PYTHON) run_replication.py --plot

# Quick test with small dataset
test: $(VENV)/bin/activate
	@echo "Running quick test with reduced dataset..."
	$(PYTHON) -c "import run_replication; \
		run_replication.CONFIG['vector']['num_layers_list'] = [5, 10]; \
		run_replication.CONFIG['raster']['num_layers_list'] = [5, 10]; \
		run_replication.main()" --all

# Cleanup
clean:
	rm -rf results/
	rm -rf data/
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-results:
	rm -rf results/

clean-data:
	rm -rf data/
