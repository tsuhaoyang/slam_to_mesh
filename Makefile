# slam_to_mesh — container workflow.
# Everything runs in Docker (CPU by default). GPU pieces need host setup first
# (see docs/deployment.md and scripts/).

COMPOSE ?= docker compose
IMAGE   ?= slam2mesh:cpu

.PHONY: help build up up-gpu down logs shell test lint clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build the CPU image (pipeline + service + QuadriFlow + COLMAP)
	$(COMPOSE) build

up: ## Start the service (CPU) at http://localhost:8000/ui/
	$(COMPOSE) up -d slam2mesh
	@echo "→ http://localhost:8000/ui/  (healthz: /healthz)"

up-gpu: ## Start the GPU service (needs nvidia-container-toolkit; see docs/deployment.md)
	$(COMPOSE) --profile gpu up -d slam2mesh-gpu
	@echo "→ http://localhost:8000/ui/  (GPU profile)"

down: ## Stop and remove containers
	$(COMPOSE) --profile gpu down

logs: ## Tail service logs
	$(COMPOSE) logs -f

shell: ## Open a shell in the running container
	$(COMPOSE) exec slam2mesh bash

test: ## Run the test suite inside a throwaway container (mounts the repo)
	$(COMPOSE) run --rm --no-deps \
		-v "$(CURDIR)":/app -w /app \
		slam2mesh sh -c "pip install -q '.[dev,service,usd]' && python -m pytest -q"

lint: ## Run ruff inside the container
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR)":/app -w /app \
		slam2mesh sh -c "pip install -q ruff && ruff check src tests"

clean: ## Remove containers, image, and the jobs volume
	-$(COMPOSE) --profile gpu down -v
	-docker image rm $(IMAGE)
