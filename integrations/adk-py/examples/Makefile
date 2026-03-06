PORT ?= 8888
UV ?= uv
RUN := $(UV) run

.PHONY: dev help clean install test lint format check
.DEFAULT_GOAL := help

dev:
	$(RUN) adk web --port $(PORT)

install:
	$(UV) sync

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff format .

check:
	$(RUN) mypy .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

help:
	@echo "Available targets:"
	@echo "  dev      - Start development server (default port: $(PORT))"
	@echo "  install  - Install dependencies with uv"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linter"
	@echo "  format   - Format code"
	@echo "  check    - Run type checker"
	@echo "  clean    - Clean up temporary files"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PORT     - Port for development server (default: $(PORT))"
	@echo ""
	@echo "Usage examples:"
	@echo "  make dev"
	@echo "  make dev PORT=3000"
