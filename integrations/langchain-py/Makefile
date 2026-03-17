UV ?= uv
RUN := $(UV) run

.PHONY: help install test lint format fixup

install:
	$(UV) sync

test:
	$(RUN) pytest src

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff format .

fixup: format

help:
	@echo "Available targets:"
	@echo "  fixup   - Format the integration sources with ruff"
	@echo "  format  - Format the integration sources with ruff"
	@echo "  install - Install dependencies with uv sync"
	@echo "  lint    - Run ruff checks"
	@echo "  test    - Run pytest against src/"

.DEFAULT_GOAL := help
