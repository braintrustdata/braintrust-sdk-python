SHELL := /bin/bash

.PHONY: help develop install-dev install-deps fixup test test-core test-wheel lint pylint nox bench bench-compare

develop: install-dev
	mise exec -- pre-commit install
	@echo "Use 'mise activate' in your shell for automatic tool and env activation."

install-dev:
	mise install

install-deps:
	mise exec -- $(MAKE) -C py install-dev

fixup:
	mise exec -- pre-commit run --all-files

test:
	mise exec -- $(MAKE) -C py test

test-core:
	mise exec -- $(MAKE) -C py test-core

test-wheel:
	mise exec -- $(MAKE) -C py test-wheel

lint:
	mise exec -- $(MAKE) -C py lint

pylint:
	mise exec -- $(MAKE) -C py pylint

bench:
	mise exec -- $(MAKE) -C py bench BENCH_ARGS="$(BENCH_ARGS)"

bench-compare:
	mise exec -- $(MAKE) -C py bench-compare BENCH_BASE="$(BENCH_BASE)" BENCH_NEW="$(BENCH_NEW)"

nox: test

help:
	@echo "Available targets:"
	@echo "  bench         - Run benchmarks via py/Makefile (pass extra flags via BENCH_ARGS=)"
	@echo "  bench-compare - Compare two benchmark results via py/Makefile (BENCH_BASE=... BENCH_NEW=...)"
	@echo "  develop       - Install tools with mise, install py/ deps, and install pre-commit hooks"
	@echo "  fixup         - Run pre-commit hooks across the repo"
	@echo "  install-deps  - Install Python SDK dependencies via py/Makefile"
	@echo "  install-dev   - Install pinned tools and create/update the repo env via mise"
	@echo "  lint          - Run pre-commit hooks plus Python SDK pylint via py/Makefile"
	@echo "  pylint        - Run Python SDK pylint only via py/Makefile"
	@echo "  nox           - Alias for test"
	@echo "  test          - Run the Python SDK nox matrix via py/Makefile"
	@echo "  test-core     - Run Python SDK core tests via py/Makefile"
	@echo "  test-wheel    - Run Python SDK wheel sanity tests via py/Makefile (requires a built wheel)"

.DEFAULT_GOAL := help
