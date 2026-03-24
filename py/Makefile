PYTHON ?= python
UV := $(PYTHON) -m uv
UV_VERSION := $(shell awk '$$1=="uv" { print $$2 }' ../.tool-versions)

.PHONY: lint pylint test test-wheel _template-version clean fixup build verify-build verify help install-build-deps install-dev install-optional test-core _check-git-clean

clean:
	rm -rf build dist

fixup:
	# just run the whole repos fixup (aka pre-commit hooks)
	cd .. && make fixup

lint: fixup
	nox -s pylint

pylint:
	nox -s pylint

test:
	nox -x

test-wheel:
	# This target runs a small set of sanity checks before we release our
	# build artifact to pypi. We skip running the full test suite because
	# (a) theoretically it should have been run before (b) it has
	# integration tests and we don't want to block the release because
	# some service is down, etc. This decision could be revisited anytime.
	# Note: Caller must run 'make build' first.
	nox -s test_core -- --wheel

test-core:
	nox -s test_core

_template-version:
	@$(PYTHON) scripts/template-version.py

build: clean _template-version
	$(PYTHON) -m build
	# Restore the original version file after the build
	git checkout src/braintrust/version.py

_check-git-clean:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Git working directory is not clean."; \
		exit 1; \
	fi

verify-build: _check-git-clean build test-wheel

verify: lint test

install-build-deps:
	$(if $(UV_VERSION),,$(error Failed to read uv version from ../.tool-versions))
	$(PYTHON) -m pip install uv==$(UV_VERSION)
	$(UV) pip install -e .
	$(UV) pip install -r requirements-build.txt

install-dev: install-build-deps
	$(UV) pip install -r requirements-dev.txt

install-optional: install-dev
	$(UV) pip install -r requirements-optional.txt
	$(UV) pip install -e .[temporal,otel]

.DEFAULT_GOAL := help
help:
	@echo "Available targets:"
	@echo "  build               - Build Python package"
	@echo "  clean               - Remove build artifacts"
	@echo "  help                - Show this help message"
	@echo "  install-build-deps  - Install build dependencies for CI"
	@echo "  install-dev         - Install package in development mode with all dependencies"
	@echo "  lint                - Run pylint checks"
	@echo "  pylint              - Run pylint without pre-commit hooks"
	@echo "  test                - Run all tests"
	@echo "  test-core           - Run core tests only"
	@echo "  test-wheel          - Run tests against built wheel"
	@echo "  verify              - Run all CI checks"
	@echo "  verify-build        - Verify git clean, build, and test wheel"
