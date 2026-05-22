PYTHON ?= python

.PHONY: lint pylint test test-wheel _template-version clean fixup build verify-build verify help install-build-deps install-dev test-core check-stale-cassettes _check-git-clean bench bench-compare

clean:
	rm -rf build dist

fixup:
	# just run the whole repos fixup (aka pre-commit hooks)
	cd .. && make fixup

lint: fixup
	uv run nox -s pylint

pylint:
	uv run nox -s pylint

test:
	uv run nox -x

test-wheel:
	# This target runs a small set of sanity checks before we release our
	# build artifact to pypi. We skip running the full test suite because
	# (a) theoretically it should have been run before (b) it has
	# integration tests and we don't want to block the release because
	# some service is down, etc. This decision could be revisited anytime.
	# Note: Caller must run 'make build' first.
	uv run nox -s test_core -- --wheel

test-core:
	uv run nox -s test_core

check-stale-cassettes:
	$(PYTHON) scripts/check-stale-cassettes.py

bench:
	$(PYTHON) -m benchmarks $(BENCH_ARGS)

bench-compare:
	@if [ -z "$(BENCH_BASE)" ] || [ -z "$(BENCH_NEW)" ]; then \
		echo "Usage: make bench-compare BENCH_BASE=/tmp/base.json BENCH_NEW=/tmp/new.json"; \
		exit 1; \
	fi
	$(PYTHON) -m pyperf compare_to $(BENCH_BASE) $(BENCH_NEW)

_template-version:
	@$(PYTHON) scripts/template-version.py

build: clean _template-version
	uv build
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
	uv sync --group build --no-dev

install-dev: install-build-deps
	uv sync --group dev


.DEFAULT_GOAL := help
help:
	@echo "Available targets:"
	@echo "  bench               - Run benchmarks (pass extra flags via BENCH_ARGS=, e.g. --fast)"
	@echo "  bench-compare       - Compare two benchmark results (BENCH_BASE=... BENCH_NEW=...)"
	@echo "  build               - Build Python package"
	@echo "  check-stale-cassettes - Detect orphaned cassette version directories"
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
