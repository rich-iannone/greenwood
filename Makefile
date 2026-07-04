PYTHON ?= .venv/bin/python

.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Install the package with dev extras into .venv
	@$(PYTHON) -m pip install -e ".[dev]"

.PHONY: test
test: ## Run the full test suite with coverage
	@$(PYTHON) -m pytest tests \
		--cov=greenwood \
		--cov-report=term-missing \
		--durations 10

.PHONY: test-unit
test-unit: ## Run tests excluding slow and R-parity suites
	@$(PYTHON) -m pytest tests -m "not slow and not rparity" --durations 10

.PHONY: test-rparity
test-rparity: ## Run the R-parity numeric validation suite
	@$(PYTHON) -m pytest tests -m rparity --durations 10

.PHONY: lint
lint: ## Run ruff formatter and linter (with fixes)
	@$(PYTHON) -m ruff format
	@$(PYTHON) -m ruff check --fix

.PHONY: check-format
check-format: ## Check formatting and lint without making changes
	@$(PYTHON) -m ruff format --check
	@$(PYTHON) -m ruff check

.PHONY: type-check
type-check: ## Run pyright in strict mode
	@$(PYTHON) -m pyright greenwood

.PHONY: check
check: lint type-check test ## Run all checks (the pre-push gate)

# The user-guide pages contain executable code, so Quarto needs the project venv's
# Jupyter kernel (which has greenwood and its dependencies installed).
JUPYTER_PATH := $(CURDIR)/.venv/share/jupyter

.PHONY: docs
docs: ## Build the documentation site
	@JUPYTER_PATH="$(JUPYTER_PATH)" .venv/bin/great-docs build

.PHONY: docs-preview
docs-preview: ## Preview the documentation site locally
	@JUPYTER_PATH="$(JUPYTER_PATH)" .venv/bin/great-docs preview

.PHONY: clean
clean: clean-build clean-pyc clean-test ## Remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## Remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

.PHONY: build
build: clean ## Build source and wheel distribution
	@$(PYTHON) -m build
	ls -l dist
