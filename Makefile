# Greenwood monorepo: the Python package lives in python/ and the R package in r/. Both read the
# shared contracts and conformance cases in spec/ and the R-generated numeric fixtures in
# fixtures/r/. Python targets keep their original names, and R targets are prefixed with `r-`.

PYTHON ?= $(CURDIR)/.venv/bin/python
RSCRIPT ?= Rscript
PY_DIR := python
R_DIR := r

.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -- Python -------------------------------------------------------------------------------------

.PHONY: install
install: ## Install the Python package with dev extras into .venv
	@$(PYTHON) -m pip install -e "$(PY_DIR)[dev]"

.PHONY: test
test: ## Run the full Python test suite with coverage
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests \
		-n auto \
		--cov=greenwood \
		--cov-report=term-missing \
		--durations 10

.PHONY: test-unit
test-unit: ## Run Python tests excluding slow and R-parity suites
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests -n auto -m "not slow and not rparity" --durations 10

.PHONY: test-rparity
test-rparity: ## Run the Python R-parity numeric validation suite
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests -n auto -m rparity --durations 10

.PHONY: lint
lint: ## Run ruff formatter and linter (with fixes)
	@cd $(PY_DIR) && $(PYTHON) -m ruff format
	@cd $(PY_DIR) && $(PYTHON) -m ruff check --fix

.PHONY: check-format
check-format: ## Check formatting and lint without making changes
	@cd $(PY_DIR) && $(PYTHON) -m ruff format --check
	@cd $(PY_DIR) && $(PYTHON) -m ruff check

.PHONY: type-check
type-check: ## Run pyright in strict mode
	@cd $(PY_DIR) && $(PYTHON) -m pyright greenwood

.PHONY: check
check: lint type-check test ## Run all Python checks (the pre-push gate)

# The user-guide pages contain executable code, so Quarto needs the project venv's
# Jupyter kernel (which has greenwood and its dependencies installed).
JUPYTER_PATH := $(CURDIR)/.venv/share/jupyter

.PHONY: docs
docs: ## Build the Python documentation site
	@cd $(PY_DIR) && JUPYTER_PATH="$(JUPYTER_PATH)" $(CURDIR)/.venv/bin/great-docs build

.PHONY: docs-preview
docs-preview: ## Preview the Python documentation site locally
	@cd $(PY_DIR) && JUPYTER_PATH="$(JUPYTER_PATH)" $(CURDIR)/.venv/bin/great-docs preview

.PHONY: build
build: clean ## Build the Python source and wheel distribution
	@cd $(PY_DIR) && $(PYTHON) -m build
	ls -l $(PY_DIR)/dist

# -- R ------------------------------------------------------------------------------------------

.PHONY: r-install
r-install: ## Install the R package
	@R CMD INSTALL $(R_DIR)

.PHONY: r-document
r-document: ## Regenerate the R package's NAMESPACE and man/ with roxygen2
	@$(RSCRIPT) -e 'devtools::document("$(R_DIR)")'

.PHONY: r-test
r-test: ## Run the R package's testthat suite
	@$(RSCRIPT) -e 'devtools::test("$(R_DIR)", stop_on_failure = TRUE)'

.PHONY: r-check
r-check: ## Run R CMD check on the R package
	@$(RSCRIPT) -e 'devtools::check("$(R_DIR)", document = FALSE, manual = FALSE, error_on = "warning", check_dir = "$(R_DIR)/check")'

# -- Both ---------------------------------------------------------------------------------------

.PHONY: check-all
check-all: check r-check ## Run the Python gate and R CMD check

.PHONY: clean
clean: clean-build clean-pyc clean-test ## Remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: ## Remove build artifacts
	rm -fr $(PY_DIR)/build/
	rm -fr $(PY_DIR)/dist/
	rm -fr $(PY_DIR)/.eggs/
	rm -fr $(R_DIR)/greenwood.Rcheck/ $(R_DIR)/check/
	rm -f $(R_DIR)/greenwood_*.tar.gz
	find . -name '*.egg-info' -not -path './.venv/*' -exec rm -fr {} +
	find . -name '*.egg' -not -path './.venv/*' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -not -path './.venv/*' -exec rm -f {} +
	find . -name '*.pyo' -not -path './.venv/*' -exec rm -f {} +
	find . -name '*~' -not -path './.venv/*' -exec rm -f {} +
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## Remove test and coverage artifacts
	rm -f $(PY_DIR)/.coverage
	rm -fr $(PY_DIR)/htmlcov/
	rm -fr $(PY_DIR)/.pytest_cache
