# Greenwood monorepo: the Python package lives in python/ and the R package in r/. Both read the
# shared contracts and conformance cases in spec/ and the R-generated numeric fixtures in
# fixtures/r/. Targets name their language (`py-*`, `r-*`), and the bare names (`test`, `check`,
# `install`) run both.

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

# -- Both ---------------------------------------------------------------------------------------

.PHONY: install
install: py-install r-deps ## Install the Python package (dev extras) and the R package's dependencies

.PHONY: test
test: py-test r-test ## Run the Python and R test suites

.PHONY: check
check: py-check r-check ## Run the Python gate and R CMD check (the pre-push gate)

# -- Python -------------------------------------------------------------------------------------

# The venv may not have pip (uv creates it without), so prefer uv when it's available.
.PHONY: py-install
py-install: ## Install the Python package with dev extras into .venv
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python "$(PYTHON)" -e "$(PY_DIR)[dev]"; \
	else \
		"$(PYTHON)" -m pip install -e "$(PY_DIR)[dev]"; \
	fi

.PHONY: py-test
py-test: ## Run the full Python test suite with coverage
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests \
		-n auto \
		--cov=greenwood \
		--cov-report=term-missing \
		--durations 10

.PHONY: py-test-unit
py-test-unit: ## Run Python tests excluding slow and R-parity suites
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests -n auto -m "not slow and not rparity" --durations 10

.PHONY: py-test-rparity
py-test-rparity: ## Run the Python R-parity numeric validation suite
	@cd $(PY_DIR) && $(PYTHON) -m pytest tests -n auto -m rparity --durations 10

.PHONY: py-lint
py-lint: ## Run ruff formatter and linter (with fixes)
	@cd $(PY_DIR) && $(PYTHON) -m ruff format
	@cd $(PY_DIR) && $(PYTHON) -m ruff check --fix

.PHONY: py-check-format
py-check-format: ## Check Python formatting and lint without making changes
	@cd $(PY_DIR) && $(PYTHON) -m ruff format --check
	@cd $(PY_DIR) && $(PYTHON) -m ruff check

.PHONY: py-type-check
py-type-check: ## Run pyright in strict mode
	@cd $(PY_DIR) && $(PYTHON) -m pyright greenwood

.PHONY: py-check
py-check: py-lint py-type-check py-test ## Run all Python checks: ruff, pyright, pytest

# The user-guide pages contain executable code, so Quarto needs the project venv's
# Jupyter kernel (which has greenwood and its dependencies installed).
JUPYTER_PATH := $(CURDIR)/.venv/share/jupyter

# Number of Quarto processes that render pages at once (a number, or `auto` for one per CPU).
# Override with `make py-docs DOCS_JOBS=1` to render one page at a time.
DOCS_JOBS ?= auto

# CI builds with Great Docs from GitHub main, so install the same version locally. An older copy
# in .venv renders the pages differently and rewrites the committed _freeze/ output.
.PHONY: py-docs-install
py-docs-install: ## Reinstall Great Docs from GitHub main into .venv (matches CI)
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python "$(PYTHON)" --reinstall "git+https://github.com/posit-dev/great-docs.git@main"; \
	else \
		"$(PYTHON)" -m pip install --force-reinstall "git+https://github.com/posit-dev/great-docs.git@main"; \
	fi

.PHONY: py-docs
py-docs: ## Build the Python documentation site
	@cd $(PY_DIR) && JUPYTER_PATH="$(JUPYTER_PATH)" $(CURDIR)/.venv/bin/great-docs build --jobs $(DOCS_JOBS)

# `preview` builds the site first when it hasn't been built, and reads the job count from the
# environment for that build.
.PHONY: py-docs-preview
py-docs-preview: ## Preview the Python documentation site locally
	@cd $(PY_DIR) && JUPYTER_PATH="$(JUPYTER_PATH)" GREAT_DOCS_RENDER_JOBS=$(DOCS_JOBS) $(CURDIR)/.venv/bin/great-docs preview

.PHONY: py-build
py-build: clean ## Build the Python source and wheel distribution
	@cd $(PY_DIR) && $(PYTHON) -m build
	ls -l $(PY_DIR)/dist

# -- R ------------------------------------------------------------------------------------------

.PHONY: r-deps
r-deps: ## Install the R package's dependencies, including Suggests
	@$(RSCRIPT) -e 'pak::local_install_deps("$(R_DIR)", dependencies = TRUE, upgrade = FALSE, ask = FALSE)'

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

# -- Cleanup ------------------------------------------------------------------------------------

.PHONY: clean
clean: clean-build clean-pyc clean-test ## Remove build, test, and coverage artifacts for both packages

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
