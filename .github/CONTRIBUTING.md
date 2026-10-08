# Contributing to Greenwood

Thanks for your interest in improving Greenwood! This project is in active early
development (see [`ROADMAP.md`](../python/ROADMAP.md)). Contributions of code, tests, docs, and
R-parity fixtures are all welcome.

## Repository layout

Greenwood is an R and Python monorepo:

- `python/`: the Python package, its tests, and its Great Docs site (`great-docs.yml`,
`user_guide/`, `_freeze/`).
- `r/`: the R package.
- `spec/`: language-neutral contracts and their conformance cases, replayed by both test suites.
- `fixtures/r/`: numeric fixtures exported from R, replayed by both test suites.
- `scripts/`: the R scripts that generate `spec/conformance/` and `fixtures/r/`.

Run `make` targets from the repository root. Targets name their language (`make py-test`,
`make r-test`), and the bare names run both (`make test`, `make check`).

## Development setup

```bash
git clone https://github.com/rich-iannone/greenwood.git
cd greenwood
python -m venv .venv && source .venv/bin/activate
make py-install       # Python package with dev extras, into .venv
make r-deps           # the R package's dependencies (with pak)
pre-commit install    # optional but recommended
```

`make install` does both.

## The check gate

Before opening a pull request, make sure the full gate is green:

```bash
make check            # both packages: make py-check and make r-check
```

If you only changed one package, its gate is enough: `make py-check` (ruff, pyright, pytest) or
`make r-check` (R CMD check).

Individual targets include `make py-lint`, `make py-type-check`, `make py-test`,
`make py-test-rparity`, `make r-test`, and `make r-document`. Run `make help` for the full list.

## House conventions

- **R-validated numerics.** Correctness to tolerance against R's **survival** (and
`cmprsk`/`flexsurv`/`riskRegression`/`mstate` for specialized estimators) is the brand.
Regenerate fixtures from the repository root with `Rscript scripts/regenerate_r_fixtures.R`
(they are written to `fixtures/r/`) and validate with `make py-test-rparity` and `make r-test`.
- **Cross-language parity.** Package-level names stay identical across Python and R. Language
conventions follow each language, and differences are recorded as adaptations in `spec/`.
- **Narwhals-native.** Never assume Pandas but write data handling against Narwhals and drop
to NumPy only inside numeric kernels.
- **Typed & deterministic.** Full type hints (`py.typed`), `pyright` clean, byte-identical
output for identical inputs.
- **Prose style.**  Docstrings use Quarto/Markdown, not RST: single backticks for inline
code, numpydoc section headers (`Parameters`, `Returns`).
- Implementation lives in underscore-prefixed private modules and the public surface is
curated explicitly in `__init__.py`.

## Releases

The Python and R packages are versioned and released separately: Python with `v*` tags and R with
`r-v*` tags. See [`RELEASING.md`](../RELEASING.md).

## Pull requests

- Keep PRs focused and add tests for new behavior (target >=90% coverage).
- For new/changed statistics, add or update R-parity fixtures.
- Update docs (docstrings, `python/user_guide/`) alongside code.

By contributing you agree that your contributions are licensed under the MIT License.
