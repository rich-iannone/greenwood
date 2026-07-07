# Contributing to Greenwood

Thanks for your interest in improving Greenwood! This project is in active early
development (see [`ROADMAP.md`](../ROADMAP.md)). Contributions of code, tests, docs, and
R-parity fixtures are all welcome.

## Development setup

```bash
git clone https://github.com/rich-iannone/greenwood.git
cd greenwood
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]"
pre-commit install    # optional but recommended
```

## The check gate

Before opening a pull request, make sure the full gate is green:

```bash
make check            # ruff format + ruff check + pyright + pytest
```

Individual targets: `make lint`, `make type-check`, `make test`, `make test-rparity`.
Run `make help` for the full list.

## House conventions

- **R-validated numerics.** Correctness to tolerance against R's `survival` (and
`cmprsk`/`flexsurv`/`riskRegression`/`mstate` for specialized estimators) is the brand.
Regenerate fixtures with `Rscript scripts/regenerate_r_fixtures.R` and validate with
`make test-rparity`.
- **Narwhals-native.** Never assume pandas; write data handling against Narwhals and drop
to NumPy only inside numeric kernels.
- **Typed & deterministic.** Full type hints (`py.typed`), `pyright` clean, byte-identical
output for identical inputs.
- **Prose style.**  Docstrings use Quarto/Markdown, not RST: single backticks for inline
code, numpydoc section headers (`Parameters`, `Returns`).
- Implementation lives in underscore-prefixed private modules; the public surface is
curated explicitly in `__init__.py`.

## Pull requests

- Keep PRs focused and add tests for new behavior (target >=90% coverage).
- For new/changed statistics, add or update R-parity fixtures.
- Update docs (docstrings, `user_guide/`) alongside code.

By contributing you agree that your contributions are licensed under the MIT License.
