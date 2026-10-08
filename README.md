<p align="center">
<a href="https://rich-iannone.github.io/greenwood/">
<img src="https://rich-iannone.github.io/greenwood/assets/logo.png" alt="Greenwood" width="350">
</a>
</p>
<p align="center">Modern survival analysis for Python and R: R-validated, one response model across both languages.</p>
<p align="center">
<a href="https://pypi.org/project/greenwood/"><img src="https://img.shields.io/pypi/v/greenwood?logo=python&logoColor=white&color=orange" alt="PyPI"></a>
<a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
<a href="https://github.com/rich-iannone/greenwood/actions/workflows/ci.yml"><img src="https://github.com/rich-iannone/greenwood/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
<a href="https://github.com/rich-iannone/greenwood/actions/workflows/r-check.yml"><img src="https://github.com/rich-iannone/greenwood/actions/workflows/r-check.yml/badge.svg?branch=main" alt="R-CMD-check"></a>
<a href="https://rich-iannone.github.io/greenwood/"><img src="https://img.shields.io/badge/docs-project_website-blue.svg" alt="Documentation"></a>
</p>

---

Greenwood is a survival analysis library for the study of time-to-event outcomes. This repository
holds both language implementations:

| Package | Directory | Status |
|---|---|---|
| Python | [`python/`](python/) | Feature-rich. See the [Python README](python/README.md) and the [documentation site](https://rich-iannone.github.io/greenwood/). |
| R | [`r/`](r/) | Early. `Surv()` and `kaplan_meier()`, held to the same contracts as Python. See the [R README](r/README.md). |

## Shared across languages

Both packages are native implementations, validated against R's **survival** package. They replay
the same cases, so they agree on inputs and results.

- [`spec/`](spec/) holds the language-neutral contracts (`Surv()`, `event_time()`, formulas,
  Kaplan-Meier) and their conformance cases as JSON.
- [`fixtures/r/`](fixtures/r/) holds numeric fixtures exported from R for the parity tests.
- [`scripts/`](scripts/) holds the R scripts that generate both.

The R package replays the `Surv()` conformance cases and the Kaplan-Meier fixtures so far. The
Python package replays all of them.

Shared functions keep the same names in both languages (`Surv()`, metrics). Estimators follow each
language's idiom: classes in Python (`KaplanMeier`) and lower snake case functions returning S3
objects in R (`kaplan_meier()`). Language-level conventions follow each language (for example, 0-based error locations and `NaN`/`None` in Python), and any differences are
recorded as adaptations in `spec/`.

## Development

From the repository root:

```bash
make install    # Python package (dev extras, into .venv) and R dependencies
make test       # both test suites (make py-test, make r-test)
make check      # both gates: ruff, pyright, pytest, and R CMD check
make py-check   # one package only (or make r-check)
```

Run `make help` for every target, and see [`CONTRIBUTING.md`](.github/CONTRIBUTING.md).

## License

MIT (c) Richard Iannone. See the [`LICENSE`](LICENSE) file.
