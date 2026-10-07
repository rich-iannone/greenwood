# greenwood (R)

The R package of the Greenwood monorepo. It is at an early stage. For now it provides the response
model that the R and Python packages share: `Surv()` is `survival::Surv()`, re-exported unchanged.

The Python package in [`../python/`](../python/) implements the same function under the same name.
R Greenwood imports **survival** rather than reimplementing it. The Python package's `event_time()`
API (a port of [etd](https://github.com/topepo/etd)) has no R counterpart yet.

## Installation

```r
# install.packages("pak")
pak::pak("rich-iannone/greenwood/r")
```

## Shared tests

The contracts in [`../spec/`](../spec/) and the numeric fixtures in [`../fixtures/r/`](../fixtures/r/)
are generated in R by the scripts in [`../scripts/`](../scripts/). Both test suites replay them. The
R tests find them by walking up from the test directory (or from `GREENWOOD_REPO_ROOT`), and skip
when they aren't available. From the repository root:

```bash
make r-test     # devtools::test()
make r-check    # R CMD check
```
