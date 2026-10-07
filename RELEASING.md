# Releasing Greenwood

The Python and R packages are released independently, each with its own version and its own git
tags. A release is a GitHub release whose tag says which package it is for.

| Package | Tag | Example | Version comes from | Published by |
|---|---|---|---|---|
| Python | `v<version>` | `v0.9.0` | the tag (setuptools_scm) | CI, to PyPI |
| R | `r-v<version>` | `r-v0.1.0` | `r/DESCRIPTION` | you, to CRAN |

Python keeps the plain `v*` tags it has used since `v0.1.0`. Never create a `v*` tag for an R release
or an `r-v*` tag for a Python release. CI decides what to do from the tag prefix alone.

## Python

1. Make sure `main` is green.
2. Draft a GitHub release on `main` with a new tag `vX.Y.Z` and the release notes (the
   `release-notes` skill can draft them). Publish it.
3. CI (`.github/workflows/ci.yml`) runs lint and tests, checks that the version built from the tag is
   `X.Y.Z`, builds the sdist and wheel, and publishes them to PyPI.

The version is never written in a file. Between releases, builds get a development version such as
`0.9.1.dev3+g1234abc`. Only `v*` tags count (`tag_regex` and `describe_command` in
`python/pyproject.toml`), so R tags can't change the Python version.

## R

1. Set the release version in `r/DESCRIPTION` (for example `0.1.0`, replacing `0.0.0.9000`) and
   update `r/NEWS.md`.
2. Run `make r-check` locally, then `R CMD check --as-cran` on the built tarball, and the usual
   CRAN pre-checks (for example `devtools::check_win_devel("r")`).
3. Submit with `devtools::submit_cran("r")`.
4. Once CRAN accepts it, draft a GitHub release on the accepted commit with tag `r-vX.Y.Z` and
   publish it. CI (`.github/workflows/r-check.yml`) runs R CMD check, confirms the tag matches
   `DESCRIPTION`, and attaches the source tarball to the release.
5. Bump `r/DESCRIPTION` to the next development version (`X.Y.Z.9000`).

## Known gaps

- The Python docs site builds its changelog page from every GitHub release, so R releases will
  appear there too. Great Docs has no tag filter yet.
- The `release-notes` skill links releases as `releases/tag/v{VERSION}`, which is right for Python
  only.
