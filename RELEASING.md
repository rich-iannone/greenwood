# Releasing Greenwood

The Python and R packages are released independently, each with its own version and its own git
tags. A release is a GitHub release whose tag says which package it is for.

| Package | Tag | Example | Version comes from | Published to |
|---|---|---|---|---|
| Python | `v<version>` | `v0.9.0` | the tag (setuptools_scm) | PyPI, by CI |
| R | `r-v<version>` | `r-v0.1.0` | `r/DESCRIPTION` | CRAN, by submission |

Python keeps the plain `v*` tags it has used since `v0.1.0`. The tag prefix matters: publishing a
release with a `v*` tag starts the PyPI pipeline, so an R release must always use `r-v*`.

## Python

1. Open a release PR with the release notes and anything else the release needs, and merge it once
   CI is green.
2. Publish a GitHub release on `main` with a new tag `vX.Y.Z` (the `release-notes` skill can draft
   the notes).
3. CI (`.github/workflows/ci.yml`) runs lint and tests, checks that the version built from the tag is
   `X.Y.Z`, builds the sdist and wheel, and publishes them to PyPI.

The version is never written in a file. Between releases, builds get a development version such as
`0.9.1.dev3+g1234abc`. Only `v*` tags count (`tag_regex` and `describe_command` in
`python/pyproject.toml`), so R tags can't change the Python version.

## R

1. Open a release PR that sets the release version in `r/DESCRIPTION` (for example `0.1.0`,
   replacing `0.0.0.9000`) and updates `r/NEWS.md`. Run `make r-check` and the usual CRAN
   pre-checks, then merge it once CI is green.
2. Submit to CRAN.
3. Publish a GitHub release on `main` with tag `r-vX.Y.Z`, using the `NEWS.md` entry as the notes.
   No CI runs for it.
4. Bump `r/DESCRIPTION` to the next development version (`X.Y.Z.9000`).

## Known gaps

- The Python docs site builds its changelog page from every GitHub release, so R releases will
  appear there too. Great Docs has no tag filter yet.
- The `release-notes` skill links releases as `releases/tag/v{VERSION}`, which is right for Python
  only.
