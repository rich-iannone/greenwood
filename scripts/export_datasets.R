#!/usr/bin/env Rscript
# Export the bundled survival datasets from R's `survival` package.
#
# These back the R-parity harness and the docs examples. Run from the repo root:
#   Rscript scripts/export_datasets.R
#
# Outputs gzipped CSVs into greenwood/data/. Provenance (all survival::):
#   lung, veteran, ovarian, pbc, colon.

suppressPackageStartupMessages(library(survival))

out_dir <- file.path("greenwood", "data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_gz <- function(df, name) {
  path <- file.path(out_dir, paste0(name, ".csv.gz"))
  con <- gzfile(path, "wt")
  utils::write.csv(df, con, row.names = FALSE, na = "")
  close(con)
  cat(sprintf("wrote %-28s %d x %d\n", path, nrow(df), ncol(df)))
}

data(cancer, package = "survival") # provides lung
write_gz(lung, "lung")

data(veteran, package = "survival")
write_gz(veteran, "veteran")

data(ovarian, package = "survival")
write_gz(ovarian, "ovarian")

data(pbc, package = "survival")
write_gz(pbc, "pbc")

data(colon, package = "survival")
write_gz(colon, "colon")

data(mgus2, package = "survival")
write_gz(mgus2, "mgus2")

cat("done\n")
