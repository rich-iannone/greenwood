#!/usr/bin/env Rscript
# Export the bundled datasets for the Python package.
#
# These back the R-parity harness and the docs examples. Run from the repo root:
#   Rscript scripts/export_datasets.R
#
# Outputs gzipped CSVs into python/greenwood/data/, with levels.json recording the levels of each
# factor column so load_dataset() can give them back as categoricals. Provenance: survival for
# lung, veteran, ovarian, pbc, pbcseq, colon, and mgus2, and smcure for e1684.

suppressPackageStartupMessages(library(survival))
if (!requireNamespace("smcure", quietly = TRUE)) {
  stop("smcure is required for e1684: install.packages('smcure')")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required: install.packages('jsonlite')")
}

out_dir <- Sys.getenv("GREENWOOD_DATA_DIR", file.path("python", "greenwood", "data"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

levels_of <- list()

write_gz <- function(df, name) {
  path <- file.path(out_dir, paste0(name, ".csv.gz"))
  con <- gzfile(path, "wt")
  utils::write.csv(df, con, row.names = FALSE, na = "")
  close(con)
  factors <- names(df)[vapply(df, is.factor, logical(1))]
  if (length(factors) > 0L) {
    levels_of[[name]] <<- stats::setNames(lapply(factors, function(f) levels(df[[f]])), factors)
  }
  cat(sprintf("wrote %-28s %d x %d\n", path, nrow(df), ncol(df)))
}

data(cancer, package = "survival") # provides lung, veteran, ovarian, pbc, pbcseq, colon, mgus2
write_gz(lung, "lung")
write_gz(veteran, "veteran")
write_gz(ovarian, "ovarian")
write_gz(pbc, "pbc")
write_gz(pbcseq, "pbcseq")
write_gz(colon, "colon")
write_gz(mgus2, "mgus2")

# Without the one row that is missing AGE and SEX.
data(e1684, package = "smcure", envir = environment())
write_gz(e1684[stats::complete.cases(e1684), ], "e1684")

levels_path <- file.path(out_dir, "levels.json")
writeLines(jsonlite::toJSON(levels_of, pretty = TRUE, auto_unbox = FALSE), levels_path)
cat(sprintf("wrote %s\n", levels_path))

cat("done\n")
