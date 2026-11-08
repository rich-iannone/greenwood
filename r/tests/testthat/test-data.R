# The bundled datasets: their shapes, their codings, and the same values as the Python package's
# copies in python/greenwood/data/.

DATASETS <- list(
  lung = c(228L, 10L),
  veteran = c(137L, 8L),
  ovarian = c(26L, 6L),
  pbc = c(418L, 20L),
  pbcseq = c(1945L, 19L),
  colon = c(1858L, 16L),
  mgus2 = c(1384L, 11L),
  e1684 = c(284L, 5L)
)

get_dataset <- function(name) {
  getExportedValue("greenwood", name)
}

test_that("every dataset has its documented shape", {
  for (name in names(DATASETS)) {
    df <- get_dataset(name)
    expect_s3_class(df, "data.frame")
    expect_identical(dim(df), DATASETS[[name]], label = name)
    expect_identical(rownames(df), as.character(seq_len(nrow(df))), label = name)
  }
  expect_setequal(
    utils::data(package = "greenwood")$results[, "Item"],
    names(DATASETS)
  )
})

test_that("the codings are the documented ones", {
  expect_setequal(unique(lung$status), c(1, 2))
  expect_setequal(unique(veteran$status), c(0, 1))
  expect_identical(levels(veteran$celltype), c("squamous", "smallcell", "adeno", "large"))
  expect_setequal(unique(pbc$status), 0:2)
  expect_identical(levels(pbc$sex), c("m", "f"))
  expect_identical(levels(colon$rx), c("Obs", "Lev", "Lev+5FU"))
  expect_identical(levels(mgus2$sex), c("F", "M"))
  expect_false(anyNA(e1684))
  expect_true(all(pbcseq$id %in% pbc$id))
  # No column carries a leftover label.
  for (name in names(DATASETS)) {
    for (column in get_dataset(name)) {
      expect_null(attr(column, "label"))
    }
  }
})

test_that("the datasets hold the values of the Python package's copies", {
  root <- repo_root()
  skip_if(is.null(root), "The Python package's datasets are not available.")
  for (name in names(DATASETS)) {
    path <- file.path(root, "python", "greenwood", "data", paste0(name, ".csv.gz"))
    csv <- utils::read.csv(gzfile(path), na.strings = c("", "NA"), stringsAsFactors = FALSE)
    df <- get_dataset(name)
    expect_named(df, names(csv))
    for (column in names(df)) {
      ours <- df[[column]]
      if (is.factor(ours)) {
        expect_identical(as.character(ours), csv[[column]], label = paste(name, column))
      } else {
        expect_equal(as.numeric(ours), as.numeric(csv[[column]]), label = paste(name, column))
      }
    }
  }
})

test_that("the factors have the levels of the Python package's categoricals", {
  root <- repo_root()
  skip_if(is.null(root), "The Python package's datasets are not available.")
  skip_if_not_installed("jsonlite")
  path <- file.path(root, "python", "greenwood", "data", "levels.json")
  recorded <- jsonlite::read_json(path, simplifyVector = TRUE)
  for (name in names(DATASETS)) {
    df <- get_dataset(name)
    factors <- names(df)[vapply(df, is.factor, logical(1))]
    expect_setequal(as.character(names(recorded[[name]])), factors)
    for (column in factors) {
      expect_identical(
        levels(df[[column]]), recorded[[name]][[column]],
        label = paste(name, column)
      )
    }
  }
})

test_that("the datasets fit as the documented responses", {
  fit <- kaplan_meier(Surv(time, status) ~ sex, data = lung)
  expect_identical(glance(fit)$events, c(112, 53))
  aj <- aalen_johansen(
    Surv(time, factor(status, 0:2, c("censor", "transplant", "death"))) ~ 1,
    data = pbc
  )
  expect_identical(glance(aj)$causes, "transplant, death")
})
