# Small datasets shared by the test files. The first ten rows of lung, with a made-up ecog column.
small <- data.frame(
  time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166),
  status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
  sex = c(1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
  ecog = c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
)

# The first 30 rows of lung, for the Cox model tests.
small_cox <- data.frame(
  time = c(
    306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144, 613, 707,
    61, 88, 301, 81, 624, 371, 394, 520, 574, 118, 390, 12
  ),
  status = c(2, 2, 1, 2, 2, 1, rep(2, 24)),
  age = c(
    74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57, 67, 49, 50,
    58, 72, 70, 60, 70, 53, 74
  ),
  sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1)
)

# mgus2 with the time and type of the first event: 1 for progression (pcm), 2 for death, and 0 for
# censored, as in the fixture generators.
mgus2_competing <- function() {
  mg <- survival::mgus2
  mg$etime <- ifelse(mg$pstat == 1, mg$ptime, mg$futime)
  mg$event <- ifelse(mg$pstat == 1, 1L, 2L * mg$death)
  mg$event_f <- factor(mg$event, 0:2, c("censor", "pcm", "death"))
  mg
}

# A small competing-risks dataset made up for the behavior tests.
small_cr <- data.frame(
  time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240, 45, 95),
  status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 0, 1),
  age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74, 70, 61),
  sex = rep(c("F", "M"), times = 9)
)
small_cr$event <- factor(small_cr$status, levels = 0:2, labels = c("censor", "pcm", "death"))
