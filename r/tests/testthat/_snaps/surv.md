# Surv objects format and print compactly

    Code
      Surv(c(5, 6, NA), c(1, 0, 1))
    Output
      <Surv[3]: right>
      [1] 5    6+   <NA>
    Code
      Surv(c(5, 6), c(1, 0), type = "left")
    Output
      <Surv[2]: left>
      [1] 5  6-
    Code
      Surv(c(0, 2), c(5, 6), c(1, 0))
    Output
      <Surv[2]: counting>
      [1] (0,5]  (2,6+]
    Code
      Surv(c(5, 6, 4, 2), c(NA, NA, NA, 4), c(0, 1, 2, 3), type = "interval")
    Output
      <Surv[4]: interval>
      [1] 5+     6      4-     [2, 4]
    Code
      Surv(c(5, 6, 7), factor(c("censor", "pcm", "death"), levels = c("censor", "pcm",
        "death")))
    Output
      <Surv[3]: mright> states: pcm, death
      [1] 5+      6:pcm   7:death
    Code
      Surv(numeric(), numeric())
    Output
      <Surv[0]: right>

