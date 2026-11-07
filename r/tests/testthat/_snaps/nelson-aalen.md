# a fit prints a summary per group

    Code
      nelson_aalen(Surv(time, status) ~ 1, data = small)
    Output
      <nelson_aalen> Surv(time, status) ~ 1
      
           n events max_cumhaz
      all 10      8   1.428968
    Code
      nelson_aalen(Surv(time, status) ~ sex, data = small)
    Output
      <nelson_aalen> Surv(time, status) ~ sex
      
            n events max_cumhaz
      sex=1 5      4   1.283333
      sex=2 5      4   1.283333

