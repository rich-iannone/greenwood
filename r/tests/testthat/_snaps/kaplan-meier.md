# a fit prints a summary per group

    Code
      kaplan_meier(Surv(time, status) ~ 1, data = small)
    Output
      <kaplan_meier> Surv(time, status) ~ 1
      
        n events median 0.95 LCL 0.95 UCL
       10      8    310      218       NA
    Code
      kaplan_meier(Surv(time, status) ~ sex, data = small)
    Output
      <kaplan_meier> Surv(time, status) ~ sex
      
            n events median 0.95 LCL 0.95 UCL
      sex=1 5      4    455      306       NA
      sex=2 5      4    310      218       NA

