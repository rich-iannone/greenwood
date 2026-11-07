# the printout is stable

    Code
      print(parametric(Surv(time, status) ~ 1, data = small_cox))
    Output
      <parametric> Surv(time, status) ~ 1
      Weibull distribution. n = 30, events = 28
      
       param estimate std_error conf_low conf_high
       shape    1.279    0.1993   0.9423     1.736
       scale  455.306   69.0994 338.1587   613.035
      
      Log-likelihood: -196.873, AIC: 397.747

---

    Code
      print(parametric(Surv(time, status) ~ sex, data = small_cox, dist = "lognormal"))
    Output
      <parametric> Surv(time, status) ~ sex
      Log-normal distribution. n = 30, events = 28
      
       strata sex param estimate std_error conf_low conf_high
        sex=1   1    mu   5.7035    0.2282   5.2563     6.151
        sex=1   1 sigma   1.0843    0.1708   0.7963     1.476
        sex=2   2    mu   5.6369    0.3485   4.9538     6.320
        sex=2   2 sigma   0.9221    0.2464   0.5461     1.557
      
      Log-likelihood: -198.813, AIC: 405.627

