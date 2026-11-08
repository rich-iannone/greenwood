# the printout is stable

    Code
      print(fit)
    Output
      <aft> Surv(time, status) ~ age + sex
      Weibull distribution. n = 30, events = 28
      
                  estimate time_ratio std_error statistic   p_value
      (Intercept)  5.27851   196.0777   1.28631    4.1036 4.067e-05
      age          0.01829     1.0185   0.02225    0.8219 4.111e-01
      sex         -0.25189     0.7773   0.37103   -0.6789 4.972e-01
      
      Scale: 0.773 (std. error of log(scale) 0.156)
      Log-likelihood: -196.456
      Likelihood ratio test: 0.834 on 2 df, p = 0.659

---

    Code
      print(aft(Surv(time, status) ~ age, data = small_cox, dist = "exponential"))
    Output
      <aft> Surv(time, status) ~ age
      Exponential distribution. n = 30, events = 28
      
                  estimate time_ratio std_error statistic  p_value
      (Intercept)  5.15422    173.161     1.643    3.1362 0.001712
      age          0.01452      1.015     0.026    0.5582 0.576675
      
      Log-likelihood: -197.837
      Likelihood ratio test: 0.31 on 1 df, p = 0.577

---

    Code
      print(aft(Surv(time, status) ~ age, data = small_cox, dist = "gengamma"))
    Output
      <aft> Surv(time, status) ~ age
      Generalized gamma distribution. n = 30, events = 28
      
                  estimate time_ratio std_error statistic   p_value
      (Intercept)  5.36221    213.195   1.35475    3.9581 7.555e-05
      age          0.01231      1.012   0.02044    0.6024 5.469e-01
      
      Scale: 0.767 (std. error of log(scale) 0.234)
      Q: 1.05 (std. error 0.588)
      Log-likelihood: -196.675
      Likelihood ratio test: 0.374 on 1 df, p = 0.541

