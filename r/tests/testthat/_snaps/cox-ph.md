# the printouts are stable

    Code
      fit
    Output
      <cox_ph> Surv(time, status) ~ age + sex
      Efron ties. n = 30, events = 28
      
          estimate hazard_ratio std_error statistic p_value
      age -0.02049       0.9797   0.02899   -0.7068  0.4797
      sex  0.24290       1.2749   0.48274    0.5032  0.6148
      
      Likelihood ratio test: 0.568 on 2 df, p = 0.753
      Wald test:             0.555 on 2 df, p = 0.758
      Score test:            0.556 on 2 df, p = 0.757
      Concordance: 0.498 (std. error 0.0742)
    Code
      cox_zph(fit, transform = "identity", breaks = 300)
    Output
      <cox_zph> Surv(time, status) ~ age + sex
      Transform: identity
      
             statistic df p_value
      age       0.1314  1  0.7170
      sex       0.2937  1  0.5879
      GLOBAL    0.3332  2  0.8465
      
      By window:
           window n_event   term statistic df p_value
         (0, 300]      11    age    1.6700  1  0.1963
         (0, 300]      11    sex    1.2163  1  0.2701
         (0, 300]      11 GLOBAL    2.1955  2  0.3336
       (300, Inf]      17    age    1.0655  1  0.3020
       (300, Inf]      17    sex    0.1339  1  0.7144
       (300, Inf]      17 GLOBAL    1.7204  2  0.4231

