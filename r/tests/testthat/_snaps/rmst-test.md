# a comparison prints

    Code
      rmst_test(Surv(time, status) ~ sex, data = small, tau = 500)
    Output
      <rmst_test> Surv(time, status) ~ sex
      RMST difference (tau = 500)
      
             rmst std_error
      sex=1 394.2  52.07410
      sex=2 311.0  52.11372
      
      Difference (1 vs 2): 83.2, 95% CI [-61.19, 227.6]
      z = 1.129, p = 0.259
    Code
      rmst_test(Surv(time, status) ~ sex + strata(ecog), data = small, tau = 500,
      estimand = "ratio")
    Output
      <rmst_test> Surv(time, status) ~ sex + strata(ecog)
      Stratified RMST ratio (tau = 500)
      
                rmst std_error
      sex=1 412.8687  45.39737
      sex=2 281.1125  30.73722
      
      Ratio (1 vs 2): 1.469, 95% CI [1.084, 1.99]
      z = 2.479, p = 0.0132

