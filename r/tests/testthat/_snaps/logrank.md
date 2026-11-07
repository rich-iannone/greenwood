# results print and convert to data frames

    Code
      result
    Output
      <greenwood_test> Surv(time, status) ~ sex
      Log-rank test
      
            n observed expected
      sex=1 5        4 4.643651
      sex=2 5        4 3.356349
      
      Chi-squared = 0.219 on 1 degree of freedom, p = 0.64
    Code
      logrank_test(Surv(time, status) ~ sex + strata(ecog), data = small, rho = 1)
    Output
      <greenwood_test> Surv(time, status) ~ sex + strata(ecog)
      Stratified G-rho test (rho = 1, gamma = 0)
      
            n observed expected
      sex=1 5      2.4      3.2
      sex=2 5      3.2      2.4
      
      Chi-squared = 0.6667 on 1 degree of freedom, p = 0.414
    Code
      trend_test(Surv(time, status) ~ ecog, data = small)
    Output
      <greenwood_test> Surv(time, status) ~ ecog
      Log-rank test for trend
      
             n observed expected score
      ecog=0 5        4 4.318651     0
      ecog=1 5        4 3.681349     1
      
      Chi-squared = 0.05238 on 1 degree of freedom, p = 0.819
    Code
      combo
    Output
      <maxcombo_test> Surv(time, status) ~ sex
      MaxCombo test of (0, 0), (1, 0), (0, 1), (1, 1)
      
       rho gamma           z
         0     0 -0.46799721
         1     0 -0.62217102
         0     1 -0.07828573
         1     1 -0.34212866
      
      max |z| = 0.6222, p = 0.772

