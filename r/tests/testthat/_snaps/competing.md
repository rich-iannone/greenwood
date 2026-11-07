# the printouts are stable

    Code
      print(aalen_johansen(Surv(time, event) ~ 1, data = small_cr))
    Output
      <aalen_johansen> Surv(time, event) ~ 1
      n = 18. Events: pcm = 6, death = 7
      
      Cumulative incidence at the last time (plain limits):
       cause time estimate std_error conf_low conf_high
         pcm  240   0.4224    0.1332   0.1613    0.6835
       death  240   0.5776    0.1332   0.3165    0.8387

---

    Code
      print(aalen_johansen(Surv(time, event) ~ sex, data = small_cr))
    Output
      <aalen_johansen> Surv(time, event) ~ sex
      n = 18. Events: pcm = 6, death = 7
      
      Cumulative incidence at the last time (plain limits):
       strata sex cause time estimate std_error conf_low conf_high
        sex=F   F   pcm  220   0.5370    0.2348  0.07683    0.9972
        sex=F   F death  220   0.1852    0.1653  0.00000    0.5093
        sex=M   M   pcm  240   0.3333    0.1571  0.02535    0.6413
        sex=M   M death  240   0.6667    0.1571  0.35869    0.9746

---

    Code
      print(grays_test(Surv(time, event) ~ sex, data = small_cr, cause = "death"))
    Output
      <greenwood_test> Surv(time, event) ~ sex
      Gray's test (cause = "death")
      
            n observed expected
      sex=F 9        1 2.715045
      sex=M 9        6 4.284955
      
      Chi-squared = 1.958 on 1 degree of freedom, p = 0.162

---

    Code
      print(fine_gray(Surv(time, event) ~ age + sex, data = small_cr))
    Output
      <fine_gray> Surv(time, event) ~ age + sex
      Subdistribution hazards for cause "pcm". n = 18, events = 6
      
           estimate hazard_ratio robust_se statistic p_value
      age   -0.1763       0.8383   0.08178    -2.156 0.03108
      sexM   1.4478       4.2538   1.44501     1.002 0.31637
      
      Robust Wald test: 11.2 on 2 df, p = 0.00371

