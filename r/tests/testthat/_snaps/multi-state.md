# the printout is stable

    Code
      print(multi_state(Surv(time, event) ~ 1, data = small_cr))
    Output
      <multi_state> Surv(time, event) ~ 1
      States: (s0), pcm, death. n = 18
      
      Occupancy at the last time:
       state time estimate
        (s0)  240   0.0000
         pcm  240   0.4224
       death  240   0.5776

