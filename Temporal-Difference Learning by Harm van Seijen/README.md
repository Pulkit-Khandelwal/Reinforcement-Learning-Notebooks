
An Empirical Evaluation of True Online TD(Lambda) by Harm van Seijen et. al
True Online Temporal-Difference Learning by Harm van Seijen et. al; 
-----Journal of Machine Learning Research 2016----

In this IPython Notebook, I will walk through the various Function Approximation methods for estimating an optimal solution for the value functions V(s) i.e. On-policy prediction. Refer to the above two papers and the chapters: 9 and 12 from Sutton and Barto's Book. I have implemented the following algorithms:

1. Gradient Monte Carlo Algorithm for Approximating V
2. Semi-gradient TD(0) for estimating V_pi
3. Semi-gradient TD(lambda)
4. True-online TD(lambda)
5. TD(0) for prediction. This result will act as the V_pi(s) to calculate RMSVE

We get introduced to lambda-return, eligibility traces and the dutch traces!
