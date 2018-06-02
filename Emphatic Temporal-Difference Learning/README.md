An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning---- Richard S. Sutton, A. Rupam Mahmood and Martha White

In this IPython Notebook, I will walk through the various Function Approximation methods for estimating an optimal solution for the value functions V(s) given in the above mentioned paper. We get introduced to a different type of trace: the followon trace, interest function and also a new variant of Squared Value error with interest function!
The algorithms implemented in this notebook are:

1. Emphatic TD(lambda)
2. Emphatic TD(0)
3. Off-policy Semi-gradient TD(0) for estimating V_pi

The above alogorithms are compared with the following algorithms from the previous assignment 4:

1. Gradient Monte Carlo Algorithm for Approximating V
2. Semi-gradient TD(0) for estimating V_pi
3. TD(lambda)
4. True-online TD(lambda)
5. TD(0) for prediction. This result will act as the V_pi(s) to calculate RMSVE
