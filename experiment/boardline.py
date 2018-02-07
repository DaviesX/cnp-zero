import numpy as np
from scipy.special import comb


def theoretical(k, e, s):
    pass


def empirical(k, e, s, N=10000):
    pass


print("Theoretical result: ")
pmf, E = theoretical(10, 3, 0.8)
print(pmf)
print(E)

print("Empirical result: ")
pmf, E = empirical(10, 3, 0.8)
print(pmf)
print(E)
