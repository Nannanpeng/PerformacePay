import numpy as np
from utilities import Utility
from scipy import optimize

print(3)
def LastPeriodUtility(a, transfer, params):
    return a*params["r"] + transfer


def RetiredUtility(a, c, NextUtilityFun, UtilityParams, transfer, params):
    beta = params["beta"]
    r = params["r"]
    a_prime = a*(1+r) - c + transfer
    next_utility = NextUtilityFun(a_prime)
    total_utility = Utility(c) + beta*next_utility
    return total_utility


# To do: refactor, create class for it and find best optimizer
def RetiredOptimalConsumption(a, ReUtility, NextUtilityFun, UtilityParams, transfer, params):
    ReUtilityFun = lambda c: ReUtility(a, c, NextUtilityFun, UtilityParams, transfer, params)
    consumption, utility, _, _ = optimize.brent(ReUtilityFun, brack=(0, a), full_output=True)
    return consumption, utility


def f(x):
    return x**2


outputs = optimize.brent(f, brack=(1, 2), full_output=True)
print(outputs)
