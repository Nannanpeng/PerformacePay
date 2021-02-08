import numpy as np
from utilities import Utility
from scipy import optimize


def LastPeriodUtility(a, transfer, params):
    return a*params["r"] + transfer


def RetiredUtility(a, c, NextUtilityFun, UtilityParams, transfer, params):
    beta = params["beta"]
    r = params["r"]
    a_prime = a*(1+r) - c + transfer
    next_utility = NextUtilityFun(a_prime)
    total_utility = Utility(c) + beta*next_utility # need more arguments for Utility function
    return total_utility


# To do: refactor, create class for it and find best optimizer
def RetiredOptimalConsumption(a, ReUtility, NextUtilityFun, UtilityParams, transfer, params):
    ReUtilityFun = lambda c: ReUtility(a, c, NextUtilityFun, UtilityParams, transfer, params)
    consumption, utility, _, _ = optimize.brent(ReUtilityFun, brack=(0, a), full_output=True)
    return consumption, utility


# no work
def LastWorkingPeriodUtility(a, c, NextUtilityFun, UtilityParams, transfer, params):
    beta = UtilityParams["beta"]
    b = params["b"]
    r = params["r"]
    a_prime = a*(1 + r) + transfer - c
    total_utility = Utility(c, 0, UtilityParams) + b + beta*NextUtilityFun(a_prime)
    return total_utility


# No work
def LastWorkingPeriodConsumption(a, b, ThisUtility, NextUtilityFun, UtilityParams, transfer, params):
    ThisUtilityFun = lambda c: ThisUtility(a, c, NextUtilityFun, UtilityParams, transfer, params)
    consumption, utility, _, _ = optimize.brent(ThisUtilityFun, brack=(0, a), full_output=True)
    return consumption, utility

 

