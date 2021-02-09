from utilities import ConUtility
from scipy import optimize


def LastPeriodUtility(a, *args, transfer=0.0):
    ChiC, iota, r = args
    c = a*(1+r) + transfer
    utility = ConUtility(c, ChiC, iota)
    return utility


# NextUtility is a callable function
def RetiredUtility(c, a, NextUtility, *args, transfer=0.0):
    ChiC, iota, r, beta = args
    a_prime = a*(1+r) - c + transfer
    next_utility = NextUtility(a_prime)
    total_utility = ConUtility(c, ChiC, iota) + beta*next_utility
    return total_utility


# To do: refactor, create class for it and find best optimizer
def RetiredOptimalCon(a, NextUtility, *args, transfer=0.0):
    ChiC, iota, r, beta, B = args
    add_args = (ChiC, iota, r, beta)
    consumption, utility, _, _ = optimize.brent(
        RetiredUtility, add_args, brack=(B, a), full_output=True)
    return consumption, utility
