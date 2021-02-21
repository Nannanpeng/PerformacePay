# This file solves retired problem 
from utilities import ConUtility
import numpy as np
from interpolation import interp
import matplotlib.pyplot as plt
from numba import njit
from quantecon.optimize.scalar_maximization import brent_max


@njit
def LastPeriodUtility(a, ChiC, iota, r, transfer=0.0):
    c = a*(1+r) + transfer
    utility = ConUtility(c, ChiC, iota)
    return utility


@njit
def ufun(a_, a_grids, u_grids):
    u_ = interp(a_grids, u_grids, a_)
    return u_


@njit
def RetiredUtility(c, a, a_grids, u_grids, ChiC, iota, r, beta, transfer=0.0):
    a_prime = a*(1+r) - c + transfer
    next_utility = ufun(a_prime, a_grids, u_grids)
    total_utility = ConUtility(c, ChiC, iota) + beta*next_utility
    return total_utility


@njit
def RetiredOptimalCon(a, a_grids, u_grids, ChiC, iota, r, beta, B, transfer=0.0):
    add_args = (a, a_grids, u_grids, ChiC, iota, r, beta)
    consumption, utility, _ = brent_max(RetiredUtility, B, a, add_args)
    return consumption, utility


# Solve retired problem
def Solver(a, *args, TotPeriod=5, verbose=False):
    ChiC, iota, r, beta, B = args
    utility = LastPeriodUtility(a, ChiC, iota, r)
    store_utility = np.full((TotPeriod, len(a)), np.nan)
    store_consum = np.full((TotPeriod, len(a)), np.nan)
    for i in range(TotPeriod):
        for j in range(len(a)):
            asset = a[j]
            c, u = RetiredOptimalCon(asset, a, utility, ChiC, iota, r, beta, B)
            store_utility[i, j] = u
            store_consum[i, j] = c
        utility = store_utility[i, :]
    return store_utility, store_consum
