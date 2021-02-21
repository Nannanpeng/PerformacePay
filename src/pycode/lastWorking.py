# This script solve the last-working-period problem 
import numpy as np
from numba import njit
from utilities import ConUtility
from interpolation import interp
from quantecon.optimize.scalar_maximization import brent_max

@njit
def UtilityFun(a, a_grids, u_grids):
    u = interp(a_grids, u_grids, a)
    return u


@njit
def LastNoWorkU(c, a, a_grids, u_grids, args, transfer=0.0):
    ChiC, iota, r, beta, b = args
    a_prime = a*(1 + r) + transfer - c
    next_utility = UtilityFun(a_prime, a_grids, u_grids)
    total_utility = ConUtility(c, ChiC, iota) + b + beta*next_utility
    return total_utility


@njit
def LastNoWorkC(a, a_grids, u_grids, args, transfer=0.0):
    ChiC, iota, r, beta, B, b = args
    add_args = (a, a_grids, u_grids, (ChiC, iota, r, beta, b), transfer)
    consumption, utility, _ = brent_max(LastNoWorkU, B, a, add_args)
    return consumption, utility

