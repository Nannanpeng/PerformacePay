# This script solve the last-working-period problem 
from pycode.utilities import ConUtility, TotUtility
from pycode.retired import InterpUfun
import numpy as np
from numba import njit
from quantecon.optimize.scalar_maximization import brent_max
from quantecon.quad import qnwnorm
import scipy.optimize as opt


@njit
def LastNoWorkU(c, a, a_grids, u_grids, args, transfer=0.0):
    ChiC, iota, r, beta, b = args
    a_prime = a*(1 + r) + transfer - c
    next_utility = InterpUfun(a_prime, a_grids, u_grids)
    total_utility = ConUtility(c, ChiC, iota) + b + beta*next_utility
    return total_utility


@njit
def LastNoWorkC(a, a_grids, u_grids, args, transfer=0.0):
    ChiC, iota, r, beta, B, b = args
    add_args = (a, a_grids, u_grids, (ChiC, iota, r, beta, b), transfer)
    consumption, utility, _ = brent_max(LastNoWorkU, B, a, add_args)
    return consumption, utility


def LastWorkU(choice, state, a_grids, u_girds, params,
              wage_params, tax=0.0, transfer=0.0):
    c,  l = choice
    a, h, z = state
    ChiC, iota, r, beta, ChiL, psi = params
    pre_shock, pho, sigma_w = wage_params
    weights, nodes = qnwnorm(5, sig2=sigma_w**2)
    z_next = z*pho + (1 - pho)*nodes
    wages = pre_shock + z_next
    a_prime = a*(1+r) + wages*l*h*(1 - tax) - c + transfer
    next_utility = InterpUfun(a_prime, a_grids, u_girds)
    expected_utility = np.dot(weights, next_utility)
    tot_utility = TotUtility(c, l, [ChiC, iota, ChiL, psi]) + beta*expected_utility
    return -tot_utility


def LastWorkChoice(state, a_girds, u_girds, params,
                   wage_params, tax=0.0, transfer=0.0):
    add_args = (state, a_girds, u_girds, params, wage_params,
                tax, transfer)
    bounds = np.array([[0.001, state[0]], [0, 1]])
    output = opt.minimize(LastWorkU, np.array([0, 0]),
                          bounds=bounds, args=add_args, method='L-BFGS-B')
    if output.success:
        return -output.fun, output.x
    else:
        msg = "Optimization for the consumption and labor failed in the last wp!"
        raise ValueError(msg)
