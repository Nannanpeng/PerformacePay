# This script solve the last-working-period problem 
from pycode.utilities import ConUtility, TotUtility
from pycode.retired import InterpUfun
import numpy as np
from numba import njit
from quantecon.optimize.scalar_maximization import brent_max
from quantecon.quad import qnwnorm
import scipy.optimize as opt


@njit
def LastNoWorkU(c, a, b, a_grids, u_grids, Uparams, Bparams, transfer=0.0):
    ChiC, iota, beta, _, _ = Uparams
    r, _, _ = Bparams
    a_prime = a*(1 + r) + transfer - c
    next_utility = InterpUfun(a_prime, a_grids, u_grids)
    total_utility = ConUtility(c, ChiC, iota) + b + beta*next_utility
    return total_utility


@njit
def LastNoWorkC(a, b, a_grids, u_grids, Uparams, Bparams, transfer=0.0):
    ChiC, iota, beta, _, _ = Uparams
    r, B, _ = Bparams
    add_args = (a, b, a_grids, u_grids, (ChiC, iota, r, beta, b), transfer)
    consumption, utility, _ = brent_max(LastNoWorkU, B, a, add_args)
    return consumption, utility


def LastWorkU(choice, state, pre_shock_wage, a_grids, u_girds,
              Uparams, Bparams, Sparams, tax=0.0, transfer=0.0):
    c,  l = choice
    a, h, z = state
    ChiC, iota, beta, ChiL, psi = Uparams
    pho, sigma_w = Sparams
    r, _, k = Bparams
    weights, nodes = qnwnorm(5, sig2=sigma_w**2)
    z_next = z*pho + (1 - pho)*nodes
    wages = pre_shock_wage + z_next
    a_prime = a*(1+r) + wages*l*h*k*(1 - tax) - c + transfer
    next_utility = InterpUfun(a_prime, a_grids, u_girds)
    expected_utility = np.dot(weights, next_utility)
    tot_utility = TotUtility(c, l, [ChiC, iota, ChiL, psi]) + beta*expected_utility
    return -tot_utility


def LastWorkChoice(state, pre_shock_wage, a_girds, u_girds, Uparams,
                   Bparams, Sparams, tax=0.0, transfer=0.0):
    add_args = (state, pre_shock_wage, a_girds, u_girds, Uparams, 
                Bparams, Sparams, tax, transfer)
    bounds = np.array([[0.001, state[0]], [0, 1]])
    output = opt.minimize(LastWorkU, np.array([0, 0]),
                          bounds=bounds, args=add_args, method='L-BFGS-B')
    if output.success:
        return -output.fun, output.x
    else:
        msg = "Optimization for the consumption and labor failed in the last wp!"
        raise ValueError(msg)

def LastWorkingSolver(states, b, pre_shock_wage, a_grids, u_grids, Uparams, 
                      Bparams, Sparams, industry_idx, tax=0, transfer=0):
    store_u = np.zeros(len(states))
    for i, state in enumerate(states):
        if industry_idx == 0:
            a = state[0]
            _, u = LastNoWorkC(a, b, a_grids, u_grids, Uparams,
                               Bparams, transfer)
            store_u[i] = u
        else:
            u, _ = LastWorkChoice(state, pre_shock_wage, a_grids, u_grids,
                                  Uparams, Bparams, Sparams, tax, transfer)
            store_u[i] = u
    return store_u
