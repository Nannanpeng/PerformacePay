# This script includes all the preference functions,
# like utility, human capital accumulation and so on.
import numpy as np
from numba import njit


@njit
def ConUtility(c, ChiC, iota):
    if iota == 1:
        con_utility = ChiC*np.log(c)
        return con_utility
    else:
        con_utility = ChiC*c**(1-iota)/(1-iota)
        return con_utility


@njit
def LabUtility(labor, ChiL, psi):
    labor_utility = ChiL*labor**(1+psi)/(1+psi)
    return labor_utility


@njit
def TotUtility(c, labor, args):
    ChiC, iota, ChiL, psi = args
    con_utility = ConUtility(c, ChiC, iota)
    lab_utility = LabUtility(labor, ChiL, psi)
    return con_utility - lab_utility


@njit
def HumanCapitalAccu(h, labor, args):
    A, GammaL, GammaH, Xi = args
    accu = A*labor**GammaL*h**GammaH
    remaining = (1 - Xi)*h
    return accu + remaining
