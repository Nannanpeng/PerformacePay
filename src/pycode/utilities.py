# This script includes all the preference functions,
# like utility, human capital accumulation and so on.

# jitable
def ConUtility(c, ChiC, iota):
    con_utility = ChiC*c**(1-iota)/(1-iota)
    return con_utility


# jitable
def LabUtility(labor, ChiL, psi):
    labor_utility = ChiL*labor**(1+psi)/(1+psi)
    return labor_utility


# jitable
def TotUtility(c, labor, *args):
    ChiC, iota, ChiL, psi = args
    con_utility = ConUtility(c, ChiC, iota)
    lab_utility = LabUtility(labor, ChiL, psi)
    return con_utility - lab_utility


# jitable
def HumanCapitalAccu(h, labor, *args):
    A, GammaL, GammaH, Xi = args
    accu = A*labor**GammaL*h**GammaH
    remaining = (1 - Xi)*h
    return accu + remaining
