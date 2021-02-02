# This script includes all the preference functions,
# like utility, human capital accumulation and so on.

def Utility(c, l, params):
    ChiC = params["ChiCon"]
    ChiL = params["ChiLabor"]
    phi = params["phi"]
    tao = params["tao"]
    ConUtility = ChiC*c**(1-tao)/(1-tao)
    LaborUtility = ChiL*l**(1+phi)/(1+phi)
    return ConUtility - LaborUtility


def HumanCapitalAccu(h, l, params):
    GammaL = params["GammaL"]
    GammaH = params["GammaH"]
    A = params["A"]
    Xi = params["Xi"]
    left = A*l**GammaL*h**GammaH
    right = (1 - Xi)*h
    return left - right
