import numpy as np
from pycode.utilities import TotUtility, HumanCapitalAccu
from quantecon.quad import qnwnorm
import scipy.optimize as opt

def ImmediateUtility(choice, state, Uparams):
    ChiC, iota, _, ChiL, psi = Uparams
    c, l = choice
    utility = TotUtility(c, l, [ChiC, iota, ChiL, psi])
    return utility


def _shock_transition(pre_shock_wage, state, Sparams, n_draws=5):
    pho, sigma_w = Sparams
    _, _, z = state
    nodes, weights = qnwnorm(n_draws, sigma_w**2)
    z_primes = (1-pho)*z + pho*nodes
    wages = pre_shock_wage + z_primes
    return wages, weights, z_primes
    

def _first_two_state_transition(choice, state, wage, Bparams, 
                                Hparams, tax, transfer):
    r, _, k = Bparams
    a, h, tax = state
    c, l = choice
    a_prime = wage*l*h*k*(1-tax) + a*(1+r) + transfer - c
    h_prime = HumanCapitalAccu(h, l, Hparams)
    new_first_two_state = [a_prime, h_prime]
    return new_first_two_state 


def StateTransition(choice, state, pre_shock_wage, 
                    Sparams, Bparams, Hparams, tax, transfer):
    wages, weights, z_primes = _shock_transition(pre_shock_wage, state, Sparams)
    new_states = np.zeros_like((len(weights), len(state)))
    for i, wage in enumerate(wages):
        first_two_state = _first_two_state_transition(choice, state, wage, Bparams,
                                                      Hparams, tax, transfer)
        new_states[i] = first_two_state.append(z_primes[i])
    return new_states, weights

# check for the constant
def _exp_unemployed_utility(states, weights, b, Uparams, lambda_u):
    _, _, beta, _, _ = Uparams
    exp_unemployed_u = exp_employed_u = 0
    for i, state in enumerate(states):
        exp_unemployed_u += weights[i]*beta*(1-lambda_u)*_unemployed_val_fun(state)
        exp_employed_u += weights[i]*lambda_u*_employed_val_fun(state)
    exp_future_u = exp_unemployed_u + exp_employed_u + b
    return exp_future_u

# check for the constant
def _exp_employed_utility(states, weights, industry_idx,
                          Uparams, lambda_e, lambda_l):
    _, _, beta, _, _ = Uparams
    exp_unemployed_u = exp_employed_u = exp_stay_u = 0
    for i, state in enumerate(states):
        exp_unemployed_u += weights[i]*beta*lambda_l * _unemployed_val_fun(state)
        exp_employed_u += weights[i]*beta*lambda_e * _employed_val_fun(state)
        exp_stay_u += weights[i]*beta*(1-lambda_e-lambda_l) * \
            _stay_val_fun(state, industry_idx)
    exp_future_u = exp_unemployed_u + exp_employed_u + exp_stay_u
    return exp_future_u
          
    
def ExpectedFutureUtility(states, weights, b, Uparams, industry_idx, Eparams):
    _, _, beta, _, _ = Uparams
    lambda_u, lambda_e, lambda_l = Eparams
    if industry_idx == 0:
        exp_future_u = _exp_unemployed_utility(states, weights, b, beta, lambda_u)
    else:
        exp_future_u = _exp_employed_utility(states, weights, industry_idx,
                                             beta, lambda_e, lambda_l)
    return exp_future_u


def emStateChoiceUtility(choice, state, pre_shock_wage, params,
                       industry_idx, b, tax, transfer):
    Uparams, Bparams, Hparams, Sparams, Eparams = params
    immediate_u = ImmediateUtility(choice, state, Uparams)
    new_states, weights = StateTransition(choice, state, pre_shock_wage, 
                                          Sparams, Bparams, Hparams, tax, transfer)
    exp_future_u = ExpectedFutureUtility(new_states, weights, b, Uparams,
                                         industry_idx, Eparams)
    tot_u = immediate_u + exp_future_u
    return -tot_u


def uemStateChoiceUtility(c, state, pre_shock_wage, params,
                       industry_idx, b, tax, transfer):
    choice = [c, 0]
    _tot_u = emStateChoiceUtility(choice, state, pre_shock_wage, params,
                                  industry_idx, b, tax, transfer)
    return _tot_u


def WorkingPeriodSolver(state, pre_shock_wage, params, industry_idx,
                        b=0, tax=0, transfer=0):
    args = state, pre_shock_wage, params, industry_idx, b, tax, transfer
    if industry_idx == 0:
        bounds = np.array([[0, state[0]]])
        output = opt.minimize(uemStateChoiceUtility, 0, args=args,
                              bounds=bounds, method="L-BFGS-B")
    else:
        bounds = np.array([[0, state[0]], [0, 1]])
        init = np.array([0, 0])
        output = opt.minimize(emStateChoiceUtility, x0=init, args=args,
                              bounds=bounds, method="L-BFGS-B")
    if output.success:
        return -output.fun, output.x
    else:
        msg = "Optimization for the consumption and labor failed in the wp!"
        raise ValueError(msg)
