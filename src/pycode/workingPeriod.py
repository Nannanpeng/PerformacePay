import numpy as np
from pycode.utilities import TotUtility, HumanCapitalAccu
from quantecon.quad import qnwnorm

def ImmediateUtility(choice, state, Uparams):
    c, l = choice
    utility = TotUtility(c, l, Uparams)
    return utility


def _shock_transition(pre_shock_wage, state, Wparams, n_draws=5):
    pho, sigma_w = Wparams
    _, _, z = state
    nodes, weights = qnwnorm(n_draws, sigma_w**2)
    z_primes = (1-pho)*z + pho*nodes
    wages = pre_shock_wage + z_primes
    return wages, weights, z_primes
    

def _first_two_state_transition(choice, state, wage, Rparams, HCparams):
    r, tax, transfer = Rparams
    a, h, tax = state
    c, l = choice
    a_prime = wage*l*h*(1-tax) + a*(1+r) + transfer - c
    h_prime = HumanCapitalAccu(h, l, HCparams)
    new_first_two_state = [a_prime, h_prime]
    return new_first_two_state 


def StateTransition(choice, state, pre_shock_wage, 
                    Wparams, Rparams, HCparams):
    wages, weights, z_primes = _shock_transition(pre_shock_wage, state, Wparams)
    new_states = np.zeros_like((len(weights), len(state)))
    for i, wage in enumerate(wages):
        first_two_state = _first_two_state_transition(choice, state,
                                                      wage, Rparams, HCparams)
        new_states[i] = first_two_state.append(z_primes[i])
    return new_states, weights

# check for the constant
def _exp_unemployed_utility(states, weights, b, beta, lambda_u):
    exp_unemployed_u = exp_employed_u = 0
    for i, state in enumerate(states):
        exp_unemployed_u += weights[i]*beta*(1-lambda_u)*_unemployed_val_fun(state)
        exp_employed_u += weights[i]*lambda_u*_employed_val_fun(state)
    exp_future_u = exp_unemployed_u + exp_employed_u + b
    return exp_future_u

# check for the constant
def _exp_employed_utility(states, weights, industry_idx,
                          beta, lambda_e, lambda_l):
    exp_unemployed_u = exp_employed_u = exp_stay_u = 0
    for i, state in enumerate(states):
        exp_unemployed_u += weights[i]*beta*lambda_l * _unemployed_val_fun(state)
        exp_employed_u += weights[i]*beta*lambda_e * _employed_val_fun(state)
        exp_stay_u += weights[i]*beta*(1-lambda_e-lambda_l) * \
            _stay_val_fun(state, industry_idx)
    exp_future_u = exp_unemployed_u + exp_employed_u + exp_stay_u
    return exp_future_u
          
    
def ExpectedFutureUtility(states, weights, b, beta, industry_idx, Eparams):
    lambda_u, lambda_e, lambda_l = Eparams
    if industry_idx == 0:
        exp_future_u = _exp_unemployed_utility(states, weights, b, beta, lambda_u)
    else:
        exp_future_u = _exp_employed_utility(states, weights, industry_idx,
                                             beta, lambda_e, lambda_l)
    return exp_future_u


def StateChoiceUtility(choice, state, pre_shock_wage,
                       Uparams, Wparams, Rparams, HCparams,
                       b, beta, industry_idx, Eparams):
    immediate_u = ImmediateUtility(choice, state, Uparams)
    new_states, weights = StateTransition(choice, state, pre_shock_wage, 
                                          Wparams, Rparams, HCparams)
    exp_future_u = ExpectedFutureUtility(new_states, weights, b, beta,
                                         industry_idx, Eparams)
    tot_u = immediate_u + exp_future_u
    return tot_u
