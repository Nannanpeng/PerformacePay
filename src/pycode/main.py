import numpy as np
from GPRs.kernelhepler import kernelParamsFinding
from pycode.retired import RetiredSolver
from pycode.lastWorking import LastWorkingSolver
from pycode.workingPeriod import WorkingPeriodSolver


class DPsolver:
    def __init__(self, tot_p, retired_p, n_obs=20, 
                 n_indus=10, n_states=3, n_pp=2, n_types=3): # n_types means type of agent
        self.tot_p = tot_p
        self.retired_p = retired_p
        self.n_obs = n_obs
        self.n_indus = n_indus
        self.n_states = n_states
        self.n_pp = n_pp
        self.n_types = n_types
        self.tot_retired_p = tot_p - retired_p
        
        self.store_w_states = np.zeros((n_indus, n_pp, retired_p,
                                       n_obs, n_states)) # store states
        self.store_w_u = np.zeros((n_indus, n_pp, retired_p,  
                                    n_obs, 1))           # store utilities
        self.store_w_coeff = np.zeros((n_indus, n_pp, retired_p,
                                       n_obs, 1))        # store coeff for gpr prediction
        self.store_gpr_thetas = np.zeros((n_indus, n_pp, retired_p, 2)) # store gpr params
        self.store_w_cl = np.zeros((n_indus, n_pp, retired_p, n_obs, 2))  # store consump and labor
        
        self.store_retired_states = None
        self.store_retired_u = None
    
    def RetiredProblem(self, ini_states_a, Uparams, Bparams):
        """
        len(ini_states) = self.n_obs; 
        ChiC, iota, beta, _, _ = Uparams
        r, B, _ = Bparams
        """
        self.store_retired_states = ini_states_a
        u, _ = RetiredSolver(ini_states_a, Uparams, Bparams,
                             self.tot_retired_p)
        self.store_retired_u = u
    
    def LastWorkingProblem(self, states, pre_shock_params, Uparams, Bparams,
                           Sparams, b, tax=0, trasnfer=0):
        d1, d2 = np.shape(states)
        if d1 != self.n_obs or d2 != self.n_states:
            msg = "Incorrect dimension of states in the last working period."
            raise ValueError(msg) 
        
        a_grids = self.store_retired_states
        u_grids = self.store_retired_u[-1] # only last period is used
        for i in range(self.n_indus):
            for  p in range(self.n_pp):
                for m in range(self.n_types): # they may differ in parameters
                    if i != 0:
                        pre_shock_wage = self._pre_shock_wage(i, p, m, pre_shock_params)
                    else:
                        pre_shock_wage = 0
                    u = LastWorkingSolver(states, b, pre_shock_wage, a_grids, u_grids,
                                        Uparams, Bparams, Sparams, i, tax, trasnfer)
                    for s in range(len(states)):
                        self.store_w_states[i, p, 0, s,:] = states[s]
                        self.store_w_u[i, p, 0, s, :] = u[s]
                    training_set = [self.store_w_states[i, p, 0, :, :],
                                    self.store_w_u[i, p, 0, :, :]]
                    coeff, thetas = kernelParamsFinding(training_set)
                    self.store_w_coeff[i, p, 0, :, :] = coeff
                    self.store_gpr_thetas[i, p, 0, :] = thetas
    
    #employment related probability params, m_types, pp status 
    def WorkingPeriodProblem(self, states, pre_shock_params, Uparams, Bparams,
                             Hparams, Sparams, Eparams,  b, tax=0, transfer=0):
        params = (Uparams, Bparams, Hparams, Sparams, Eparams)
        stored_data = (self.store_w_states, self.store_w_u,
                       self.store_w_coeff, self.store_gpr_thetas)
        for i in range(self.n_indus):
            for t in range(self.retired_p):
                for p in range(self.n_pp):
                    for m in range(self.n_types):
                        if i != 0:
                            pre_shock_wage = self._pre_shock_wage(i, p, m, 
                                                                  pre_shock_params)
                        else:
                            pre_shock_wage = 0 
                        for s in range(self.n_obs):
                            u, c = WorkingPeriodSolver(states[s], pre_shock_wage, params,
                                                       i, t, stored_data, b, tax, transfer)
                            self.store_w_states[i, t, p, s, :] = states[s]
                            self.store_w_u[i, t, p, s, :] = u
            
    def _pre_shock_wage(self, i, p, m, pre_shock_params):
        v_len = 1 + 1 + (self.n_indus - 1) + \
                (self.n_indus - 1) + self.n_types
        if len(pre_shock_params) != v_len:
            raise ValueError("Incorret dimensions of pre_shock_params.") 
        pre_shock_status = np.zeros(v_len)
        pre_shock_status[1+i] = 1
        pre_shock_status[1] = p
        if p != 0:
            # position of interation term: pp*industry
            idx = 1 + 1 + (self.n_indus - 1) + i - 1 
            pre_shock_status[idx] = 1
        m_idx = 1 + 1 + (self.n_indus - 1) + (self.n_indus - 1)
        pre_shock_status[m_idx] = 1
        pre_shock_wage = np.dot(pre_shock_status, pre_shock_params)
        return pre_shock_wage
