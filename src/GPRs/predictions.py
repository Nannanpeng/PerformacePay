# This file includes prediction function to predict unknow labels. 
import numpy as np
from GPRs.kernels import RBFkernel


def getDerivative(data, state, params):
    params = np.squeeze(params)
    coeff = -np.exp(params[1])
    if data.shape[1] == 1:
        return coeff*(state - data)
    else:
        return coeff*(state[0] - data[:, 0].reshape((-1, 1)))


def Prediction(state, traning_set, coeff, theta,  wantderiv=False):
    data, _ = traning_set
    kstar = RBFkernel(data, state, theta)
    mean = np.dot(kstar.T, coeff)
    if wantderiv:
        prediction = np.zeros((2, 1), dtype=np.float64)
        prediction[0] = mean
        kstarderiv = getDerivative(data, state, theta) * kstar
        prediction[1] = np.dot(kstarderiv.T, coeff)
    else:
        prediction = mean
    return prediction
