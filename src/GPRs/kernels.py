# This file generate GPR kernels
import numpy as np


def RBFkernel(data1, data2, theta, wantderiv=False):
    theta = np.squeeze(theta)
    theta = np.exp(theta)
    if data1.ndim == 1:
        d1 = np.shape(data1)[0]
        n = 1
    else:
        (d1, n) = np.shape(data1)
    d2 = np.shape(data2)[0]
    sumxy = np.zeros((d1, d2))
    for d in range(n):
        D1 = np.transpose([data1[:, d]])*np.ones((d1, d2))
        if data2.ndim == 1:
            D2 = [data2[d]]*np.ones((d1, d2))
        else:
            D2 = [data2[:, d]]*np.ones((d1, d2))
        sumxy += (D1 - D2)**2*theta[1]
    k = theta[0]*np.exp(-0.5*sumxy)
    if wantderiv:
        K = np.zeros((d1, d2, len(theta)+1))
        K[:, :, 0] = k
        K[:, :, 1] = k
        K[:, :, 2] = -0.5*k*sumxy
        return K
    else:
        return k
