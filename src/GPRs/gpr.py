# This file genearte GPRs to approximate the functions.
import numpy as np


def kernel(data1, data2, theta, wantderiv=True, measnoise=1.):
    theta = np.squeeze(theta)
    theta = np.exp(theta)
    if np.ndim(data1) == 1:
        d1 = np.shape(data1)[0]
        n = 1
    else:
        (d1, n) = np.shape(data1)
    d2 = np.shape(data2)[0]
    sumxy = np.zeros((d1, d2))
    for d in range(n):
        D1 = np.transpose([data1[:, d]])*np.ones((d1, d2))
        D2 = [data1[:, d]]*np.ones((d1, d2))
        sumxy += (D1 - D2)**2*theta[d+1]
    k = theta[0]*np.exp(-0.5*sumxy)  
    if wantderiv:
        K = np.zeros((d1, d2, len(theta)+1))
        K[:, :, 0] = k + measnoise*theta[2]*np.eye(d1, d2)
        K[:, :, 1] = k
        K[:, :, 2] = -0.5*k*sumxy
        K[:, :, 3] = theta[2]*np.eye(d1, d2)
        return K
    else:
        return k + measnoise*theta[2]*np.eye(d1, d2)


def logPosterior(theta, args):
    data, t = args
    k = kernel(data, data, theta, wantderiv=False)
    L = np.linalg.cholesky(k)
    beta = np.linalg.solve(L.T, np.linalg.solve(L, t))
    logp = -0.5*np.dot(t.T, beta) - np.sum(np.log(
        np.diag(L))) - np.shape(data)[0]/2 * np.log(2*np.pi)
    return -logp


def gradLogPosterior(theta, args):
    data, t = args
    theta = np.squeeze(theta)
    d = len(theta)
    K = kernel(data, data, theta, wantderiv=True)
    L = np.linalg.cholesky(np.squeeze(K[:, :, 0]))
    invk = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(
        np.shape(data)[0])))
    dlogpdtheta = np.zeros(d)
    for d in range(1, len(theta) + 1):
        dlogpdtheta[d-1] = 0.5*np.dot(t.T, np.dot(invk, np.dot(
            np.squeeze(K[:, :, d]), np.dot(invk, t)))) - 0.5*np.trace(
                np.dot(invk, np.squeeze(K[:, :, d])))
    return -dlogpdtheta
