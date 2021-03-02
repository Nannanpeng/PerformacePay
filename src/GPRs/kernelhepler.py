# This file helps find best kernel parameters.
import numpy as np
import scipy.optimize as opt
from GPRs.kernels import RBFkernel


def logPosterior(theta, args):
    data, t = args
    k = RBFkernel(data, data, theta, wantderiv=False)
    L = np.linalg.cholesky(k)
    beta = np.linalg.solve(L.T, np.linalg.solve(L, t))
    logp = -0.5*np.dot(t.T, beta) - np.sum(np.log(
        np.diag(L))) - np.shape(data)[0]/2. * np.log(2*np.pi)
    return -logp


def gradLogPosterior(theta, args):
    data, t = args
    theta = np.squeeze(theta)
    K = RBFkernel(data, data, theta, wantderiv=True)
    L = np.linalg.cholesky(np.squeeze(K[:, :, 0]))
    invk = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(
        np.shape(data)[0])))
    dlogpdtheta = np.zeros(len(theta))
    for d in range(1, len(theta) + 1):
        dlogpdtheta[d-1] = 0.5*np.dot(t.T, np.dot(invk, np.dot(
            np.squeeze(K[:, :, d]), np.dot(invk, t)))) - 0.5*np.trace(
                np.dot(invk, np.squeeze(K[:, :, d])))
    return -dlogpdtheta


def kernelParameterFinding(training_set, method=2):
    data, t = training_set
    if method == 1:
        best_theta = opt.fmin_cg(logPosterior, np.array([0, 0]),
                                fprime=gradLogPosterior, args=[(data, t)], disp=1)
    elif method == 2:
        output = opt.minimize(logPosterior, np.array([0, 0]), args=([data, t]),
                             jac=gradLogPosterior, method='L-BFGS-B')
        if not output.success:
            msg = "Iteration failed: check the optimization for hyperparameters."
            raise ValueError(msg)
        best_theta = output.x
    K = RBFkernel(data, data, best_theta)
    L = np.linalg.cholesky(K)
    coeff = np.linalg.solve(L.T, np.linalg.solve(L, t))
    return coeff, best_theta
