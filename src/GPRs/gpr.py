# This file genearte GPRs to approximate the functions.
import numpy as np
import scipy.optimize as opt


def kernel(data1, data2, theta, wantderiv=False):
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


def logPosterior(theta, args):
    data, t = args
    k = kernel(data, data, theta, wantderiv=False)
    L = np.linalg.cholesky(k)
    beta = np.linalg.solve(L.T, np.linalg.solve(L, t))
    logp = -0.5*np.dot(t.T, beta) - np.sum(np.log(
        np.diag(L))) - np.shape(data)[0]/2. * np.log(2*np.pi)
    return -logp


def gradLogPosterior(theta, args):
    data, t = args
    theta = np.squeeze(theta)
    K = kernel(data, data, theta, wantderiv=True)
    L = np.linalg.cholesky(np.squeeze(K[:, :, 0]))
    invk = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(
        np.shape(data)[0])))
    dlogpdtheta = np.zeros(len(theta))
    for d in range(1, len(theta) + 1):
        dlogpdtheta[d-1] = 0.5*np.dot(t.T, np.dot(invk, np.dot(
            np.squeeze(K[:, :, d]), np.dot(invk, t)))) - 0.5*np.trace(
                np.dot(invk, np.squeeze(K[:, :, d])))
    return -dlogpdtheta


def getDerivative(data, state, params):
    params = np.squeeze(params)
    coeff = -np.exp(params[1])
    if data.shape[1] == 1:
        return coeff*(state - data)
    else:
        return coeff*(state[0] - data[:, 0].reshape((-1, 1)))


def _Help_Prediction(training_set, method=2):
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
    K = kernel(data, data, best_theta)
    L = np.linalg.cholesky(K)
    coeff = np.linalg.solve(L.T, np.linalg.solve(L, t))
    return coeff, best_theta


def Prediction(state, traning_set, coeff, theta,  wantderiv=False):
    data, _ = traning_set
    kstar = kernel(data, state, theta)
    mean = np.dot(kstar.T, coeff)
    if wantderiv:
        prediction = np.zeros((2, 1), dtype=np.float64)
        prediction[0] = mean
        kstarderiv = getDerivative(data, state, theta) * kstar
        prediction[1] = np.dot(kstarderiv.T, coeff)
    else:
        prediction = mean
    return prediction
