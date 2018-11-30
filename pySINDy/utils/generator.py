"""
This module provides functions that are used to generate
simple dynamical systems
You can simulate your own systems here!

        created:    11/07/18 Yuying Liu (yliu814@uw.edu)
        modified:   11/13/18 Yuying Liu (yliu814@uw.edu)

"""

import numpy as np
import scipy as sp
import math
from scipy import integrate


def linear_dynamics_generator(mtx, x_init, dt=0.01, len_t=10, noise=0.):
    """
    :param mtx: a 2D numpy array (matrix) under which the dynamics evolve
    :param x_init: a 1D numpy array specifies the initial states
    :param dt: float, time step
    :param len_t: time length of simulation
    :param noise: the noise level being added
    :return: a numpy array with shape n x len(tSpan)
    """
    n = np.max(x_init.shape)
    x_init = x_init.reshape(n,)
    t = np.arange(0, len_t+dt, dt)
    sol = sp.integrate.solve_ivp(lambda _, x: np.dot(mtx, x), [0, len_t], x_init, t_eval=t)

    y = sol.y + noise*np.random.rand(n, len(t))

    return y


def multi_scale_linear_dynamics_generator(weights, spatial_exp, temporal_exp, x_scales, t_scales, dx, dt, noise=0.0):
    """
    :param weights: weights of each dynamics
    :param spatial_exp: spatial modes exponents
    :param temporal_exp: temporal modes exponents
    :param x_scales: n x 2 numpy array, provides scale of spatial modes
    :param t_scales: n x 2 numpy array, provides scale of temporal modes
    :param dx: spatial discretization
    :param dt: time step
    :param noise: noise level
    :return: a 2D numpy array represents the multi-scale dynamics (or signal)
    """
    n = np.max(weights.shape)
    weights = weights.reshape(n,)
    m1 = np.max(spatial_exp.shape)
    spatial_exp = spatial_exp.reshape(m1,)
    m2 = np.max(temporal_exp.shape)
    temporal_exp = temporal_exp.reshape(m2, )

    assert m1 == n and m2 == n, "weights and exponents provided should be of the same number!"
    assert len(x_scales.shape) == 2 and x_scales.shape[1] == 2, "x_scale must be a Nx2 numpy array!"
    assert len(t_scales.shape) == 2 and t_scales.shape[1] == 2, "x_scale must be a Nx2 numpy array!"
    assert x_scales.shape[0] == n and t_scales.shape[0] == n, \
        "number of x_scales and t_scales should be the same as weights! "

    # find the boundary
    x_min = np.min(x_scales, axis=0)[0]
    x_max = np.max(x_scales, axis=0)[1]
    t_min = np.min(t_scales, axis=0)[0]
    t_max = np.max(t_scales, axis=0)[1]
    x = np.arange(x_min, x_max+dx, dx)
    t = np.arange(t_min, t_max+dt, dt)

    # adjust the scales
    differences = x_scales.reshape(1, -1) - x.reshape(-1, 1)
    x_indices = np.abs(differences).argmin(axis=0)
    x_scales = np.array([x[i] for i in x_indices]).reshape(-1, 2)
    x_indices = x_indices.reshape(-1, 2)

    differences = t_scales.reshape(1, -1) - t.reshape(-1, 1)
    t_indices = np.abs(differences).argmin(axis=0)
    t_scales = np.array([t[i] for i in t_indices]).reshape(-1, 2)
    t_indices = t_indices.reshape(-1, 2)

    # construct the dynamics / signal
    data = np.zeros([len(t), len(x)])
    for i in np.arange(n):
        xk = np.arange(x_scales[i, 0], x_scales[i, 1], dx)
        tk = np.arange(t_scales[i, 0], t_scales[i, 1], dt)
        xm, tm = np.meshgrid(xk, tk)
        spatial_modes = np.exp(spatial_exp[i].real*xm)*np.sin(spatial_exp[i].imag*xm)
        temporal_modes = np.exp(temporal_exp[i].real*tm)*np.sin(temporal_exp[i].imag*tm)
        data[t_indices[i, 0]:t_indices[i, 1], x_indices[i, 0]:x_indices[i, 1]] += \
            weights[i] * spatial_modes * temporal_modes

    return data.T


def van_der_pol_rhs(mu, x):
    """
    :param mu: float, system parameter
    :param x: 1 x 2 vector, state variables
    :return: numpy array with shape 1 x 2
    """
    return np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]])


def van_der_pol_generator(mu, x_init, dt=0.01, len_t=100):
    """
    :param mu: float, system parameter
    :param x_init: 1 x 2 vector, initial states
    :param dt: time step
    :param len_t: time length of simulation
    :return: 2 x (len_t+1) numpy array of the trajectory
    """

    t = np.arange(0, len_t+dt, dt)
    sol = sp.integrate.solve_ivp(lambda _, x: van_der_pol_rhs(mu, x),
                                 [0, len_t], x_init, t_eval=t)
    return sol.y


def lorenz_rhs(sigma, beta, rho, x):
    """
    :param sigma: parameter of the system
    :param beta: parameter of the system
    :param rho: parameter of the system
    :param x: 1 x 3 vector, state variables
    :return: numpy array with shape 1 x 3
    """
    return [sigma*(x[1] - x[0]), x[0]*(rho - x[2]) - x[1], x[0]*x[1] - beta*x[2]]


def lorenz_generator(sigma, beta, rho, x_init, dt=0.01, len_t=100):
    """
    :param sigma: parameter of the system
    :param beta: parameter of the system
    :param rho: parameter of the system
    :param x_init: 1 x 3 vector, initial states
    :param dt: time length of simulation
    :param len_t: 3 x (len_t+1) numpy array of the trajectory
    :return:
    """
    t = np.arange(0, len_t + dt, dt)
    sol = sp.integrate.solve_ivp(lambda _, x: lorenz_rhs(sigma, beta, rho, x),
                                 [0, len_t], x_init, t_eval=t)
    return sol.y
