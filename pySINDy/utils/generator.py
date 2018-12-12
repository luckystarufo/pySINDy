"""
This module provides functions that are used to generate
simple dynamical systems
You can simulate your own systems here!

        created:    11/07/18 Yuying Liu (yliu814@uw.edu)
        modified:   11/13/18 Yuying Liu (yliu814@uw.edu)
                    12/11/18 Yi Chu (yic317@uw.edu)

"""

import numpy as np
import scipy as sp
from scipy import integrate


def linear_dynamics_generator(mtx, x_init, _dt=0.01, len_t=10, noise=0.):
    """
    :param mtx: a 2D numpy array (matrix) under which the dynamics evolve
    :param x_init: a 1D numpy array specifies the initial states
    :param _dt: float, time step
    :param len_t: time length of simulation
    :param noise: the noise level being added
    :return: a numpy array with shape n x len(tSpan)
    """
    shape = np.max(x_init.shape)
    x_init = x_init.reshape(shape,)
    _t = np.arange(0, len_t+_dt, _dt)
    sol = sp.integrate.solve_ivp(lambda _, _x: np.dot(mtx, _x), [0, len_t], x_init, t_eval=_t)

    _y = sol.y + noise*np.random.rand(shape, len(_t))

    return _y


def multi_scale_linear_dynamics_generator(weights, spatial_exp,
                                          temporal_exp, x_scales, t_scales, _dx, _dt):
    """
    :param weights: weights of each dynamics
    :param spatial_exp: spatial modes exponents
    :param temporal_exp: temporal modes exponents
    :param x_scales: n x 2 numpy array, provides scale of spatial modes
    :param t_scales: n x 2 numpy array, provides scale of temporal modes
    :param _dx: spatial discretization
    :param _dt: time step
    :return: a 2D numpy array represents the multi-scale dynamics (or signal)
    """
    shape = np.max(weights.shape)
    weights = weights.reshape(shape,)
    dim1 = np.max(spatial_exp.shape)
    spatial_exp = spatial_exp.reshape(dim1,)
    dim2 = np.max(temporal_exp.shape)
    temporal_exp = temporal_exp.reshape(dim2, )

    assert dim1 == shape and dim2 == shape,\
        "weights and exponents provided should be of the same number!"
    assert len(x_scales.shape) == 2 and x_scales.shape[1] == 2,\
        "x_scale must be a Nx2 numpy array!"
    assert len(t_scales.shape) == 2 and t_scales.shape[1] == 2,\
        "x_scale must be a Nx2 numpy array!"
    assert x_scales.shape[0] == shape and t_scales.shape[0] == shape, \
        "number of x_scales and t_scales should be the same as weights! "

    # find the boundary
    x_min = np.min(x_scales, axis=0)[0]
    x_max = np.max(x_scales, axis=0)[1]
    t_min = np.min(t_scales, axis=0)[0]
    t_max = np.max(t_scales, axis=0)[1]
    _x = np.arange(x_min, x_max+_dx, _dx)
    _t = np.arange(t_min, t_max+_dt, _dt)

    # adjust the scales
    differences = x_scales.reshape(1, -1) - _x.reshape(-1, 1)
    x_indices = np.abs(differences).argmin(axis=0)
    x_scales = np.array([_x[i] for i in x_indices]).reshape(-1, 2)
    x_indices = x_indices.reshape(-1, 2)

    differences = t_scales.reshape(1, -1) - _t.reshape(-1, 1)
    t_indices = np.abs(differences).argmin(axis=0)
    t_scales = np.array([_t[i] for i in t_indices]).reshape(-1, 2)
    t_indices = t_indices.reshape(-1, 2)

    # construct the dynamics / signal
    data = np.zeros([len(_t), len(_x)])
    for i in np.arange(shape):
        _xk = np.arange(x_scales[i, 0], x_scales[i, 1], _dx)
        _tk = np.arange(t_scales[i, 0], t_scales[i, 1], _dt)
        _xm, _tm = np.meshgrid(_xk, _tk)
        spatial_modes = np.exp(spatial_exp[i].real*_xm)*np.sin(spatial_exp[i].imag*_xm)
        temporal_modes = np.exp(temporal_exp[i].real*_tm)*np.sin(temporal_exp[i].imag*_tm)
        data[t_indices[i, 0]:t_indices[i, 1], x_indices[i, 0]:x_indices[i, 1]] += \
            weights[i] * spatial_modes * temporal_modes

    return data.T


def van_der_pol_rhs(_mu, _x):
    """
    :param _mu: float, system parameter
    :param _x: 1 x 2 vector, state variables
    :return: numpy array with shape 1 x 2
    """
    return np.array([_x[1], _mu*(1 - _x[0]**2)*_x[1] - _x[0]])


def van_der_pol_generator(_mu, x_init, _dt=0.01, len_t=100):
    """
    :param _mu: float, system parameter
    :param x_init: 1 x 2 vector, initial states
    :param _dt: time step
    :param len_t: time length of simulation
    :return: 2 x (len_t+1) numpy array of the trajectory
    """

    _t = np.arange(0, len_t+_dt, _dt)
    sol = sp.integrate.solve_ivp(lambda _, _x: van_der_pol_rhs(_mu, _x),
                                 [0, len_t], x_init, t_eval=_t)
    return sol.y


def lorenz_rhs(sigma, beta, rho, _x):
    """
    :param sigma: parameter of the system
    :param beta: parameter of the system
    :param rho: parameter of the system
    :param _x: 1 x 3 vector, state variables
    :return: numpy array with shape 1 x 3
    """
    return [sigma*(_x[1] - _x[0]), _x[0]*(rho - _x[2]) - _x[1], _x[0]*_x[1] - beta*_x[2]]


def lorenz_generator(sigma, beta, rho, x_init, _dt=0.01, len_t=100):
    """
    :param sigma: parameter of the system
    :param beta: parameter of the system
    :param rho: parameter of the system
    :param x_init: 1 x 3 vector, initial states
    :param _dt: time length of simulation
    :param len_t: 3 x (len_t+1) numpy array of the trajectory
    :return:
    """
    _t = np.arange(0, len_t + _dt, _dt)
    sol = sp.integrate.solve_ivp(lambda _, _x: lorenz_rhs(sigma, beta, rho, _x),
                                 [0, len_t], x_init, t_eval=_t)
    return sol.y
