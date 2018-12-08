"""
Derived module from sindybase.py for implicit SINDy
"""

import numpy as np
from scipy.linalg import null_space
from findiff import FinDiff
from .rpca import RPca
from .sindybase import SINDyBase

class ISINDy(SINDyBase):
    """
    Implicit Sparse Identification of Nonlinear Dynamics
    (Inferring biological networks by sparse identification of nonlinear dynamics)
    reference: https://arxiv.org/pdf/1605.08368.pdf
    """
    def fit(self, data, _dt, poly_degree=2, deriv_acc=2,
            lmbda=5e-3, max_iter=1000, tol=2e-3):
        """
        :param data: dynamics data to be processed
        :param dt: float, represents grid spacing
        :param poly_degree: degree of polynomials to be included in theta matrix
        :param deriv_acc: (positive) integer, derivative accuracy
        :param lmbda: threshold for doing adm
        :param max_iter: max iteration number for adm
        :param tol: tolerance for stopping criteria for adm
        :return: an implicit SINDy model
        """
        if len(data.shape) == 1:
            data = data[np.newaxis, ]
        
        if len(data.shape) == 2:
            _n, len_t = data.shape

        if len(data.shape) > 2:
            len_t = data.shape[-1]
            data = data.reshape((-1, len_t))
            print("The array is converted to 2D automatically: in Implicit-SINDy, "
                  "each dimension except for the time (default the last dimension) "
                  "are treated equally.")
            _n = data.shape[0]


        # compute time derivative
        d_dt = FinDiff(data.ndim - 1, _dt, 1, acc=deriv_acc)
        x_dot = d_dt(data).T

        # pre-process dxt
        for i in range(x_dot.shape[1]):
            x_dot[:, i] = ISINDy.smoothing(x_dot[:, i])

        # polynomial expansion of original data
        var_names = ['u%d' % i for i in np.arange(_n)]

        extended_data, extended_desp = np.array(self.polynomial_expansion(data.T,
                                                                          degree=poly_degree,
                                                                          var_names=var_names))

        # set the descriptions
        self._desp = self.expand_descriptions(extended_desp, 'uk_{t}')

        # compute sparse coefficients

        self._coef = np.zeros((len(self._desp), _n))

        for k in np.arange(_n):
            # formulate theta1, theta2, theta3 ...
            theta_k = self.build_theta_matrix(extended_data, x_dot[:, k])
            # do principal component analysis to return low-rank theta
            rpca = RPca(theta_k)
            theta_k, _ = rpca.fit()
            # compute null spaces
            null_space_k = null_space(theta_k)
            # ADM
            self._coef[:, k] = ISINDy.adm_initvary(null_space_k, lmbda, max_iter, tol)
        return self

    def coefficients(self):
        """
        :return: coefficients of the system
        """
        return self._coef

    def descriptions(self):
        """
        :return: descriptions of corresponding coefficients
        """
        return self._desp

    @staticmethod
    def expand_descriptions(desp, var_name=None):
        """
        :param desp: descriptions for expanding
        :param var_name: define names of variables
        :return: 2D numpy array after expanding descriptions
        """
        if var_name:
            if not isinstance(var_name, str):
                raise ValueError("var_name should be of str type!")
            expanded = []
            for _d in desp:
                if _d == '1':
                    expanded.append(var_name)
                else:
                    expanded.append(_d + var_name)
            return desp + expanded
        return desp

    @staticmethod
    def build_theta_matrix(data, d_vec):
        """
        :param data: a 2D numpy array of features stored in each column
        :param d_vec: a vector of time derivatives of a state
        :return: a 2D numpy array expanded by the multiplication of data and d_vec
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("input data should be a 2D numpy array!")

        _m, _n = data.shape

        if not isinstance(d_vec, np.ndarray) or d_vec.ndim != 1:
            raise ValueError("input d_vec should be a 1D numpy vector!")

        assert len(d_vec) == _m, "the length of d_vec should be the same as columns in data!"

        extended = np.zeros((_m, _n))
        for i in np.arange(_n):
            extended[:, i] = data[:, i]*d_vec

        return np.hstack([data, extended])

    @staticmethod
    def adm(subspace, qinit, lmbda, max_iter, tol):
        """
        :param subspace: a subspace for finding sparsest vector
        :param qinit: initialize coefficient matrix
        :param lmbda: threshold for doing adm
        :param max_iter: max iteration number for adm
        :param tol: tolerance for stopping criteria for adm
        :return: find a sparse representation
        """
        _q = qinit
        for _ in range(max_iter):
            q_old = _q
            soft_thresholding = lambda s, d: \
                np.multiply(np.sign(s), np.maximum(np.absolute(s) - d, 0))
            _x = soft_thresholding(np.matmul(subspace, _q), lmbda)
            _q = np.matmul(subspace.T, _x)/(np.linalg.norm(np.matmul(subspace.T, _x), ord=2))
            res_q = np.linalg.norm(q_old - _q, ord=2)
            if res_q <= tol:
                return _q
        print('warning, cant find optimum q within number of max iteration')
        return _q

    @staticmethod
    def adm_initvary(null_space, lmbda, max_iter, tol):
        """
        :param self:
        :param null_space:
        :param lmbda: threshold for doing adm
        :param max_iter: max iteration number for adm
        :param tol: tolerance for stopping criteria for adm
        :return: find a sparse representation
        """
        ntn = np.empty(null_space.shape)
        _q = np.empty((null_space.shape[1], null_space.shape[0]))
        out = np.empty((null_space.shape[0], null_space.shape[0]))
        nzeros = np.empty((1, null_space.shape[0]))
        for _ii in range(null_space.shape[1]):
            ntn[:, _ii] = null_space[:, _ii]/np.mean(null_space[:, _ii])

        for _jj in range(ntn.shape[0]):
            # initial conditions
            qinit = ntn[_jj, :].T
            # algorithm for finding coefficients resulting in  sparsest vector in null space
            _q[:, _jj] = ISINDy.adm(null_space, qinit, lmbda, max_iter, tol)
            # compose sparest vectors
            out[:, _jj] = np.matmul(null_space, _q[:, _jj])
            # check how many zeros each of the found sparse vectors have
            nzeros[0, _jj] = np.count_nonzero(abs(out[:, _jj]) < lmbda)

        # find the vector with the largest number of zeros
        indsparse = np.nonzero((nzeros == np.max(nzeros)).flatten())[0]

        # save the indices of non zero coefficients
        indtheta = np.nonzero((np.absolute(out[:, indsparse[0]]) >= lmbda))[0]
        # get sparsest vector
        _xi = out[:, indsparse[0]]
        smallinds = (abs(out[:, indsparse[0]]) < lmbda)
        # set threshold coefficients to zero
        _xi[smallinds] = 0

        # check that the solution found by ADM is unique.
        if len(indsparse) > 1:
            xidiff = (out[indtheta, indsparse[0]] - out[indtheta, indsparse[1]])
            if np.all(xidiff > tol) == 1:
                print('ADM has discovered more than 1 sparsest vectors')

        # calculate how many terms are in the sparsest vector
#        numterms = len(indtheta)
        return _xi

    @staticmethod
    def smoothing(vec):
        """
        :param vec: input data vector for smoothing
        :return: smoothed data vector same shape as vec
        """
        mean = np.mean(vec)
        std = np.std(vec)
        upper = mean + std
        lower = mean - std

        _m = len(vec)
        if vec[0] > upper or vec[0] < lower:
            vec[0] = ISINDy.smoothing_helper(vec, 0, upper, lower)
        for j in range(1, _m):
            left = vec[j - 1]
            if vec[j] > upper or vec[j] < lower:
                vec[j] = left

        return vec

    @staticmethod
    def smoothing_helper(vec, start, upper, lower):
        """
        :param vec: input vector for checking
        :param start: starting index
        :param upper: upper bound
        :param lower: lower bound
        :return:
        """
        if vec[start] > upper or vec[start] < lower:
            return ISINDy.smoothing_helper(vec, start + 1, upper, lower)
        return vec[start]

    # @staticmethod
    # def smoothing(data, dxt):
    #     """
    #     :param vec: input data vector for smoothing
    #     :return: smoothed data vector same shape as vec
    #     """
    #     mean = np.mean(dxt)
    #     std = np.std(dxt)
    #     print(std)
    #     upper = mean+std/2
    #     lower = mean-std/2
    #     idx = np.ones((data.shape[1]),dtype = bool)
    #     print(idx.shape)
    #
    #     m,n = data.shape
    #     for i in range(m):
    #         if (dxt[0, i]>upper or dxt[0, i]<lower):
    #             print('first zero')
    #             dxt[0, i] = 0
    #         for j in range(1, n):
    #             if dxt[j,i]>upper or dxt[j,i]<lower:
    #                 idx[j] = False
    #     print(data.shape)
    #     print(dxt.shape)
    #     data = data.T[idx].T
    #     dxt = dxt[idx]
    #
    #
    #     return data, dxt

    # @staticmethod
    # def smoothing_helper(self, vec, start, upper, lower):
    #     if vec[start]>upper or vec[start]<lower:
    #         return self.smoothing_helper(self, vec, start + 1, upper, lower)
    #     else:
    #         return vec[start]
