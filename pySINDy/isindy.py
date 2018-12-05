"""
Derived module from sindybase.py for implicit SINDy
"""
import numpy as np
import scipy as sp
from .sindybase import SINDyBase
from findiff import FinDiff


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1

        self.L = Lk
        self.S = Sk
        return Lk, Sk


class ISINDy(SINDyBase):
    """
    Implicit Sparse Identification of Nonlinear Dynamics
    (Inferring biological networks by sparse identification of nonlinear dynamics)
    reference: https://arxiv.org/pdf/1605.08368.pdf
    """
    def fit(self, data, dt, poly_degree=2, deriv_acc=2, Lambda =5e-3, max_iter = 1000, tol = 2e-3):
        """
        :param data: dynamics data to be processed
        :param dt: float, represents grid spacing
        :param poly_degree: degree of polynomials to be included in theta matrix
        :param deriv_acc: (positive) integer, derivative accuracy
        :return: an implicit SINDy model
        """
        if len(data.shape) == 1:
            data = data[np.newaxis, ]

        n, len_t = data.shape

        if len(data.shape) > 2:
            data = data.reshape((-1, len_t))
            print("The array is converted to 2D automatically: in Implicit-SINDy, "
                  "each dimension except for the time (default the last dimension) "
                  "are treated equally.")


        # compute time derivative
        d_dt = FinDiff(data.ndim - 1, dt, 1, acc=deriv_acc)
        x_dot = d_dt(data).T

        data, x_dot = self.smoothing(data, x_dot)

        # polynomial expansion of orginal data
        var_names = ['u%d' % i for i in np.arange(n)]
        self.extended_data = []

        self.extended_data, extended_desp = np.array(self.polynomial_expansion(data.T, degree=poly_degree,
                                                                 var_names=var_names))

        # set the descriptions
        self._desp = self.expand_descriptions(extended_desp, 'uk_{t}')

        # compute sparse coefficients

        self._coef = np.zeros((len(self._desp), n))
        theta_k = []
        for k in np.arange(n):
            # formulate theta1, theta2, theta3 ...
            theta_k = self.build_theta_matrix(self.extended_data, x_dot[:, k])
            rpca = R_pca(theta_k)
            theta_k, _ = rpca.fit()
            # compute null spaces
            null_space_k = self.nullify(theta_k)
            # ADM
            self._coef[:, k] = self.adm_initvary(self, null_space_k, Lambda, max_iter, tol)
        return self

    @staticmethod
    def expand_descriptions(desp, var_name=None):
        if var_name:
            if not isinstance(var_name, str):
                raise ValueError("var_name should be of str type!")
            expanded = []
            for d in desp:
                if d == '1':
                    expanded.append(var_name)
                else:
                    expanded.append(d + var_name)
            return desp + expanded
        else:
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

        m, n = data.shape

        if not isinstance(d_vec, np.ndarray) or d_vec.ndim != 1:
            raise ValueError("input d_vec should be a 1D numpy vector!")

        assert len(d_vec) == m, "the length of d_vec should be the same as columns in data!"

        extended = np.zeros((m, n))
        for i in np.arange(n):
            extended[:, i] = data[:, i]*d_vec

        return np.hstack([data, extended])

    @staticmethod
    def nullify(A):
        """
        :param theta: a 2D numpy array
        :return: a 2D numpy array representing the null space of theta
        """
        return sp.linalg.null_space(A)

    @staticmethod
    def adm(null_space, qinit, Lambda, max_iter, tol):
        """
        :param null_space:
        :param qinit:
        :param Lambda
        :param maxiter
        :param tol
        :return: find a sparse representation
        """
        q = qinit
        for k in range(max_iter):
            q_old = q
            soft_thresholding = lambda X, d: np.multiply(np.sign(X), np.maximum(np.absolute(X) - d, 0))
            x = soft_thresholding(np.matmul(null_space, q), Lambda)
            q = np.matmul(null_space.T, x)/(np.linalg.norm(np.matmul(null_space.T, x), ord=2))
            res_q = np.linalg.norm(q_old - q, ord=2)
            if res_q <= tol:
                return q
        print('warning, cant find optimum q within number of maxiter')
        return q

    @staticmethod
    def adm_initvary(self, null_space, Lambda, max_iter, tol):
        """
        :param null_space:
        :param Lambda:
        :param maxiter
        :param tol
        :return: find a sparse representation
        """
        nTn = np.empty(null_space.shape)
        q = np.empty((null_space.shape[1],null_space.shape[0]))
        out = np.empty((null_space.shape[0],null_space.shape[0]))
        nzeros = np.empty((1,null_space.shape[0]))
        for ii in range (null_space.shape[1]):
            nTn[:,ii] = null_space[:,ii]/np.mean(null_space[:,ii])

        for jj in range (nTn.shape[0]):
            # intial conditions
            qinit = nTn[jj,:].T
            # algrorithm for finding coefficients resutling in  sparsest vector in null space
            q[:,jj] = self.adm(null_space, qinit, Lambda, max_iter, tol)
            # compose sparsets vectors
            out[:,jj] = np.matmul(null_space, q[:,jj])
            # chech how many zeros each of the found sparse vectors have
            nzeros[0,jj]= np.count_nonzero(abs(out[:,jj]) < Lambda)

        # find the vector with the largest number of zeros
        indsparse = np.nonzero((nzeros==np.max(nzeros)).flatten())[0]

        # save the indices of non zero coefficients
        indTheta = np.nonzero((np.absolute(out[:,indsparse[0]]) >= Lambda))[0]
        # get sparsest vector
        Xi = out[:, indsparse[0]]
        smallinds =(abs(out[:,indsparse[0]]) < Lambda)
        # set thresholded coefficients to zero
        Xi[smallinds] = 0

        # check that the solution found by ADM is unique.
        if len(indsparse) > 1:
            Xidiff = (out[indTheta, indsparse[0]] - out[indTheta,indsparse[1]])
            if np.all(Xidiff > tol) == 1:
                print('ADM has discovered more than 1 sparsest vectors')

        # calculate how many terms are in the sparsest vector
#        numterms = len(indTheta)
        return Xi

    @staticmethod
    def smoothing(self, vec):
        """
        :param vec: input data vector for smoothing
        :return: smoothed data vector same shape as vec
        """
        mean = np.mean(vec)
        std = np.std(vec)
        upper = mean + std
        lower = mean - std

        m = len(vec)
        if (vec[0] > upper or vec[0] < lower):
            vec[0] = self.smoothing_helper(self, vec, 0, upper, lower)
        for j in range(1, m):
            left = vec[j - 1]
            if vec[j] > upper or vec[j] < lower:
                vec[j] = left

        return vec

    @staticmethod
    def smoothing_helper(self, vec, start, upper, lower):
        if vec[start] > upper or vec[start] < lower:
            return self.smoothing_helper(self, vec, start + 1, upper, lower)
        else:
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

