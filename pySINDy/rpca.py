import numpy as np


class RPca:
    """
    robust principal component analysis for low-rank array
    """

    def __init__(self, data, _mu=None, lmbda=None):
        """
            :param data: input data
            :param _mu: threshold
            :param lmbda: sparse threshold
            """

        self.data = data
        self._s = np.zeros(self.data.shape)
        self._y = np.zeros(self.data.shape)
        self._l = np.zeros(self.data.shape)

        if _mu:
            self._mu = _mu
        else:
            self._mu = np.prod(self.data.shape) / (4 * self.norm_p(self.data, 2))

        self.mu_inv = 1 / self._mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.data.shape))

    @staticmethod
    def norm_p(data, power):
        """
            :param data: input data
            :param power: exponential power
            :return: p norm
            """
        return np.sum(np.power(data, power))

    @staticmethod
    def shrink(data, tau):
        """
            :param data: input data
            :param tau: threshold
            :return: shrinked data
            """
        return np.sign(data) * np.maximum((np.abs(data) - tau), np.zeros(data.shape))

    def svd_threshold(self, data, tau):
        """
            :param data: input data
            :param tau: threshold
            :return: svd recovery
            """
        _u, _s, _v = np.linalg.svd(data, full_matrices=False)
        return np.dot(_u, np.dot(np.diag(self.shrink(_s, tau)), _v))

    def fit(self, tol=None, max_iter=1000):
        """
            :param tol: tolerance
            :param max_iter: max iteration
            :return: low rank and sparse recovery
            """
        _iter = 0
        err = np.Inf
        sparse = self._s
        _yk = self._y
        low_rk = np.zeros(self.data.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.data), 2)

        while (err > _tol) and _iter < max_iter:
            low_rk = self.svd_threshold(
                self.data - sparse + self.mu_inv * _yk, self.mu_inv)
            sparse = self.shrink(
                self.data - low_rk + (self.mu_inv * _yk), self.mu_inv * self.lmbda)
            _yk = _yk + self._mu * (self.data - low_rk - sparse)
            err = self.norm_p(np.abs(self.data - low_rk - sparse), 2)
            _iter += 1

        self._l = low_rk
        self._s = sparse
        return low_rk, sparse
