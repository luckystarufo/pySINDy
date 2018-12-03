"""
Derived module from sindybase.py for implicit SINDy
"""
import numpy as np
from .sindybase import SINDyBase
from findiff import FinDiff


class ISINDy(SINDyBase):
    """
    Implicit Sparse Identification of Nonlinear Dynamics
    (Inferring biological networks by sparse identification of nonlinear dynamics)
    reference: https://arxiv.org/pdf/1605.08368.pdf
    """
    def fit(self, data, dt, poly_degree=2, deriv_acc=2):
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

        # polynomial expansion of orginal data
        var_names = ['u%d' % i for i in np.arange(n)]
        extended_data, extended_desp = self.polynomial_expansion(data.T, degree=poly_degree,
                                                                 var_names=var_names)

        # set the descriptions
        self._desp = self.expand_description(extended_desp, 'uk_{t}')

        # compute sparse coefficients
        self._coef = np.zeros((len(self._desp), n))
        for k in np.arange(n):
            # formulate theta1, theta2, theta3 ...
            theta_k = self.build_theta_matrix(extended_data, x_dot[:, k])

            # compute null spaces
            null_space_k = self.nullify(theta_k)

            # ADM
            self._coef[:, k] = self.adm(null_space_k)

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
    def nullify(theta):
        """
        :param theta: a 2D numpy array
        :return: a 2D numpy array representing the null space of theta
        """
        return None

    @staticmethod
    def adm(null_space):
        """
        :param null_space:
        :return: find a sparse representation
        """
        return None