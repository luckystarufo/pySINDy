"""
Base Module for SINDy: 'fit' method must be implemented in inherited classes
"""
import numpy as np


class SINDyBase(object):
    """
    Sparse Identification of Nonlinear Dynamics base class
    """
    def __init__(self, name='SINDy model'):
        self.name = name

        self._coef = None
        self._desp = None

    @property
    def coefficients(self):
        """
        :return: get the coefficients of the model
        """
        return self._coef

    @property
    def descriptions(self):
        """
        :return: get the items we need to fit the data
        """
        return self._desp

    def fit(self, data):
        """
        Abstract method to fit the snapshot matrix, it has to be
        implemented in subclasses
        :param data: the snapshot matrix
        :return: None
        """
        raise NotImplementedError('Subclass must implement abstract method {}.fit'.format(
            self.__class__.__name__))

    @staticmethod
    def finite_difference(data, _dx, order=1, dim=0):
        """
        Take derivative using 2nd order finite difference method
        :param data: a tensor to be differentiated
        :param _dx: grid spacing, assume to be uniform
        :param order: the order of the derivative to be applied
        :param dim: the dimension to be taken the derivative
        :return: a tensor after differentiation
        """
        data = np.squeeze(data)
        if dim >= data.ndim:
            raise ValueError('The selected dim should be less than #of dimensions of data!')

        data_shape = data.shape
        _n = data_shape[dim]
        idxs = [slice(None)]*len(data_shape)
        data_dx = np.zeros(data_shape)

        if order == 1:

            for i in np.arange(1, _n-1):
                idxs[dim] = i
                data_dx[idxs] = (np.take(data, i+1, dim) - np.take(data, i-1, dim))/(2*_dx)

            idxs[dim] = 0
            data_dx[idxs] = (-3.0/2*np.take(data, 0, dim) + 2*np.take(data, 1, dim) -
                             np.take(data, 2, dim)/2)/_dx

            idxs[dim] = _n - 1
            data_dx[idxs] = (3.0/2*np.take(data, _n-1, dim) - 2*np.take(data, _n-2, dim) +
                             np.take(data, _n-3, dim)/2)/_dx

        elif order == 2:

            for i in np.arange(1, _n-1):
                idxs[dim] = i
                data_dx[idxs] = (np.take(data, i+1, dim) - 2*np.take(data, i, dim) +
                                 np.take(data, i-1, dim))/_dx**2

            idxs[dim] = 0
            data_dx[idxs] = (2*np.take(data, 0, dim) - 5*np.take(data, 1, dim) +
                             4*np.take(data, 2, dim) - np.take(data, 3, dim))/_dx**2

            idxs[dim] = _n - 1
            data_dx[idxs] = (2*np.take(data, _n-1, dim) - 5*np.take(data, _n-2, dim) +
                             4*np.take(data, _n-3, dim) - np.take(data, _n-4, dim))/_dx**2

        elif order == 3:

            for i in np.arange(2, _n-2):
                idxs[dim] = i
                data_dx[idxs] = (np.take(data, i+2, dim)/2 - np.take(data, i+1, dim) +
                                 np.take(data, i-1, dim) - np.take(data, i-2, dim)/2)/_dx**3

            idxs[dim] = 0
            data_dx[idxs] = (-2.5*np.take(data, 0, dim) + 9*np.take(data, 1, dim) -
                             12*np.take(data, 2, dim) + 7*np.take(data, 3, dim) -
                             1.5*np.take(data, 4, dim))/_dx**3

            idxs[dim] = 1
            data_dx[idxs] = (-2.5*np.take(data, 1, dim) + 9*np.take(data, 2, dim) -
                             12*np.take(data, 3, dim) + 7*np.take(data, 4, dim) -
                             1.5*np.take(data, 5, dim))/_dx**3

            idxs[dim] = _n - 1
            data_dx[idxs] = (2.5 * np.take(data, _n-1, dim) - 9 * np.take(data, _n-2, dim) +
                             12 * np.take(data, _n-3, dim) - 7 * np.take(data, _n-4, dim) +
                             1.5 * np.take(data, _n-5, dim)) /_dx**3

            idxs[dim] = _n - 2
            data_dx[idxs] = (2.5*np.take(data, _n-2, dim) - 9*np.take(data, _n-3, dim) +
                             12*np.take(data, _n-4, dim) - 7*np.take(data, _n-5, dim) +
                             1.5*np.take(data, _n-6, dim))/_dx**3

        elif order > 3:

            return SINDyBase.finite_difference(SINDyBase.finite_difference(data, _dx, 3, dim),
                                               _dx, order-3, dim)

        else:
            raise ValueError('order of the derivative should be a positive integer!')

        return data_dx

    @staticmethod
    def pointwise_polynomial_difference(data, xgrid, order=1, degree=2, index=None):
        """
        :param data: a 1D flattened vector represents nearby function values
        :param xgrid: grid information
        :param order: the order of the derivatives to be applied
        :param index: index of the derivative to take
        :param degree: degree of polynomial to use
        :return: value of derivative at this point
        """
        if isinstance(order, int):
            order = [order]

        data = data.flatten()
        _n = len(data)
        if index is None:
            index = int((_n - 1)/2)

        # Fit to a Chebyshev polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(xgrid, data, degree)
        return np.array([poly.deriv(m=order[i])(xgrid[index]) for i in np.arange(len(order))])

    @staticmethod
    def polynomial_difference(data, xgrid, order=1, dim=0, degree=2):
        """
        Taking derivatives using Chebyshev polynomial interpolation
        :param data: a tensor to be differentiated
        :param xgrid: grid information
        :param order: an integer, or a list of orders of the derivative to be applied
        :param dim: the dimension to be taken the derivative
        :param degree: degree of polynomials to be used for interpolation
        :return: a list of tensors after differentiation, same length of order
        """
        data = np.squeeze(data)
        if dim >= data.ndim:
            raise ValueError('The selected dim should be less than #of dimensions of data!')
        if dim < 0:
            dim = data.ndim + dim

        if isinstance(order, int):
            order = [order]

        data_shape = data.shape
        _n = data_shape[dim]
        idxs = [slice(None)]*len(data_shape)
        new_data_shape = list(data_shape)
        data_slice_shape = list(data_shape)
        new_data_shape[dim] -= 2*degree
        data_slice_shape[dim] = 1
        data_dx = [np.zeros(tuple(new_data_shape))]*len(order)

        if _n != len(xgrid):
            raise ValueError('Grids information does not match with the data!')

        for j in np.arange(degree, _n - degree):
            pts = np.arange(j - degree, j + degree)
            idxs[dim] = slice(j - degree, j + degree)
            pos = (dim, ) + tuple(np.arange(0, dim)) + tuple(np.arange(dim+1, data.ndim))
            batch_data = np.transpose(data[idxs], pos).reshape((2*degree, -1))
            data_dx_tmp = np.zeros((1, batch_data.shape[1], len(order)))
            for k in np.arange(batch_data.shape[1]):
                deriv = SINDyBase.pointwise_polynomial_difference(batch_data[:, k].flatten(),
                                                                  xgrid[pts], order=order,
                                                                  degree=degree)
                data_dx_tmp[0, k, :] = deriv

            for i in np.arange(len(order)):
                idxs[dim] = j - degree
                data_dx[i][idxs] = np.squeeze(data_dx_tmp[..., i].reshape(tuple(data_slice_shape)))

        if len(order) == 1:
            return data_dx[0]

        return data_dx

    @staticmethod
    def get_poly_exponents(nfeat, degree=1):
        """
        :param nfeat: number of original features
        :param degree: maximum degree of the polynomials
        :return: a 2D array consists of the exponents
        """
        if nfeat == 0:
            yield ()
        else:
            for _x in np.arange(degree+1):
                for _t in SINDyBase.get_poly_exponents(nfeat - 1, degree):
                    if sum(_t) + _x <= degree:
                        yield _t + (_x,)

    @staticmethod
    def get_ordered_poly_exponents(nfeat, degree=1, remove_zero_order=False):
        """
        :param nfeat: number of original features
        :param degree: maximum degree of the polynomials
        :param remove_zero_order: boolean value, indicate whether to remove the zero order term
        :return: a 2D array consists of ordered exponents according to the sum
        """
        exponents = np.array(list(SINDyBase.get_poly_exponents(nfeat, degree)))
        all_exponents = exponents[np.argsort(np.sum(exponents, axis=1))]
        if remove_zero_order:
            return all_exponents[1:, :]
        return all_exponents

    @staticmethod
    def polynomial_expansion(data, degree=1, remove_zero_order=False, var_names=None):
        """
        :param data: a 2D numpy array of original features stored in each column
        :param degree: degree of polynomials of features to be expanded
        :param remove_zero_order: boolean value, indicate whether to remove the zero order term
        :param var_names: variable names, default as None
        :return: a tensor consists of extended features, and corresponding descriptions
        """
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        if len(data.shape) > 2:
            raise ValueError("The input array is not 2D!")

        # extended features
        nfeat = data.shape[-1]
        exponents = SINDyBase.get_ordered_poly_exponents(nfeat, degree, remove_zero_order)
        result = np.array([np.prod([data[:, k] ** e[k] for k in np.arange(nfeat)],
                                   axis=0) for e in exponents]).T

        # descriptions of each extended feature
        desp = SINDyBase.exponent_to_description(exponents, 'sup', remove_zero_order,
                                                 var_names=var_names)

        return result, desp

    @staticmethod
    def threshold_ls(mtx, _b, cut_off=1e-3, max_iter=10, normalize=0):
        """
        Find the sparse coefficients of fit using threshold least squares
        :param mtx: the training theta matrix of shape (M, N)
        :param _b: a vector or an array of shape (M,) or (M, K)
        :param cut_off: the threshold cutoff value
        :param max_iter: # of iterations
        :param normalize: normalization methods, default as 0 (no normalization)
        :return: coefficients of fit
        """
        if len(_b.shape) == 1:
            _b = _b[:, np.newaxis]
        dim = _b.shape[-1]

        # normalize each column of mtx
        if normalize != 0:
            w_col_norms = np.linalg.norm(mtx, ord=normalize, axis=0)
            b_col_norms = np.linalg.norm(_b, ord=normalize, axis=0)
            mtx = mtx / w_col_norms[np.newaxis, :]
            _b = _b / b_col_norms[np.newaxis, :]

        _w = np.linalg.lstsq(mtx, _b, rcond=None)[0]
        for _ in np.arange(max_iter):
            small_inds = np.abs(_w) <= cut_off
            _w[small_inds] = 0
            if np.all(np.sum(np.abs(_w), axis=0)):
                for ind in np.arange(dim):
                    big_inds = ~small_inds[:, ind]
                    _w[big_inds, ind] = np.linalg.lstsq(mtx[:, big_inds], _b[:, ind], rcond=None)[0]
            else:
                break

        if normalize != 0:
            _w = _w * w_col_norms[:, np.newaxis]
            _w = _w / b_col_norms[np.newaxis, :]

        return _w

    @staticmethod
    def sparsify_dynamics(mtx, _b, init_tol, max_iter=25, thresh_iter=10,
                          l0_penalty=None, split=0.8, normalize=0):
        """
        :param mtx: the theta matrix of shape (M, N)
        :param _b: a vector or an array of shape (M,) or (M, K)
        :param init_tol: maximum tolerance (cut_off value)
        :param max_iter: maximum iteration of the outer loop
        :param thresh_iter: maximum iteration for threshold least squares
        :param l0_penalty: penalty factor for nonzero coefficients
        :param split: proportion of the training set
        :param normalize: normalization methods, default as 0 (no normalization)
        :return: the best coefficients of fit
        """
        if mtx.ndim != 2:
            raise ValueError('mtx is not a 2D numpy array!')
        if _b.ndim == 1:
            _b = _b[:, np.newaxis]
        elif _b.ndim > 2:
            raise ValueError('b is not a 1D/2D numpy array!')

        # split the data
        np.random.seed(12345)
        _n = mtx.shape[0]
        train = np.random.choice(_n, int(_n*split), replace=False)
        test = [x for x in np.arange(_n) if x not in train]
        train_mtx = mtx[train, :]
        test_mtx = mtx[test, :]
        train_b = _b[train, :]
        test_b = _b[test, :]
        # set up initial tolerance, l0 penalty, best error, etc.
        if l0_penalty is None:
            # l0_penalty = 0.001*np.linalg.cond(mtx)
            l0_penalty = np.linalg.norm(test_b) / len(test)

        tol = d_tol = float(init_tol)

        # no sparsity constraints
        w_best = np.linalg.lstsq(train_mtx, train_b, rcond=None)[0]
        err_best = np.linalg.norm(test_b - test_mtx.dot(w_best), 2) + \
                   l0_penalty*np.count_nonzero(w_best)
        tol_best = 0.
        imp_flag = True
        for i in np.arange(max_iter):
            _w = SINDyBase.threshold_ls(train_mtx, train_b, tol, thresh_iter, normalize)
            err = np.linalg.norm(test_b - test_mtx.dot(_w), 2) + l0_penalty*np.count_nonzero(_w)
            if err < err_best:
                err_best = err
                w_best = _w
                tol_best = tol
                tol += d_tol
                imp_flag = False
            else:
                # tol = max([0, tol - d_tol])
                tol = max([0, tol - 2*d_tol])
                # d_tol /= 2
                d_tol = 2 * d_tol/(max_iter - i)
                tol = tol + d_tol

        if imp_flag:
            print('cutoff value maybe too small/large to threshold ....')

        return w_best, tol_best

    @staticmethod
    def exponent_to_description(exponents, typ='sup', remove_zero_order=False, as_dict=False,
                                var_names=None):
        """
        :param exponents: a 2D numpy array of exponents
        :param typ: a string, can be either 'sup' (superscript) or 'sub' (subscript)
        :param remove_zero_order: boolean value, indicate whether to remove the zero order term
        :param as_dict: whether to include exponents in the descriptions as a dict
        :param var_names: variable name, default to be None
        :return: a list or a dict (depends on 'as_dict') of descriptions of corresponding exponents
        """
        if not isinstance(exponents, np.ndarray) or exponents.ndim != 2:
            raise ValueError("exponents must be a 2D numpy array!")

        desp = []
        desp_dict = {}
        _m, _n = exponents.shape

        if typ == 'sup':
            if var_names is not None:
                assert isinstance(var_names, list), "var_names must be a list of strings when " \
                                                    "typ =='sup'!"
                assert len(var_names) == _n, "length of var_names doesn't match with exponents!"
            else:
                var_names = ['u%d' % i for i in np.arange(_n)]

            for i in np.arange(_m):
                if np.any(exponents[i, :]):
                    # exist nonzero element
                    key = ''
                    for j in np.arange(_n):
                        if exponents[i, j] == 1:
                            key += var_names[j]
                        elif exponents[i, j]:
                            key += (var_names[j] + '^{%d}' % exponents[i, j])

                    desp.append(key)
                    desp_dict[key] = exponents[i, :].tolist()

                elif not remove_zero_order:
                    key = '1'
                    desp.append(key)
                    desp_dict[key] = exponents[i, :].tolist()

        elif typ == 'sub':
            # name of each dimension
            # (with xyz coordinates as default except for higher dimensional cases)
            if var_names is not None:
                assert isinstance(var_names, str), "var_names must be of type str when " \
                                                   "typ == 'sub'!"
            else:
                var_names = 'u'

            if _n == 1:
                dim_strs = ['x']
            elif _n == 2:
                dim_strs = ['x', 'y']
            elif _n == 3:
                dim_strs = ['x', 'y', 'z']
            else:
                dim_strs = ['x%d' % i for i in np.arange(_n)]

            for i in np.arange(_m):
                if np.any(exponents[i, :]):
                    # exist nonzero element
                    key = (var_names + '_{')
                    for j in np.arange(_n):
                        key += dim_strs[j]*exponents[i, j]
                    key += '}'
                    desp.append(key)
                    desp_dict[key] = exponents[i, :].tolist()

                elif not remove_zero_order:
                    key = 'u'
                    desp.append(key)
                    desp_dict[key] = exponents[i, :].tolist()

        else:
            raise ValueError("type argument should be either 'sub' or 'sup'!")

        # which type of description to return
        if as_dict:
            return desp_dict

        return desp
