"""
Derived module from sindybase.py for SINDy of PDE
"""
import numpy as np
from findiff import FinDiff
from .sindybase import SINDyBase


class SINDyPDE(SINDyBase):
    """
    Data-driven discovery of partial differential equations
    reference: http://advances.sciencemag.org/content/3/4/e1602614/tab-pdf
    """
    def fit(self, data, dt, _dx, poly_degree=2, space_deriv_order=2,
            cut_off=1e-3, deriv_acc=2, sample_rate=1.):
        """
        :param data: a numpy array or a dict of arrays, dynamics data to be processed
        :param dt: float, for temporal grid spacing
        :param _dx: float or list of floats, for spatial grid spacing
        :param poly_degree: degree of polynomials to be included in theta matrix
        :param space_deriv_order: maximum order of derivatives applied on spatial dimensions
        :param cut_off: the threshold cutoff value for sparsity
        :param deriv_acc: (positive) integer, derivative accuracy
        :param sample_rate: float, proportion of the data to use
        :return: a SINDyPDE model
        """
        if isinstance(data, np.ndarray):
            data = {'u': data}

        array_shape = data[list(data.keys())[0]].shape

        for i, k in enumerate(data.keys()):
            if len(data[k].shape) not in [2, 3, 4]:
                raise ValueError("SINDyPDE supports 2D, 3D and 4D arrays only, "
                                 "with the last dimension be the time dimension.")
            if data[k].shape != array_shape:
                raise ValueError("The arrays that you provide should have the same shapes!")

        if not isinstance(dt, float):
            raise ValueError("dt should of type float, specifying the temporal grids ...")

        if isinstance(_dx, list):
            if len(_dx) != len(array_shape) - 1:
                raise ValueError("The length of _dx does not match the shape of the array!")
        elif isinstance(_dx, float):
            _dx = [_dx] * (len(array_shape) - 1)
        else:
            raise ValueError("_dx could either be float or a list of floats ...")

        # compute time derivative
        d_dt = FinDiff(len(array_shape)-1, dt, 1, acc=deriv_acc)
        time_deriv = np.zeros((np.prod(array_shape), len(data.keys())))
        for i, k in enumerate(data.keys()):
            time_deriv[:, i] = d_dt(data[k]).flatten()
        print("Progress: finished computing time derivatives  ...")

        # compute spatial derivatives
        if space_deriv_order < 1 or not isinstance(space_deriv_order, int):
            raise ValueError('Order of the spatial derivative should be a positive integer!')

        space_deriv, space_deriv_desp = self.compute_spatial_derivatives(data, _dx,
                                                                         space_deriv_order,
                                                                         deriv_acc)
        print("Progress: finished computing spatial derivatives  ...")

        # prepare the library
        all_data, data_var_names = self.dict_to_2darray(data)
        extended_data, extended_data_desp = self.polynomial_expansion(all_data, degree=poly_degree,
                                                                      var_names=data_var_names)
        lib, self._desp = self.product_expansion(extended_data, space_deriv, extended_data_desp,
                                                 space_deriv_desp)

        # sparse regression
        if not isinstance(sample_rate, float) or sample_rate < 0 or sample_rate > 1:
            raise ValueError("sample rate must be a float number between 0-1!")
        idxs = np.random.choice(lib.shape[0], int(sample_rate*lib.shape[0]), replace=False)
        self._coef, _ = self.sparsify_dynamics(lib[idxs, :], time_deriv[idxs, :], cut_off,
                                               normalize=0)
        print("Progress: finished sparse regression  ...")

        return self

    def compute_spatial_derivatives(self, data, _dx, order, acc):
        """
        Compute the spatial derivatives presented in descriptions with proposed method
        :param data: a dict of arrays, value of state variables
        :param _dx: float or a list of floats, for spatial grid spacing
        :param order: maximum order of derivatives applied on spatial dimensions
        :param acc: accuracy of the derivatives
        :return: a dict of arrays of spatial derivatives
        """
        if isinstance(data, np.ndarray):
            data = {'u': data}

        if not isinstance(data, dict):
            raise ValueError("data should be a numpy array or a dict of arrays!")

        if not isinstance(_dx, float) and not isinstance(_dx, list):
            raise ValueError("_dx should be float or a list of floats specifying the "
                             "spatial grids!")

        array_shape = data[list(data.keys())[0]].shape
        exponents = self.get_ordered_poly_exponents(len(array_shape) - 1, order)
        space_deriv = []
        space_deriv_desp = []

        for k in data.keys():
            deriv_desp_dict = self.exponent_to_description(exponents, 'sub',
                                                           remove_zero_order=True,
                                                           as_dict=True, var_names=k)

            cur_space_deriv = np.zeros((np.prod(array_shape), len(deriv_desp_dict.keys())))
            for i, key in enumerate(deriv_desp_dict.keys()):
                _op = self.orders_to_op(deriv_desp_dict[key], _dx, acc)
                cur_space_deriv[:, i] = _op(data[k]).flatten()
                space_deriv_desp.append(key)

            space_deriv.append(cur_space_deriv)

        return np.hstack(space_deriv), space_deriv_desp

    @staticmethod
    def sampling_idxs(shape, space_sample_rate, time_sample_rate,
                      width_x=2, width_t=2, major='row'):
        """
        :param shape: numpy array shape information
        :param space_sample_rate: proportion of sampling for spatial dimensions
        :param time_sample_rate: proportion of sampling for time dimension
        :param width_x: specify the boundary for spatial sampling
        :param width_t: specify the boundary for temporal sampling
        :param major: either 'row' (row major) or 'col' (column major)
        :return: a list of indices of sampling points
        """
        n_space = np.prod(shape[:-1])
        n_time = shape[-1]
        # number of space sampling
        if space_sample_rate and 0 < space_sample_rate < 1:
            n_space_sample = int(n_space*space_sample_rate)
        else:
            n_space_sample = n_space
            print("The input of space_sample is None or not between 0-1, no spatial sampling ...")
        # number of time sampling
        if time_sample_rate and 0 < space_sample_rate < 1:
            n_time_sample = int(n_time*time_sample_rate)
        else:
            n_time_sample = n_time
            print("The input of time_sample is None or not between 0-1, no temporal sampling ...")

        # actual sampling
        idxs_list = []
        for i in np.arange(len(shape)-1):
            idxs_list.append(list(np.random.choice(np.arange(width_x, shape[i]-width_x),
                                                   n_space_sample))*n_time_sample)

        time_pts = np.linspace(width_t, shape[-1]-width_t-1, n_time_sample)  # uniform sampling on t
        idxs_list.append([int(x) for _ in np.arange(n_space_sample) for x in time_pts])

        if major == 'row':
            return idxs_list
        elif major == 'col':
            return list(map(list, zip(*idxs_list)))
        else:
            raise ValueError("the major argument can only be either 'row' or 'col'!")

    @staticmethod
    def product_expansion(_u, _v, u_desp, v_desp, included=True):
        """
        :param _u: a 2D numpy array of features stored in each column
        :param _v: another 2D numpy array of features stored in each column
        :param u_desp: a list of descriptions for each columns of u
        :param v_desp: a list of descriptions for each columns of v
        :param included: a boolean value, indicates whether to include original columns or not
        :return: a 2D numpy array of product features and corresponding descriptions
        """
        if _u.ndim == 1:
            _u = _u[:, np.newaxis]
        if _v.ndim == 1:
            _v = _v[:, np.newaxis]

        if _u.ndim != 2 or _v.ndim != 2:
            raise ValueError("input arrays should be 2D!")

        if not isinstance(u_desp, list) or not isinstance(v_desp, list):
            raise ValueError("descriptions (u_desp & v_desp) should be of list type!")

        if _u.shape[1] != len(u_desp) or _v.shape[1] != len(v_desp):
            raise ValueError("the array and the description should of the same length!")

        u_one_flag = False  # if there's a column of ones in u
        v_one_flag = False  # if there's a column of ones in v

        prod_feat = []
        prod_desp = []
        for i in np.arange(len(u_desp)):
            for j in np.arange(len(v_desp)):
                prod_feat.append(_u[:, i]*_v[:, j])
                if u_desp[i] == '1' and v_desp[j] == '1':
                    prod_desp.append('1')
                    u_one_flag = True
                    v_one_flag = True
                elif u_desp[i] == '1':
                    prod_desp.append(v_desp[j])
                    u_one_flag = True
                elif v_desp[j] == '1':
                    prod_desp.append(u_desp[i])
                    v_one_flag = True
                else:
                    prod_desp.append(u_desp[i] + v_desp[j])

        prod_feat = np.vstack(prod_feat).T

        # add original features
        if included:
            if not u_one_flag:
                prod_feat = np.concatenate([_v, prod_feat], axis=1)
                prod_desp = v_desp + prod_desp
            if not v_one_flag:
                prod_feat = np.concatenate([_u, prod_feat], axis=1)
                prod_desp = u_desp + prod_desp

        return prod_feat, prod_desp

    @staticmethod
    def orders_to_op(order, _dx, acc):
        """
        :param order: orders of the derivative
        :param _dx: a float or a list of floats, grid spacings
        :param acc: a positive integer, accuracy of the derivatives
        :return:
        """
        if not isinstance(order, list):
            raise ValueError("order argument must be a list of positive integers!")

        if isinstance(_dx, list):
            assert len(order) == len(_dx), "length of order and _dx are not the same!"
        elif isinstance(_dx, float):
            _dx = [_dx] * len(order)
        else:
            raise ValueError("dx must be a float or a list of floats, "
                             "specifying the grid spacing information!")

        args = [(int(i), _dx[i], order[i]) for i in np.arange(len(order)) if order[i] != 0]
        return FinDiff(*args, acc=acc)

    @staticmethod
    def dict_to_2darray(data_dict):
        """
        This function flattens all the arrays in a dict and stack them together
        :param data_dict: a dict of arrays
        :return: a 2D numpy array and a list of descriptions
        """
        if not isinstance(data_dict, dict):
            raise ValueError("data should be of dict type!")

        desp = list(data_dict.keys())
        data = [data_dict[k].flatten() for k in desp]

        return np.vstack(data).T, desp
