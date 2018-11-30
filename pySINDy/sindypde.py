"""
Derived module from sindybase.py for SINDy of PDE
"""
import numpy as np
from .sindybase import SINDyBase
from findiff import FinDiff


class SINDyPDE(SINDyBase):
    """
    Data-driven discovery of partial differential equations
    reference: http://advances.sciencemag.org/content/3/4/e1602614/tab-pdf
    """
    def fit(self, data, dt, dx, poly_degree=2, space_deriv_order=2,
            cut_off=1e-3, deriv_acc=2):
        """
        :param data: dynamics data to be processed
        :param dt: float, for temporal grid spacing
        :param dx: float, for spatial grid spacing
        :param poly_degree: degree of polynomials to be included in theta matrix (all derivatives remain degree 1)
        :param space_deriv_order: maximum order of derivatives applied on spatial dimensions
        :param cut_off: the threshold cutoff value for sparsity
        :param deriv_acc: (positive) integer, derivative accuracy
        :return: a SINDyPDE model
        """
        if len(data.shape) not in [2, 3, 4]:
            raise ValueError('SINDyPDE supports 2D, 3D and 4D datasets only, '
                             'with the last dimension be the time dimension.')

        # compute time derivative
        d_dt = FinDiff(data.ndim-1, dt, 1, acc=deriv_acc)
        time_deriv = d_dt(data)
        print("Progress: finished computing time derivatives  ...")

        # compute spatial derivatives
        if space_deriv_order < 1 or not isinstance(space_deriv_order, int):
            raise ValueError('Order of the spatial derivative should be a positive integer!')

        space_deriv = self.compute_spatial_derivatives(data, dx, space_deriv_order)

        '''
        exponents = self.get_ordered_poly_exponents(data.ndim - 1, space_deriv_order)
        spatial_deriv_desp_dict = self.exponent_to_description(exponents, 'sub', remove_zero_order=True, as_dict=True)
        spatial_derivatives_dict = self.compute_spatial_derivatives(data, dx, spatial_deriv_desp_dict,
                                                                    space_diff, width_x)
        # organize the spatial derivatives as the same as in descriptions
        spatial_derivatives = np.vstack([spatial_derivatives_dict[key].flatten() for key
                                         in spatial_deriv_desp_dict.keys()]).T
        print("Progress: finished computing spatial derivatives  ...")

        # adjust size of data to match with spatial derivatives
        if space_diff == 'poly':
            data = data[[slice(width_x, -width_x)]*(data.ndim-1)+[slice(None)]]

        # prepare the library
        extended_data, extended_data_desp = self.polynomial_expansion(data.flatten(), degree=poly_degree)
        lib, self._desp = self.product_expansion(extended_data, spatial_derivatives,
                                                 extended_data_desp,
                                                 list(spatial_deriv_desp_dict.keys()))

        # sparse regression
        self._coef, _ = self.sparsify_dynamics(lib, x_dot.flatten(), cut_off)'''

        return self

    def compute_spatial_derivatives(self, data, dx, order):
        """
        Compute the spatial derivatives presented in descriptions with proposed method
        :param data: value of state variables
        :param dx: float, for spatial grid spacing
        :param order: order of derivatives applied on spatial dimensions
        :return: array of spatial derivatives together with its descriptions
        """
        exponents = self.get_ordered_poly_exponents(data.ndim - 1, order)
        deriv_desp_dict = self.exponent_to_description(exponents, 'sub',
                                                       remove_zero_order=True,
                                                       as_dict=True)

        spatial_derivatives = {}
        max_order = 0
        for key, value in deriv_desp_dict.items():
            if np.any(value):
                spatial_derivatives[key] = None
                max_order = np.max([max_order, np.sum(value)])

        tmp_dict = {'root': [-1, max_order, data]}
        queue = ['root']

        # BFS: reduce the derivative dependencies as much as possible
        while len(queue):
            old_key = queue.pop()
            # compute all possible derivatives at current depth, if not a leaf
            cur_depth = tmp_dict[old_key][0] + 1
            if cur_depth < data.ndim - 1:
                old_data = tmp_dict[old_key][2]
                if diff_method == 'fd':
                    for i in np.arange(tmp_dict[old_key][1]+1):
                        new_key = old_key + '/%d' % i
                        new_order = tmp_dict[old_key][1] - i
                        if i:
                            new_data = self.finite_difference(old_data, dx, dim=cur_depth, order=i)
                            tmp_dict[new_key] = [cur_depth, new_order, new_data]
                        else:
                            tmp_dict[new_key] = [cur_depth, new_order, data]

                        queue = [new_key] + queue

                elif diff_method == 'poly':
                    xgrid = np.arange(old_data.shape[cur_depth])*dx
                    order_list = [i for i in np.arange(1, tmp_dict[old_key][1]+1)]
                    all_new_data = self.polynomial_difference(old_data, xgrid, dim=cur_depth,
                                                                  order=order_list, degree=width)
                    # add the original one
                    if not isinstance(all_new_data, list):
                        all_new_data = [data, all_new_data]
                    else:
                        all_new_data = [data] + all_new_data

                    for i in np.arange(tmp_dict[old_key][1]+1):
                        new_key = old_key + '/%d' % i
                        new_order = tmp_dict[old_key][1] - i
                        tmp_dict[new_key] = [cur_depth, new_order, all_new_data[i]]
                        queue = [new_key] + queue

                else:
                    raise ValueError("space_diff can be either 'df' (finite difference) "
                                     "or 'poly'(polynomial interpolation) only!")

            # trim the dimensions & write back to spatial_derivatives dict
            mask_slice = slice(width, -width)
            for key, value in spatial_derivatives.items():
                dir_key = self.exponent_to_dir(deriv_desp_dict[key])
                if diff_method == 'fd':
                    idxs = [slice(None)]*data.ndim
                elif diff_method == 'poly':
                    conflicts = [True if x == y else False for idx, (x, y) in
                                 enumerate(zip(data.shape, tmp_dict[dir_key][2].shape))]
                    idxs = [mask_slice if x else slice(None) for x in conflicts]
                    idxs[-1] = slice(None)  # we don't want to change time dimension
                else:
                    raise ValueError("space_diff can be either 'df' (finite difference) "
                                     "or 'poly'(polynomial interpolation) only!")

                spatial_derivatives[key] = tmp_dict[dir_key][2][idxs]

            return spatial_derivatives

    @staticmethod
    def sampling_idxs(shape, space_sample_rate, time_sample_rate, acc_x=2, acc_t=2, major='row'):
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
    def product_expansion(u, v, u_desp, v_desp, included=True):
        """
        :param u: a 2D numpy array of features stored in each column
        :param v: another 2D numpy array of features stored in each column
        :param u_desp: a list of descriptions for each columns of u
        :param v_desp: a list of descriptions for each columns of v
        :param included: a boolean value, indicates whether to include original columns or not
        :return: a 2D numpy array of product features and corresponding descriptions
        """
        if u.ndim == 1:
            u = u[:, np.newaxis]
        if v.ndim == 1:
            v = v[:, np.newaxis]

        if u.ndim != 2 or v.ndim != 2:
            raise ValueError("input arrays should be 2D!")

        if not isinstance(u_desp, list) or not isinstance(v_desp, list):
            raise ValueError("descriptions (u_desp & v_desp) should be of list type!")

        if u.shape[1] != len(u_desp) or v.shape[1] != len(v_desp):
            raise ValueError("the array and the description should of the same length!")

        u_one_flag = False  # if there's a column of ones in u
        v_one_flag = False  # if there's a column of ones in v

        prod_feat = []
        prod_desp = []
        for i in np.arange(len(u_desp)):
            for j in np.arange(len(v_desp)):
                prod_feat.append(u[:, i]*v[:, j])
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
                prod_feat = np.concatenate([v, prod_feat], axis=1)
                prod_desp = v_desp + prod_desp
            if not v_one_flag:
                prod_feat = np.concatenate([u, prod_feat], axis=1)
                prod_desp = u_desp + prod_desp

        return prod_feat, prod_desp

    @staticmethod
    def exponent_to_dir(exponent):
        """
        :param exponent: a list of exponents
        :return: a string of structured directories

        Note: only used in compute_spatial_derivatives. eg. [1, 2, 0] --> "root/1/2/0"
        """
        dir_key = 'root'
        for i in np.arange(len(exponent)):
            dir_key += '/%d' % exponent[i]

        return dir_key