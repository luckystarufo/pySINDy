# pylint: skip-file

from unittest import TestCase
from ..isindy import ISINDy
import numpy as np
import scipy.integrate as integrate
from scipy.linalg import null_space
import warnings


class TestISINDy(TestCase):
    @staticmethod
    def subtilis_competence(t, _s):
        _s_1 = _s[0]

        a1 = 0.6

        v1 = a1 - (1.5 * _s_1) / (0.3 + _s_1)

        return [v1]

    def test_name_isindy(self):
        model = ISINDy('omg')
        self.assertEqual(model.name, 'omg')

    def test_shape_1d(self):
        warnings.simplefilter("ignore")
        model = ISINDy()
        data = np.random.rand(10)
        dt = 0.1
        model.fit(data, dt)
        assert True

    def test_shape_2d(self):
        model = ISINDy()
        data = np.random.rand(2, 10)
        dt = 0.1
        model.fit(data, dt)
        assert True

    def test_shape_3d(self):
        model = ISINDy()
        data = np.random.rand(2,2,10)
        dt = 0.1
        model.fit(data, dt)
        assert True

    def test_expand_descriptions_non_string_name(self):
        model = ISINDy()
        data = np.random.rand(2, 10)
        _, extended_desp = np.array(model.polynomial_expansion(data.T))
        with self.assertRaises(ValueError):
            model.expand_descriptions(extended_desp, var_name=[1, 2])

    def test_expand_descriptions(self):
        model = ISINDy()
        data = np.random.rand(2, 10)
        _, extended_desp = np.array(model.polynomial_expansion(data.T))
        extended_desp = model.expand_descriptions(extended_desp, var_name='y')
        expected = ['1', 'u0', 'u1', 'y', 'u0y', 'u1y']
        self.assertListEqual(expected, extended_desp)

    def test_build_theta_matrix_3d(self):
        model = ISINDy()
        data = np.random.rand(2, 2, 10)
        d_vec = np.random.rand(2, 10)
        with self.assertRaises(ValueError):
            model.build_theta_matrix(data, d_vec)

    def test_build_theta_matrix_2d_dvec(self):
        model = ISINDy()
        data = np.random.rand(2, 10)
        d_vec = np.random.rand(2, 2, 10)
        with self.assertRaises(ValueError):
            model.build_theta_matrix(data, d_vec)

    def test_build_theta_matrix(self):
        model = ISINDy()
        data = np.random.rand(10, 2)
        d_vec = np.random.rand(10)
        xdx = data[:, 0] * d_vec
        ydx = data[:, 1] * d_vec
        expected = np.empty((10, 4))
        expected[:, 0] = data[:, 0]
        expected[:, 1] = data[:, 1]
        expected[:, 2] = xdx
        expected[:, 3] = ydx
        np.testing.assert_allclose(expected, model.build_theta_matrix(data, d_vec))

    def test_adm_initvary(self):
        n = 1
        dt = 0.1
        tspan = np.arange(0, 5 + dt, dt)
        len_t = len(tspan)
        np.random.seed(0)
        sinit = np.random.rand(n)
        sol = integrate.solve_ivp(TestISINDy.subtilis_competence, [0, len_t], sinit, t_eval=tspan, rtol=1e-7, atol=1e-7)
        xt = sol.y
        dxt = TestISINDy.subtilis_competence(tspan, xt)[0]
        xt = xt.T
        model = ISINDy()
        extended_data, _ = np.array(model.polynomial_expansion(xt, degree=5, var_names=None))
        theta_k = model.build_theta_matrix(extended_data, dxt.flatten())
        null_space_k = null_space(theta_k)
        _coef = np.array(ISINDy.adm_initvary(null_space_k, 5e-3, 1000, 2e-3))
        expected = np.array([0.1295, -0.6474,
                             0, 0, 0, 0, -0.2158,
                             -0.7194, 0, 0, 0, 0])
        np.testing.assert_allclose(np.absolute(expected), np.around(np.absolute(_coef), decimals=4))

    def test_smoothing_initial_large_value(self):
        model = ISINDy()
        vec = np.array([1e10, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        after_smoothing = model.smoothing(vec)
        expected = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_allclose(after_smoothing, expected)

    def test_smoothing(self):
        model = ISINDy()
        vec = np.array([1, 2, 1e10, 4, 5, 6, 1e10, 8, 9, 1e10])
        after_smoothing = model.smoothing(vec)
        expected = np.array([1, 2, 2, 4, 5, 6, 6, 8, 9, 9])
        np.testing.assert_allclose(after_smoothing, expected)
