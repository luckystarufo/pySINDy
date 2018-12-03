from unittest import TestCase
from pySINDy.sindybase import SINDyBase
import numpy as np


class TestSINDyBase(TestCase):
    def test_name_default(self):
        model = SINDyBase()
        assert model.name == 'SINDy model'

    def test_fit(self):
        model = SINDyBase()
        sample_data = np.random.rand(2, 10)
        with self.assertRaises(NotImplementedError):
            model.fit(sample_data)

    def test_finite_difference_order(self):
        model = SINDyBase()
        data = np.array([1, 2, 3, 4, 5, 6])
        dx = 0.1
        with self.assertRaises(ValueError):
            model.finite_difference(data, dx, order=1.2)

        first_order_derivative = model.finite_difference(data, dx, order=1)
        second_order_derivative = model.finite_difference(data, dx, order=2)
        third_order_derivative = model.finite_difference(data, dx, order=3)
        fourth_order_derivative = model.finite_difference(data, dx, order=4)
        expected_first_order = np.array([10, 10, 10, 10, 10, 10])
        expected_second_order = np.array([0, 0, 0, 0, 0, 0])
        expected_third_order = np.array([0, 0, 0, 0, 0, 0])
        expected_fourth_order = np.array([0, 0, 0, 0, 0, 0])

        np.testing.assert_allclose(first_order_derivative, expected_first_order)
        np.testing.assert_allclose(second_order_derivative, expected_second_order)
        np.testing.assert_allclose(third_order_derivative, expected_third_order)
        np.testing.assert_allclose(fourth_order_derivative, expected_fourth_order)

    def test_finite_difference_dim(self):
        model = SINDyBase()
        data = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        dx = 0.1

        with self.assertRaises(ValueError):
            model.finite_difference(data, dx, dim=2)

        derivative_dim0 = model.finite_difference(data, dx, dim=0)
        derivative_dim1 = model.finite_difference(data, dx, dim=1)
        derivative_dim_last = model.finite_difference(data, dx, dim=-1)

        expected_dim0 = np.array([[20, 20, 20], [20, 20, 20], [20, 20, 20]])
        expected_dim1 = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])

        np.testing.assert_allclose(derivative_dim0, expected_dim0)
        np.testing.assert_allclose(derivative_dim1, expected_dim1)
        np.testing.assert_allclose(derivative_dim_last, expected_dim1)

    def test_pointwise_polydiff_order(self):
        model = SINDyBase()
        data = np.array([1, 1.2, 1.8, 2.8, 4.2])
        xgrid = np.arange(5)*0.1

        first_deriv = model.pointwise_polynomial_difference(data, xgrid, order=1)
        second_deriv = model.pointwise_polynomial_difference(data, xgrid, order=2)
        expected_first = 8
        expected_second = 40

        np.testing.assert_allclose(first_deriv, expected_first)
        np.testing.assert_allclose(second_deriv, expected_second)


    def test_polynomial_difference_dim_2d(self):
        model = SINDyBase()
        data = np.array([[1, 2, 3, 4, 5],
                         [3, 4, 5, 6, 7],
                         [5, 6, 7, 8, 9],
                         [7, 8, 9, 10, 11],
                         [9, 10, 11, 12, 13]])
        xgrid = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        with self.assertRaises(ValueError):
            model.polynomial_difference(data, xgrid, dim=2)

        derivative_dim0 = model.polynomial_difference(data, xgrid, dim=0)
        derivative_dim1 = model.polynomial_difference(data, xgrid, dim=1)
        derivative_dim_last = model.polynomial_difference(data, xgrid, dim=-1)
        expected_dim0 = np.array([20, 20, 20, 20, 20])
        expected_dim1 = np.array([10, 10, 10, 10, 10])

        np.testing.assert_allclose(np.squeeze(derivative_dim0), np.squeeze(expected_dim0))
        np.testing.assert_allclose(np.squeeze(derivative_dim1), np.squeeze(expected_dim1))
        np.testing.assert_allclose(np.squeeze(derivative_dim_last), np.squeeze(expected_dim1))

    def test_polynomial_difference_dim_3d(self):
        model = SINDyBase()
        data = np.array([[[1, 2, 3, 4, 5], [3, 4, 5, 6, 7],
                          [5, 6, 7, 8, 9], [7, 8, 9, 10, 11], [9, 10, 11, 12, 13]],
                         [[1.2, 2.2, 3.2, 4.2, 5.2], [3.2, 4.2, 5.2, 6.2, 7.2],
                          [5.2, 6.2, 7.2, 8.2, 9.2], [7.2, 8.2, 9.2, 10.2, 11.2],
                          [9.2, 10.2, 11.2, 12.2, 13.2]],
                         [[1.4, 2.4, 3.4, 4.4, 5.4], [3.4, 4.4, 5.4, 6.4, 7.4],
                          [5.4, 6.4, 7.4, 8.4, 9.4], [7.4, 8.4, 9.4, 10.4, 11.4],
                          [9.4, 10.4, 11.4, 12.4, 13.4]],
                         [[1.6, 2.6, 3.6, 4.6, 5.6], [3.6, 4.6, 5.6, 6.6, 7.6],
                          [5.6, 6.6, 7.6, 8.6, 9.6], [7.6, 8.6, 9.6, 10.6, 11.6],
                          [9.6, 10.6, 11.6, 12.6, 13.6]],
                         [[1.8, 2.8, 3.8, 4.8, 5.8], [3.8, 4.8, 5.8, 6.8, 7.8],
                          [5.8, 6.8, 7.8, 8.8, 9.8], [7.8, 8.8, 9.8, 10.8, 11.8],
                          [9.8, 10.8, 11.8, 12.8, 13.8]]])
        xgrid = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        with self.assertRaises(ValueError):
            model.polynomial_difference(data, xgrid, dim=3)

        derivative_dim0 = model.polynomial_difference(data, xgrid, dim=0)
        derivative_dim1 = model.polynomial_difference(data, xgrid, dim=1)
        derivative_dim2 = model.polynomial_difference(data, xgrid, dim=2)
        expected_dim0 = np.ones((5, 5))*2
        expected_dim1 = np.ones((5, 5))*20
        expected_dim2 = np.ones((5, 5))*10

        np.testing.assert_allclose(np.squeeze(derivative_dim0), np.squeeze(expected_dim0))
        np.testing.assert_allclose(np.squeeze(derivative_dim1), np.squeeze(expected_dim1))
        np.testing.assert_allclose(np.squeeze(derivative_dim2), np.squeeze(expected_dim2))

    def test_polynomial_difference_degree(self):
        model = SINDyBase()
        data1d = np.random.rand(1, 10)
        data2d = np.random.rand(2, 30)
        xgrid1 = np.arange(10)*0.1
        xgrid2 = np.arange(30)*0.1

        derivative1d = model.polynomial_difference(data1d, xgrid1, degree=2)
        derivative2d = model.polynomial_difference(data2d, xgrid2, dim=1, degree=3)
        assert derivative1d.shape == (6, )
        assert derivative2d.shape == (2, 24)

    def test_poly_exponents(self):
        model = SINDyBase()
        exponents1 = model.get_poly_exponents(0, 10)
        exponents2 = model.get_poly_exponents(1, 2)
        exponents3 = model.get_poly_exponents(5, 1)
        exponents4 = model.get_poly_exponents(2, 3)

        expected1 = [()]
        expected2 = [(0,), (1,), (2,)]
        expected3 = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 1, 0, 0, 0),
                     (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
        expected4 = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1),
                     (2, 1), (0, 2), (1, 2), (0, 3)]

        np.testing.assert_allclose(sorted(exponents1), sorted(expected1))
        np.testing.assert_allclose(sorted(exponents2), sorted(expected2))
        np.testing.assert_allclose(sorted(exponents3), sorted(expected3))
        np.testing.assert_allclose(sorted(exponents4), sorted(expected4))

    def test_polynomial_expansion(self):
        model = SINDyBase()
        mtx1 = np.array([1, 2, 3]).T
        mtx2 = np.array([[1, 2], [3, 4], [5, 6]])
        expanded_mtx1, _ = model.polynomial_expansion(mtx1, 3)
        expanded_mtx2, _ = model.polynomial_expansion(mtx2, 2)
        expected1 = np.array([[1, 1, 1], [1, 2, 3], [1, 4, 9], [1, 8, 27]]).T
        expected2 = np.array([[1, 1, 1], [1, 3, 5], [2, 4, 6], [1, 9, 25], [2, 12, 30], [4, 16, 36]]).T

        np.testing.assert_allclose(expanded_mtx1, expected1)
        np.testing.assert_allclose(expanded_mtx2, expected2)

