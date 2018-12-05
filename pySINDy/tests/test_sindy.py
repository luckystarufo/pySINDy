# pylint: skip-file

from unittest import TestCase
from ..sindy import SINDy
import numpy as np
import warnings


class TestSINDy(TestCase):
    def test_shape1(self):
        warnings.simplefilter("ignore")
        model = SINDy()
        data = np.random.rand(10,)
        dt = 0.1
        model.fit(data, dt)
        assert True
