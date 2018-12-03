from unittest import TestCase
from pySINDy.sindy import SINDy
from pySINDy.utils import generator
import numpy as np


class TestSINDy(TestCase):
    def test_shape1(self):
        model = SINDy()
        data = np.random.rand(10,)
        dt = 0.1
        model.fit(data, dt)
        assert True
