from unittest import TestCase
from ..sindy import SINDy
from ..utils import generator
import numpy as np


class TestSINDy(TestCase):
    def test_shape1(self):
        model = SINDy()
        data = np.random.rand(10,)
        dt = 0.1
        model.fit(data, dt)
        assert True
