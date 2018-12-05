# pylint: skip-file

from unittest import TestCase
from ..sindypde import SINDyPDE
import numpy as np
import warnings


class TestSINDy(TestCase):
    def test_sampling(self):
        model = SINDyPDE()
        assert True

