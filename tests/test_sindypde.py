from unittest import TestCase
from pySINDy.sindypde import SINDyPDE
from pySINDy.utils import generator
import numpy as np


class TestSINDy(TestCase):
    def test_sampling(self):
        model = SINDyPDE()
        test_shape = (10, 10, 10)
        test_space_rate = 0.06
        test_time_rate = 0.2
        test_width_x = 1
        test_width_t = 1
        test_sample_idxs = model.sampling_idxs(test_shape, test_space_rate,
                                               test_time_rate, test_width_x,
                                               test_width_t)

        test_n_sample = 12
        test_time_idxs = [1, 9]*6
        self.assertEqual(len(test_shape), len(test_sample_idxs))
        for i in np.arange(len(test_sample_idxs)):
            self.assertEqual(len(test_sample_idxs[i]), test_n_sample)
        self.assertEqual(test_sample_idxs[-1], test_time_idxs)


