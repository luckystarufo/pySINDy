import numpy as np 
from .sindybase import SINDyBase

class DSINDy(SINDyBase):
    """
    Discrete-Time Sparse Identification of Nonlinear Dynamics:
    reference: http://www.pnas.org/content/pnas/113/15/3932.full.pdf
    
    for discovering systems in the form of x_k+1 = f(x_k)
    """
    def fit(self, data, poly_degree=2, cut_off=1e-3):
        """
        :param data: dynamics data to be processed
        :param poly_degree: degree of polynomials to be included in theta matrix
        :param cut_off: the threshold cutoff value for sparsity
        :return: a SINDy model
        """
        if len(data.shape) == 1:
            data = data[np.newaxis, ]

        len_t = data.shape[-1]

        if len(data.shape) > 2:
            data = data.reshape((-1, len_t))
            print("The array is converted to 2D automatically: in SINDy, "
                  "each dimension except for the time (default the last dimension) "
                  "are treated equally.")

        # x_prime is one step into the future
        x_prime = data[:,1:].T
        data = data[:,:-1] 
        
        # prepare for the library
        lib, self._desp = self.polynomial_expansion(data.T, degree=poly_degree)

        # sparse regression
        self._coef, _ = self.sparsify_dynamics(lib, x_prime, cut_off)

        return self
