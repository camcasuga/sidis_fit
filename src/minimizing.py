from scipy.optimize import minimize, basinhopping, leastsq
import numpy as np

class MyMinimizer:

    def __init__(self, NrYinterpolator, y, rarray, bounds = None):
        self.initial_params = [0.1, 1.0, 1.0, 1.0]
        self.bounds = bounds
        self.NrYinterpolator = NrYinterpolator # interpolator/fun wiht input (r, y)
        #self.Y_range = (np.min(Y_values), np.max(Y_values))
        #self.r_range = (np.min(r_values), np.max(r_values))
        self.y = y
        self.rarray = rarray
        self.fit_func = lambda Qs2, gamma, ec, p, r: self.parametrization(Qs2, gamma, ec, p, r)

    def parametrization(self, Qs2, gamma, ec, p, r):
        ''' Returns function F(r) for parameters (Qs2, ec, p, gamma) '''
        lambda_qcd = 0.241
        return 1 - np.exp(-(((r**2 * Qs2) ** gamma)/4 * np.log( (1/(r*lambda_qcd)) + 1/ (np.exp(1)*ec) ) )** p) ** (1/p)
    
    def fit_ls(self):
        r = self.rarray
        init = self.initial_params
        fit_func = lambda p, r: self.fit_func(p[0], p[1], p[2], p[3],r)
        data = self.NrYinterpolator(self.y)
        err_func = lambda p: data - fit_func(p,r)
        out = leastsq(err_func, init)
        # out = basinhopping(err_func, init, args =(r,y))
        return out[0]
