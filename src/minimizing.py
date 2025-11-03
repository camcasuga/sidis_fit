from scipy.optimize import minimize, basinhopping, leastsq, least_squares
import numpy as np

class MyMinimizer:

    def __init__(self, NrYinterpolator, y, rarray, initial_params, lbounds = None, ubounds = None):
        self.initial_params = initial_params #[0.05, 2.0, 0.4, 0.2]
        self.lbounds = lbounds
        self.ubounds = ubounds
        self.NrYinterpolator = NrYinterpolator # interpolator/fun wiht input (r, y)
        #self.Y_range = (np.min(Y_values), np.max(Y_values))
        #self.r_range = (np.min(r_values), np.max(r_values))
        self.y = y
        self.rarray = rarray
        self.fit_func = lambda Qs2, gamma, ec, p: self.parametrization(Qs2, gamma, ec, p)

    def Ff(self, r, theta):
        lambda_qcd = 0.241
        exponent = (((r**2 * theta[0]) ** theta[1])/4 * np.log( (1/(r*lambda_qcd)) + np.exp(1)*theta[2] ) )** theta[3]
        return 1 - np.exp(-exponent) ** (1/theta[3])

    def parametrization(self, Qs2, gamma, ec, p):
        ''' Returns function F(r) for parameters (Qs2, ec, p, gamma) '''
        r = self.rarray
        theta = [Qs2, gamma, ec, p]
        
        if self.lbounds is not None and self.ubounds is not None:
            for i in range(4):
                if theta[i] < self.lbounds[i] or theta[i] > self.ubounds[i]:
                    return np.inf * np.ones_like(r)

        return np.array([np.log(self.Ff(ri, theta)) for ri in r])
    
    def fit_ls(self):
        init = self.initial_params
        r = self.rarray
        #fit_func = lambda p: self.fit_func(p[0], p[1], p[2], p[3])
        err_func = lambda params, self: self.NrYinterpolator(self.y, r) - self.fit_func(params[0], params[1], params[2], params[3])
        #print("interpolator", np.shape(self.NrYinterpolator(self.y, r)))
        #print("err_func", np.shape(err_func([1.0, 0.0, 1.0, 1.0],self)))
        #out = leastsq(err_func, init, args = (self))[0] # out = basinhopping(err_func, init, args =(r,y))
        out = least_squares(err_func, init, args = (self,))
        return out
