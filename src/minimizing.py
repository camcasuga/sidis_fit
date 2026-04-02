from scipy.optimize import minimize, basinhopping, leastsq, least_squares, curve_fit
import numpy as np

class MyMinimizer:

    def __init__(self, NrYinterpolator, y, rarray, initial_params, lbounds = None, ubounds = None, lsmethod = 'trf'):
        self.initial_params = initial_params #[0.05, 2.0, 0.4, 0.2]
        self.lbounds = lbounds
        self.ubounds = ubounds
        self.NrYinterpolator = NrYinterpolator # interpolator/fun wiht input (r, y), most likely ln of N(r,y)
        #self.Y_range = (np.min(Y_values), np.max(Y_values))
        #self.r_range = (np.min(r_values), np.max(r_values))
        self.y = y
        self.rarray = rarray
        self.lsmethod = lsmethod  #'lm' #'trf' #'dogbox'
        #self.fit_func = lambda Qs2, gamma, ec, p: self.parametrization(Qs2, gamma, ec, p)
        self.fit_func = lambda Qs2, gamma, ec, p : self.parametrization(Qs2, gamma, ec, p)

    # def Ff(self, r, theta):
    #     ''' Returns function F(r) for parameters (Qs2, gamma, ec, p) '''
    #     lambda_qcd = 0.241
    #     exponent = (((r**2 * theta[0]) ** theta[1])/4 * np.log( (1/(r*lambda_qcd)) + np.exp(1)*theta[2] ) )** theta[3]
    #     return 1 - np.exp(-exponent) ** (1/theta[3])
    
    def Ff(self, r, Qs2, gamma, ec, p):
        ''' Returns function F(r) for parameters (Qs2, gamma, ec, p) '''
        lambda_qcd = 0.241
        exponent = (((r**2 * Qs2) ** gamma)/4 * np.log( (1/(r*lambda_qcd)) + np.exp(1)*ec ) )** p
        
        return 1 - np.exp(-exponent) ** (1/p)
        #return np.log(1 - np.exp(-exponent))

    def parametrization(self, Qs2, gamma, ec, p):
        ''' Returns bounded F(r) values over array of r'''
        r = self.rarray
        theta = [Qs2, gamma, ec, p]
        
        if self.lbounds is not None and self.ubounds is not None:
            for i in range(4):
                if theta[i] < self.lbounds[i] or theta[i] > self.ubounds[i]:
                    return np.inf #99999. * np.ones_like(r)

        return np.array([np.log(self.Ff(ri, Qs2, gamma, ec, p)) for ri in r])
    
    def fit_ls(self):
        ''' Least squares fit to find optimal parameters '''
        init = self.initial_params
        r = self.rarray
        #fit_func = lambda p: self.fit_func(p[0], p[1], p[2], p[3])
        err_func = lambda params, self: self.NrYinterpolator(self.y, r) - self.fit_func(params[0], params[1], params[2], params[3])
        out = least_squares(err_func, init, args = (self,), method = self.lsmethod)

        #f = lambda r, Qs2, gamma, ec, p: self.fit_func(r, Qs2, gamma, ec, p)
        #out = curve_fit(f, r, self.NrYinterpolator(self.y, r), p0 = init, bounds=(self.lbounds, self.ubounds), method = self.lsmethod)
        return out
