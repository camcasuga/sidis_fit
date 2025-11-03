# Python code for reading and interpolating the dipole amplitude from BK solver output files.
# C. Casuga, H. Hänninen and H. Mäntysaari, 2025

import numpy as np
import scipy.interpolate as interpolate
import os

class BKDipole:

    def __init__(self, bk_file):
        self.bkfile = bk_file
        self.interpolator, self.Y_range, self.r_range = self.ReadBKDipole()

    def ReadBKDipole(self): # 
        '''
        Reads the dipole amplitude from the given datafile produced running BK solver from 
        
        Returns an interpolator for the dipole: N(Y, r), where r is in GeV^-1, 
        and Y is the evolution rapidity
        '''

        with open(self.bkfile) as f:
            content = f.read().split("###")
        
        content = content[1:]   # gets rid of the stuff at the beginning
        content = [i.split() for i in content] # cleans up lines
        NrY_data = []
        pars = []
        for i in content:

            x = list(map(float, i))
            if len(x) == 1:
                pars.append(x)
            else:
                NrY_data.append(x)

            
        rmYs = np.array(NrY_data).T[1:]     # removes Y values
        N_values = rmYs.T
        Y_values = np.array(NrY_data).T[0]

        pars = np.ndarray.flatten(np.array(pars))
        minr = pars[0]
        mult = pars[1]
        n = int(pars[2])
        r_values = np.array([minr*mult**i for i in range(n)])
        
        
        rgrid=[]
        ygrid=[]
        for y in Y_values:
            for r in r_values:
                rgrid.append(r)
                ygrid.append(y)
        
        self.interpolator = interpolate.CloughTocher2DInterpolator((ygrid, rgrid), N_values.flatten(), rescale=True)
        self.Y_range = (np.min(Y_values), np.max(Y_values))
        self.r_range = (np.min(r_values), np.max(r_values))
        return self.interpolator, self.Y_range, self.r_range

    def N(self, Y, r):
        '''
        Returns the interpolated value of N(Y, r).

        Returns NaN before the initial condition (Y < 0) or if the requested
        evolution rapidity is outside the range  

        Args:
            Y (float): Evolution rapidity
            r (float): Dipole size in GeV^-1
        '''
        r_min, r_max = self.r_range
        Y_min, Y_max = self.Y_range
        
        if r < 0 or Y < 0:
            return np.nan
        
        if Y > Y_max:
            return np.nan
        
        if r > r_max and 0 <= Y <= Y_max: # large r, Y within limits
            return 1.0 
        
        if 0 <= r < r_min and 0 <= Y <= Y_max: # small r, Y within limits
            return 0.0

        N = self.interpolator(Y, r)

        return N

    def Nr(self, Y, r_values): #r_min = 1e-3, r_max = 1e2, r_points = 50):
        '''
        Returns the dipole amplitude N evaluated at 
        multiple points r [GeV^-1] logarithimically spaced between r_min and r_max.
        '''
        #bk_interpolator = ReadBKDipole(bk_file_dir)
        #r_values = np.logspace(np.log10(r_min), np.log10(r_max), r_points)
        N_values = [self.N(Y, r) for r in r_values]
        return np.array(N_values)
    
    def ln_Nr(self, Y, r_values): #r_min = 1e-3, r_max = 1e2, r_points = 50):

        N_values = [np.log(self.N(Y, r)) for r in r_values]
        return np.array(N_values)