import dipole
import minimizing
import numpy as np
import os
from hankel import HankelTransform
import pandas as pd

def S(interpolator, y,k):

    ht = HankelTransform(nu = 0, N = 1000, h = 0.001)

    def func_to_ht(r):
        f = interpolator(y, r) #minimizer.Ff(r, fit_params)
        f = np.nan_to_num(f, nan=1.0)
        return 1 - f

    sp_values = [ht.transform(func_to_ht, i, ret_err=False) for i in k]
    return np.array(sp_values)

def log_S2(minimizer, k, fit_params):
    ht = HankelTransform(nu = 0, N = 1000, h = 0.001)

    def func_to_ht(r):
        f = minimizer.Ff(r, fit_params)
        f = np.nan_to_num(f, nan=1.0)
        return 1 - f

    sp_values = [np.log10(ht.transform(func_to_ht, i, ret_err=False)) for i in k]
    return np.array(sp_values)

def S2(minimizer, k, fit_params):
    ht = HankelTransform(nu = 0, N = 1000, h = 0.001)

    def func_to_ht(r):
        f = minimizer.Ff(r, fit_params)
        f = np.nan_to_num(f, nan=1.0)
        return 1 - f

    sp_values = [ht.transform(func_to_ht, i, ret_err=False) for i in k]
    return np.array(sp_values)

def main(dipoles_dir):

    ''' 
    
    Loops through all BK-evolved dipoles inside dipoles_dir (each with a corresponding) and optimizes the hyperparameters in form: N(r) = 1 - exp()^gamma. Solves for 2DFT: log_S(k) = log10 S(k) = log10 [ \int r dr J0(k r) ]
    
    '''

    bk_folder_list = [f for f in os.listdir(dipoles_dir) if f.endswith('.dat')]
    lbounds = [0.001, 0.1, 1/np.exp(1), 0.001]
    ubounds = [10.0, 100.0, 100.0, 10.0]
    initial_guess =  [1.0, 1.0, 1.0, 1.0]
    param_names = ['Qs2', 'gamma', 'ec', 'p']
    y_grid_size = 100 # divisions of rapidity grid
    y_max = 8.0
    y_grid = np.linspace(0, y_max, y_grid_size)
    r_grid = np.logspace(-3, 2, 200) # dipole size logspace 
    b_grid = range(0, 32, 2)
    log_k_min = -2
    log_k_max = 2
    k_grid_size = 500
    log_k_grid = np.linspace(log_k_min, log_k_max, k_grid_size)
    y_array_repeat = np.repeat(y_grid, k_grid_size)

    # load initial condition parameters
    # ic_qs, ic_ec, ic_c2, ic_sigma02 = np.loadtxt("../data/posteriorsamples_100_LOmvefit.dat", unpack = True)
    all_data = pd.DataFrame(columns = ['b', 'y', 'log10_k', 'log10_S(k)'])
    for i, fname in enumerate(bk_folder_list): # for each posterior sample

        print(i, fname)

        fit_params = np.ones((y_grid_size,4))
        #file_params = open(f"fit_params/{i}.dat", "w")
        bk_file = os.path.join(dipoles_dir, fname)
        interpolator = dipole.BKDipole(bk_file)
        Nyr = lambda y, r: interpolator.ln_Nr(y,r)
        log_k_all = []
        log_Sk2_all = []
        j = 0
        for y in y_grid:
            minimizer = minimizing.MyMinimizer(Nyr, y, r_grid, initial_guess, lbounds=lbounds, ubounds=ubounds)
            out = minimizer.fit_ls()
            fit_params[i,:] = out.x
            log_Sk2 = log_S2(minimizer, 10**log_k_grid, fit_params[i,:])
            log_k_all.extend(log_k_grid)
            log_Sk2_all.extend(log_Sk2)
            j += 1

        b = b_grid[i]*np.ones(k_grid_size)
        data = {'b': b, 'y': y_array_repeat, 'log10_k':log_k_all, 'log10_S(k)':log_Sk2_all}
        all_data.append(data)

            # while file_params:
            #     file_params.write(f"# Parameter file of fit of dipole from {bk_file}" + "\n")
            #     file_params.write('# Dipoles / cross section generated with initial condition: MV model, Qs0^2 = {: .6e}, e_c = {: .6e}, C^2 = {: .6e}, sigma0/2 = {: .6e}'.format(ic_qs[0], ic_ec[0], ic_c2[0], ic_sigma02[0]))
            #     file_params.write(f"# y ranges from 0 to {y_max} in {y_grid_size} steps" + "\n")
            #     file_params.write("# Columns: y, Qs2, gamma, ec, p" + "\n")
            #     for j in range(y_grid_size):
            #         file_params.write("{: .6e} {: .6e} {: .6e} {: .6e} {: .6e}\n".format(y_grid[j], fit_params[j,0], fit_params[j,1], fit_params[j,2], fit_params[j,3]))
    
    all_data.to_csv('BKfit_Sk_data.csv')

            
if __name__ == "__main__":
    main("../data/nucleus/0")