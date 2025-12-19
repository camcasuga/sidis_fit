import dipole
import minimizing
import numpy as np
import os
from hankel import HankelTransform
import pandas as pd
from scipy import integrate

lambda_qcd = 0.241 #GeV
conv_fmGeV = 5.068 # fm to GeV^-1
d = 0.54 * conv_fmGeV # GeV-1 
A = 197 # 197 for Au beam, 208 for Pb beam
conv_mbGeV2 = 2.56819

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

def RA(A):
    return (1.12 * A**(1/3) - 0.86*A**(-1/3))*conv_fmGeV # return in GeV^-1

def rhoA(z, bt, A): # Woods-Saxon distribution # GeV-1 
    ra = RA(A)
    n = 3.0/(4.0 * np.pi * RA(A)**3) * 1.0/(1.0 + (np.pi * d/RA(A))**2)
    B = (np.sqrt(bt**2 + z**2) - ra)/d
    rhoA = n / (1 + np.exp(B))
    return rhoA

# Woods-Saxon distribution T_A: rhoA integrated over z
def get_TA(bt, A): 
    TA = integrate.quad(rhoA, -999.0, 999.0, args = (bt, A))
    return TA[0]

def dipA(r, bt, A, Qs02, gamma, e_c, sigma0_2): # dipole nucleus
    sigma0_2 = sigma0_2 * 2.56819 # converted to 1/GeV2
    #n = woods_saxon_norm(A)
    #test = integrate.quad(lambda b: 2.0*np.pi*b*get_TA(b, A), 0, np.inf)
    #assert np.isclose(test[0], 1.0, atol=1e-3), "Thickness function not normalized to 1.0"
    TA = get_TA(bt, A) 
    B = A * TA * (sigma0_2)*((r**2)*(Qs02))**gamma
    C = 1/(r*lambda_qcd) + e_c * np.exp(1)
    N = 1 - np.exp(-B/4 * np.log(C))
    #print(TA)
    return N

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
    r_grid = np.logspace(-2, 2, 200) # dipole size logspace 
    b_grid = range(0, 33, 2)
    log_k_min = -2
    log_k_max = 2
    k_grid_size = 500
    log_k_grid = np.linspace(log_k_min, log_k_max, k_grid_size)
    y_array_repeat = np.repeat(y_grid, k_grid_size)

    #load initial condition parameters
    ic_qs, ic_ec, ic_c2, ic_sigma02 = np.loadtxt("../data/dipole/posteriorsamples_100_LOmvefit.dat", unpack = True)
    all_data = pd.DataFrame(columns = ['b', 'y', 'log10_k', 'log10_S(k)'])
    
    for b in b_grid: 
        #fname in enumerate(bk_folder_list): # for each posterior sample
        fname = f"ic_197_{b}.dat"
        fit_params = np.ones((y_grid_size,4))
        #print(b, fname)
        #file_params = open(f"fit_params/{i}.dat", "w")
        bk_file = os.path.join(dipoles_dir, fname)
        interpolator = dipole.BKDipole(bk_file)
        Nyr = lambda y, r: interpolator.ln_Nr(y,r)
        initial_params = [ic_qs[0], 1.0, ic_ec[0], 1.0]
        initial_params[0] = 197 * get_TA(b, A) * ic_sigma02[0] * conv_mbGeV2 * ic_qs[0]
        log_k_all = []
        log_Sk2_all = []
        j = 0
        #print("b value", b)
        for y in y_grid:
            minimizer = minimizing.MyMinimizer(Nyr, y, r_grid, initial_guess, lbounds=lbounds, ubounds=ubounds)
            out = minimizer.fit_ls()
            fit_params[j,:] = out.x

            if y == 0:
                fit_params[j, :] = initial_params
            
            log_Sk2 = log_S2(minimizer, 10**log_k_grid, fit_params[j,:])
            if np.isnan(log_Sk2).any():
                print(f"Nan values in log S(k) for b = {b} and y = {y}")
                fit_params[j, :] = initial_params
                log_Sk2 = log_S2(minimizer, 10**log_k_grid, fit_params[j,:])
            
            #print(f'y = {y},  params = {fit_params[j,:]}')
            log_k_all.extend(log_k_grid)
            log_Sk2_all.extend(log_Sk2)
            j += 1

        b_repeat = b*np.ones(len(y_array_repeat))
        data = pd.DataFrame({'b': b_repeat, 'y': y_array_repeat, 'log10_k':log_k_all, 'log10_S(k)':log_Sk2_all})
        all_data = pd.concat([all_data, data])

            # while file_params:
            #     file_params.write(f"# Parameter file of fit of dipole from {bk_file}" + "\n")
            #     file_params.write('# Dipoles / cross section generated with initial condition: MV model, Qs0^2 = {: .6e}, e_c = {: .6e}, C^2 = {: .6e}, sigma0/2 = {: .6e}'.format(ic_qs[0], ic_ec[0], ic_c2[0], ic_sigma02[0]))
            #     file_params.write(f"# y ranges from 0 to {y_max} in {y_grid_size} steps" + "\n")
            #     file_params.write("# Columns: y, Qs2, gamma, ec, p" + "\n")
            #     for j in range(y_grid_size):
            #         file_params.write("{: .6e} {: .6e} {: .6e} {: .6e} {: .6e}\n".format(y_grid[j], fit_params[j,0], fit_params[j,1], fit_params[j,2], fit_params[j,3]))
    
    #csv_file = 'BKfit_Sk_wb_data.csv'
    csv_file = 'BKfit_Sk_wb_data_nonans.csv'
    comments = [f'## Fitted to BK evolved dipoles with initial condition: MVe model, Q_s0^2 = {ic_qs[0]} GeV^2, \gamma = 1, coefficient of E inside Log is {ic_ec[0]}',                  
                    f'## Other initial condition parameters: proton transverse area (sigma0/2) is {ic_sigma02[0]} mb and running coupling parameter (C^2) is {ic_c2[0]}',
                    f'## Note definition of log S(k): log_S(k) = log10 S(k) = log10 [ \int r dr J0(k r) (1 - N(r)) ]',
                    f'## log10_k grid ranges from {log_k_min} to {log_k_max} with {k_grid_size} points',
                    f'## y grid ranges from 0 to {y_max} with {y_grid_size} points',
                    f'## Impact parameter values, b, ranges from 0 to 32 GeV^-1']

    with open(csv_file, 'w') as f:
        for comment in comments:
            f.write(comment + '\n')

    all_data.to_csv(csv_file, index=False, sep = ' ', columns=['b', 'y', 'log10_k', 'log10_S(k)'], mode='a', float_format='%.6e')
        
    # all_data.to_csv('BKfit_Sk_data.csv', index = False)

            
if __name__ == "__main__":
    main("../data/nucleus/0")