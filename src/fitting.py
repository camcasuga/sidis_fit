import dipole
import minimizing
import numpy as np
from nucleus import *

def main(dipoles_dir):
    ''' 
    
    Loops through all BK-evolved dipoles inside dipoles_dir and optimizes the hyperparameters in form: N(r) = 1 - exp()^gamma. Solves for 2DFT: log_S(k) = log10 S(k) = log10 [ \int r dr J0(k r) ]
    
    '''

    bk_folder_list = [f for f in os.listdir(dipoles_dir) if f.endswith('.dat')]

    lbounds = [0.001, 0.1, 1/np.exp(1), 0.001]
    ubounds = [10.0, 100.1, 100.0, 10.0]
    initial_guess =  [1.0, 0.4, 1.0, 1.0]
    param_names = ['Qs2', 'gamma', 'ec', 'p']
    y_grid_size = 100 # 100 # divisions of rapidity grid
    y_max = 8.0
    y_grid = np.linspace(0, y_max, y_grid_size)
    r_grid = np.logspace(-2, 2, 100) # dipole size logspace 
    k_grid_size = 500 #100#500
    log_k_grid = np.linspace(-2, 2, k_grid_size)
    y_array_repeat = np.repeat(y_grid, k_grid_size)

    # load initial condition parameters
    ic_qs, ic_ec, ic_c2, ic_sigma02 = np.loadtxt("../data/dipole/posteriorsamples_100_LOmvefit.dat", unpack = True)
    file_index = range(68, 100)
    for i in file_index: # for each posterior sample

        fname = f"{i}.dat"
        print(f"Reading file: {i+1}/{len(bk_folder_list)}")
        bk_file = os.path.join(dipoles_dir, fname)
        fit_params = np.ones((y_grid_size,4))
        file_params = f"../data/dipole/fit_params/fp_{i}.dat"
        interpolator = dipole.BKDipole(bk_file)
        Nyr = lambda y, r: interpolator.ln_Nr(y,r)
        log_k_all = []
        log_Sk2_all = []
        j = 0
        for y in y_grid:
            print(f"{j+1}/{y_grid_size}")
            minimizer = minimizing.MyMinimizer(Nyr, y, r_grid, initial_guess, lbounds=lbounds, ubounds=ubounds, lsmethod='trf')
            out = minimizer.fit_ls()
            fit_params[j,:] = out.x

            # now for the 2dft of the dipole, for different rapidities and kt
            log_Sk2 = log_S2(minimizer, 10**log_k_grid, fit_params[j,:])
            if np.isnan(log_Sk2).any():
                print(f"Nan values in log S(k) for y = {y}")

            log_k_all.extend(log_k_grid)
            log_Sk2_all.extend(log_Sk2)
            j += 1
        
        file_params_comments = [f"# Parameter file of fit of dipole from {bk_file}",
                                '# Dipoles / cross section generated with initial condition: MV model, Qs0^2 = {: .6e}, e_c = {: .6e}, C^2 = {: .6e}, sigma0/2 = {: .6e}'.format(ic_qs[i], ic_ec[i], ic_c2[i], ic_sigma02[i]),
                                f"# y ranges from 0 to {y_max} in {y_grid_size} steps" ,
                                "# Columns: y, Qs2, gamma, ec, p" ]
        
        with open(file_params, 'w') as f:
            for comment in file_params_comments:
                f.write(comment + '\n')
            for l in range(y_grid_size):
                f.write("{: .6e} {: .6e} {: .6e} {: .6e} {: .6e}\n".format(y_grid[l], fit_params[l,0], fit_params[l,1], fit_params[l,2], fit_params[l,3]))


        data = {'y': y_array_repeat, 'log10_k':log_k_all, 'log10_S(k)':log_Sk2_all}
        df = pd.DataFrame(data)
        csv_file = f"../data/dipole/Sk_fit/2Dft_{i}.csv"
        csv_comments = ['## Generated from BK dipole with initial condition: MVe model, Qs0^2 = {:.6e}, e_c = {:.6e}, C^2 = {:.6e}, sigma0/2 = {:.6e}'.format(ic_qs[i], ic_ec[i], ic_c2[i], ic_sigma02[i]),
                        '## Fitted to file {0}'.format(bk_file),
                        '## Note definition of log S(k): log_S(k) = log10 S(k) = log10 [ \int r dr J0(k r) (1 - N(r)) ]',
                        '## log10 k grid ranges from {0} to {1} with {2} points'.format(-2, 2, k_grid_size),
                        '## y grid ranges from 0 to {0} with {1} points'.format(y_max, y_grid_size),]

        with open(csv_file, 'w') as f:
            for comment in csv_comments:
                f.write(comment + '\n')
        df.to_csv(csv_file, index=False, sep = ' ', columns=['y', 'log10_k', 'log10_S(k)'], mode='a', float_format='%.6e')
        i += 1
    return

if __name__ == "__main__":
    dipoles_dir = "../data/dipole/posterior/LOmvefit"
    main(dipoles_dir)
