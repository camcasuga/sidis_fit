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
    initial_guess =  [1.0, 0.8, 1.0, 1.0]
    param_names = ['Qs2', 'gamma', 'ec', 'p']
    y_grid_size = 100 # 100 # divisions of rapidity grid
    y_max = 8.0
    y_grid = np.linspace(0, y_max, y_grid_size)
    r_grid = np.logspace(-2, 2, 100) # dipole size logspace 
    log_k_min = -2
    log_k_max = 2
    k_grid_size = 500
    log_k_grid = np.linspace(log_k_min, log_k_max, k_grid_size)
    log_k_all = []
    log_Sk2_all = 0
    y_array_repeat = np.repeat(y_grid, k_grid_size)

    # load initial condition parameters
    ic_qs, ic_ec, ic_c2, ic_sigma02 = np.loadtxt("../data/dipole/posteriorsamples_100_LOmvefit.dat", unpack = True)
    start = 0
    n_samples = 100
    for i in range(start, n_samples): # for each posterior sample

        # if i < 91:
        #     continue

        fname = f"{i}.dat"
        print(f"Reading file: {fname}")
        bk_file = os.path.join(dipoles_dir, fname)
        fit_params = np.ones((y_grid_size,4))
        file_params = f"../data/dipole/fit_params/{i}.dat"
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

            log_Sk2 = log_S2(minimizer, 10**log_k_grid, fit_params[j,:] )
            log_k_all.extend(log_k_grid)
            log_Sk2_all.extend(log_Sk2)
            j += 1

        with open(file_params, 'w') as file:
            file.write(f"# Parameter file of fit of dipole from {bk_file}" + "\n")
            file.write('# Dipoles / cross section generated with initial condition: MV model, Qs0^2 = {: .6e}, e_c = {: .6e}, C^2 = {: .6e}, sigma0/2 = {: .6e}'.format(ic_qs[i], ic_ec[i], ic_c2[i], ic_sigma02[i]))
            file.write(f"# y ranges from 0 to {y_max} in {y_grid_size} steps" + "\n")
            file.write("# Columns: y, Qs2, gamma, ec, p" + "\n")
            for j in range(y_grid_size):
                file.write("{: .6e} {: .6e} {: .6e} {: .6e} {: .6e}\n".format(y_grid[j], fit_params[j,0], fit_params[j,1], fit_params[j,2], fit_params[j,3]))

        
        data = {'y': y_array_repeat, 'log10_k':log_k_all, 'log10_S(k)':log_Sk2_all}
        df = pd.DataFrame(data)
        csv_file = "Sk_{i}.csv"

        comments = ['## Generated from BK dipole with initial condition: MVe model, Q_s0^2 = 0.0544245 GeV^2, \gamma = 1, coefficient of E inside Log is 58.5321',
                   f'## Fitted to file {bk_file}',
                   '## Note definition of log S(k): log_S(k) = log10 S(k) = log10 [ \int r dr J0(k r) (1 - N(r)) ]',
                   f'## log10 k grid ranges from {log_k_min} to {log_k_max} with {k_grid_size} points',
                   f'## y grid ranges from 0 to {y_max} with {y_grid_size} points',]
        
        with open(csv_file, 'w') as f:
            for comment in comments:
                f.write(comment + '\n')
        
        df.to_csv(csv_file, index=False, sep = ' ', columns=['y', 'log10_k', 'log10_S(k)'], mode = 'a', float_format = '%.6e')       

        return 
    
if __name__ == '__main__':
    main("../data/dipole/posterior/LOmvefit")

