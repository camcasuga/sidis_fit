import dipole
import minimizing
import numpy as np

def main(dipoles_dir):

    ''' 
    
    Loops through all BK-evolved dipoles inside dipoles_dir (each with a corresponding) and optimizes the hyperparameters in form: N(r) = 1 - exp()^gamma. Solves for 2DFT: log_S(k) = log10 S(k) = log10 [ \int r dr J0(k r) ]
    
    '''

    bk_folder_list = [f for f in os.listdir(dipoles_dir) if f.endswith('.dat')]

    lbounds = [0.001, 0.1, 1/np.exp(1), 0.001]
    ubounds = [10.0, 100.1, 100.0, 10.0]
    initial_guess =  [1.0, 0.8, 1.0, 1.0]
    param_names = ['Qs2', 'gamma', 'ec', 'p']
    y_grid_size = 100 # divisions of rapidity grid
    y_max = 8.0
    y_grid = np.linspace(0, y_max, y_grid_size)
    r_grid = np.logspace(-3, 2, 200) # dipole size logspace 
    b_grid = range(0, 32, 2)

    # load initial condition parameters
    ic_qs, ic_ec, ic_c2, ic_sigma02 = np.loadtxt("data/posteriorsamples_100_LOmvefit.dat", unpack = True)

    for i, fname in enumerate(bk_folder_list): # for each posterior sample
        print(f"Reading file: {i+1}/{len(bk_folder_list)}")
        bk_file = os.path.join(dipoles_dir, fname)
        
        fit_params = np.ones(y_grid_size,4)
        file_params = open(f"fit_params/{i}.dat", "w")
        interpolator = dipole.BKDipole(bk_file)
        Nyr = lambda y, r: interpolator.ln_Nr(y,r)
        i = 0
        for y in y_grid:
            minimizer = minimizing.MyMinimizer(Nyr, y, r_grid, initial_guess, lbounds=lbounds, ubounds=ubounds, lsmethod='trf')
            out = minimizer.fit_ls()
            fit_params[i,:] = out.x
            i += 1

        while file_params:
            file_params.write(f"# Parameter file of fit of dipole from {bk_file}" + "\n")
            file_params.write('# Dipoles / cross section generated with initial condition: MV model, Qs0^2 = {: .6e}, e_c = {: .6e}, C^2 = {: .6e}, sigma0/2 = {: .6e}'.format(ic_qs[0], ic_ec[0], ic_c2[i], ic_sigma02[i]))
            file_params.write(f"# y ranges from 0 to {y_max} in {y_grid_size} steps" + "\n")
            file_params.write("# Columns: y, Qs2, gamma, ec, p" + "\n")
            for j in range(y_grid_size):
                file_params.write("{: .6e} {: .6e} {: .6e} {: .6e} {: .6e}\n".format(y_grid[j], fit_params[j,0], fit_params[j,1], fit_params[j,2], fit_params[j,3]))


        # now for the 2dft of the dipole, for different rapidities and kt

