import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import sys
from scipy.interpolate import CloughTocher2DInterpolator

colors = ['blue', 'red', 'green', 'orange', 'gray']

def Sk_interp(sk_file):
    """ Should return a 2D interpolator for log S(k) as a function of log k and y """
    df = pd.read_csv(sk_file, comment='#', sep=' ')
    y = df['y'].values
    log_k = df['log10_k'].values
    log_Sk2 = df['log10_S(k)'].values

    interpolator = CloughTocher2DInterpolator(list(zip(y, log_k)), log_Sk2)
    return interpolator


def main(sk_files_dir, y_vals):
    #sk_files_dir_list = [f for f in os.listdir(sk_files_dir)]
    sk_files_dir_list = glob.glob(sk_files_dir + "/*.csv")
    sk_files_dir_list.sort()
    
    interpolators = []
    for sk_file in sk_files_dir_list:
        print(f"Reading file: {sk_file}")
        interpolators.append(Sk_interp(sk_file))
    
    k_values = 100
    log_k_grid = np.linspace(-2, 2, k_values)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    j = 0
    for y_val in y_vals:

        mean = np.zeros(k_values)
        up_sd = np.zeros(k_values)
        down_sd = np.zeros(k_values)

        i = 0
        for log_k in log_k_grid:
            log_SK = np.array([interpolator(y_val, log_k) for interpolator in interpolators])
            mean[i] = np.mean(log_SK)
            lower_region = log_SK < mean[i]
            upper_region = log_SK > mean[i]
            up_sd[i] = np.std(log_SK[upper_region])
            down_sd[i] = np.std(log_SK[lower_region])
            i += 1

        ax.plot(log_k_grid, mean, label=f'y = {y_val}', color=colors[j])
        ax.fill_between(log_k_grid, mean - 2*down_sd, mean + 2*up_sd, color=colors[j], alpha=0.4)

        j += 1

    ax.legend()
    ax.set_xlabel('log10(k)')
    ax.set_ylabel('log10(S(k))')
    fig.savefig("plots/logsk_fory_proton.pdf", dpi=300)

if __name__ == "__main__":

    " Run code as: python3 src/plot_Skband.py y1 y2 y3, i.e. pass as any rapidities as you want"
    
    y = np.zeros(len(sys.argv)-1)
    for n in range(1, len(sys.argv)):
        y[n-1] = float(sys.argv[n])
    
    sk_files_dir = "data/dipole/Sk_fit"
    main(sk_files_dir, y_vals = y)

