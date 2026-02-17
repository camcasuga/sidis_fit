import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import sys
from scipy.interpolate import CloughTocher2DInterpolator

colors = ['blue', 'red', 'green', 'orange', 'gray']

def Sk_interp_b(b, sk_file):

    """ Returns a 2D interpolator for log S(k) as a function of log k and y FOR A GIVEN b in range(0,32,2)"""

    df = pd.read_csv(sk_file, comment='#', sep=' ')
    df_b = df[(df['b'] < b+1.0) & (df['b'] > b-1.0)]
    y = df_b['y'].values
    log_k = df_b['log10_k'].values
    log_Sk2 = df_b['log10_S(k)'].values
    #print(len(df_b['b'].values))
    interpolator = CloughTocher2DInterpolator(list(zip(y, log_k)), log_Sk2)
    return interpolator


def main(sk_files_dir, y_val, b_val = None):

    sk_files_dir_list = glob.glob(sk_files_dir + "/*.csv")
    sk_files_dir_list.sort()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    interpolators = []
    for sk_file in sk_files_dir_list:
        print(f"Reading file: {sk_file}")
        interpolators.append(Sk_interp_b(b_val, sk_file))
    
    k_values = 100
    log_k_grid = np.linspace(-2, 2, k_values)
    mean = np.zeros(k_values)
    up_sd = np.zeros(k_values)
    down_sd = np.zeros(k_values)

    i = 0
    for log_k in log_k_grid:
        log_SK = np.array([interpolator(y_val, log_k) for interpolator in interpolators])
        print(log_SK)
        mean[i] = np.mean(log_SK)
        lower_region = log_SK < mean[i]
        upper_region = log_SK > mean[i]
        up_sd[i] = np.std(log_SK[upper_region])
        down_sd[i] = np.std(log_SK[lower_region])
        i += 1

    ax.plot(log_k_grid, mean, label=f'y = {y_val}, b = {b_val}', color = 'blue')
    ax.fill_between(log_k_grid, mean - 2*down_sd, mean + 2*up_sd, color='blue', alpha=0.8)

    ax.legend()
    ax.set_xlabel('log10(k)')
    ax.set_ylabel('log10(S(k))')
    fig.savefig("../plots/logsk_fory_nucleus_bdep.pdf", dpi=300)

if __name__ == "__main__":

    " Run code as: python3 plot_Skband_b.py y_val b_val"

    if len(sys.argv) < 2:
        print("python3 plot_Skband_b.py y_val b_val")
    
    y = sys.argv[1]
    b = int(sys.argv[2])
    
    sk_files_dir = "../data/nucleus/Sk_fit"
    main(sk_files_dir, y, b)

