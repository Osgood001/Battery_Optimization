import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from optimize import objective_function


def plot_cover(vf_start, vf_stop, step, N):
    """
        Plot the cover line for a given range of Vf values, and return the P_max vs eta data

        Params:
            vf_start: float
            vf_stop: float
            step: int
            N: int
        
        Returns:
            (etas, P_maxs): (np.array, np.array)
    """ 
    eta_values = np.linspace(0.1, 1, 40)

    etas = []
    P_maxs = []

    for vf in tqdm(np.linspace(vf_start, vf_stop, step)):
        # Objective function
        obj = lambda eta: objective_function(eta, N, vf=vf)
        # map the objective function to the eta values and return the one with the max P_max
        ps = []
        real_etas = []
        for eta in eta_values:
            p, real_eta = obj(eta)
            ps.append(p)
            real_etas.append(real_eta)
        p_max = max(ps)
        # eta is the corresponding eta
        eta = real_etas[ps.index(p_max)]
        # Append the results
        etas.append(eta)
        P_maxs.append(p_max)

    return np.array(etas), np.array(P_maxs)



if __name__ == "__main__":
    # Define the range of eta values
    import numpy as np
    # add parser for command line arguments
    import argparse

    start  = 0.1
    stop = 0.99
    step = 10
    N = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type = float, default = start)
    parser.add_argument('--stop', type = float, default = stop)
    parser.add_argument('--step', type = int, default = step)
    parser.add_argument('--N', type = int, default = 3)
    args = parser.parse_args()

    etas, p_maxs = plot_cover(args.start, args.stop, args.step, args.N)
    # save the data to a csv file using pandas
    import pandas as pd
    df  = pd.DataFrame({"eta": etas, "Pmax": p_maxs})
    df.to_csv(f'data/cover_start_{start}_stop_{stop}_step_{step}_N_{N}.csv', index=False)
    plt.plot(etas, p_maxs, '.-', label="Covering line")
    plt.xlabel(f"$\eta$")
    plt.ylabel('$P_{max}$')
    plt.title("Covering Line")
    # save the figure
    plt.savefig(f"figures/cover_start_{start}_stop_{stop}_step_{step}_N_{N}.png", dpi=300)