# plot the cover line by scanning through V_f

# INPUT: Vf start, stop, num, N
# (Vf < 1)
# OUTPUT:
# image named f"scan_{Vf_start}_{Vf_stop}_{step}_{N}.png"

from optimize import objective_function
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


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
        p_s = list(map(obj, eta_values))
        p_max = max(p_s)
        # eta is the corresponding eta
        eta = eta_values[p_s.index(p_max)]
        # Append the results
        etas.append(eta)
        P_maxs.append(p_max)
    return np.array(etas), np.array(P_maxs)



if __name__ == "__main__":
    # Define the range of eta values
    import numpy as np
    start  = 0.1
    stop = 0.99
    step = 10
    N = 2
    etas, p_maxs = plot_cover(start, stop, step, N)
    plt.xlabel(f"$\eta$")
    plt.ylabel(f'$P_max$')
    plt.title("Covering Line")
    # save the figure
    plt.savefig(f"figures/cover_start_{start}_stop_{stop}_step_{step}_N_{N}.png", dpi=300)