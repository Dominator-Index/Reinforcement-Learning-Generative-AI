import sys
sys.path.append('../')

import numpy as np
import ot
from ot import emd, bregman
import torch

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import linear_sum_assignment

def plot_ot_plan(samples0, samples1, G0):
    samples0 = samples0.detach().cpu()
    samples1 = samples1.detach().cpu()
    G0 = G0.detach().cpu()

    # Find the non-zero indices in G0 (where lines should be drawn)
    nonzero_G0 = G0.nonzero()
    nonzero_i, nonzero_j = nonzero_G0[:, 0], nonzero_G0[:, 1]

    # Extract the start and end points for the line segments corresponding to non-zero entries
    start_points = samples0[nonzero_i]
    end_points = samples1[nonzero_j]

    # Construct the segments array (shape should be [num_lines, 2, 2])
    segments = np.stack((start_points, end_points), axis=1)

    # Extract linewidths based on the values from G0 where the transport is non-zero
    linewidths = G0[nonzero_i, nonzero_j] / G0[nonzero_i, :].sum(axis=1)    
    lc = LineCollection(segments, linewidths=linewidths, colors='olive', zorder=1)    

    # Set rescaling factor for scatterplot depending on sample size
    rescale_factor = 1000.0 / len(samples0)

    # Add the line collection to the current Axes
    fig, ax = plt.subplots()
    ax.scatter(samples0[:, 0], samples0[:, 1], label='Samples0', zorder=2, s=rescale_factor, c='black')
    ax.scatter(samples1[:, 0], samples1[:, 1], label='Samples1', zorder=2, s=rescale_factor, c='blue')

    # color dark green
    # ax.add_collection(lc_1_0)
    # ax.add_collection(lc_0_1)
    ax.add_collection(lc)

    # Set plot limits if necessary
    ax.autoscale()

    # label the axes
    ax.set_xlabel('$Y$')
    ax.set_ylabel('$U$')
    plt.legend()
    plt.show()

# def COT_empirical_resample(y0, u0, y1, u1, eps=1e-5, method='emd', reg=1e-1):
#     if y0.dim() == 1:
#         y0 = y0.unsqueeze(1)
#         y1 = y1.unsqueeze(1)
#     if u0.dim() == 1:
#         u0 = u0.unsqueeze(1)
#         u1 = u1.unsqueeze(1)
#     C_y = ot.dist(y0, y1)
#     C_u = ot.dist(u0, u1)
#     C = eps*C_u + C_y
#     ind = ot_matching(C, method=method, reg=reg)
#     y1 = y1[ind]
#     u1 = u1[ind]
#     return y1, u1
    
def ot_matching(cost_matrix, method='emd', reg=1e-1, kmax=1e6, save_matrix_path=None, return_ot_cost=False):
    """
    Computes the optimal transport matching between two sets of samples
    """
    if method == 'hungarian':
        return hungarian_matching(cost_matrix)
    elif method == 'emd':
        otmatrix = emd_map(cost_matrix)
    elif method == 'sinkhorn':
        otmatrix = sinkhorn_map(cost_matrix, reg=reg, kmax=kmax)
    else:
        raise ValueError('Unknown OT method')
    if save_matrix_path is not None:
        torch.save(otmatrix, save_matrix_path)
    otmatrix[otmatrix.sum(1) == 0] = 1/otmatrix.shape[1] # fix rows that sum to zero by substituting them with a uniform distribution
    otmatrix = otmatrix / otmatrix.sum(1, keepdim=True)
    indices = torch.multinomial(otmatrix, 1, replacement=True).squeeze()
    if return_ot_cost:
        ot_cost = cost_matrix[:, indices]
        return indices, ot_cost
    return indices

def hungarian_matching(cost_matrix):
    """
    Optimal assignment using the Hungarian algorithm.
    """
    indices = linear_sum_assignment(cost_matrix)
    return np.array(indices[1])

def emd_map(cost_matrix, kmax=1e6):
    """
    OT map using the Earth Mover's Distance.
    """
    n0 = cost_matrix.shape[0]
    n1 = cost_matrix.shape[1]
    device = cost_matrix.device
    dtype = cost_matrix.dtype
    a = torch.ones(n0, device=device).to(dtype) / n0
    b = torch.ones(n1, device=device).to(dtype) / n1
    otmatrix = emd(a=a, b=b, M=cost_matrix, numItermax=kmax)
    return otmatrix

def sinkhorn_map(cost_matrix, reg = 1e-1, kmax=1000):
    """
    Probabilistic OT map using the Sinkhorn algorithm.
    """
    n0 = cost_matrix.shape[0]
    n1 = cost_matrix.shape[1]
    device = cost_matrix.device
    a = torch.ones(n0, device=device) / n0
    b = torch.ones(n1, device=device) / n1
    otmatrix = bregman.sinkhorn_log(a=a, b=b, M=cost_matrix,
                                    reg=reg, numItermax=kmax)
    return otmatrix

def compute_wd_diff(model, initial_u, initial_y, n_eval=1000, print_results=False):
    def compute_dynamic_wd(trajectories):
        return ((n_eval*(trajectories[1:] - trajectories[:-1]))**2).mean()
    def compute_static_wd(trajectories):
        gen = trajectories[-1]
        # C_y = ot.dist(gen.reshape(-1, 1), initial_y.reshape(-1, 1), metric='euclidean')
        # a, b = torch.ones(len(gen))/len(gen), torch.ones(len(initial_y))/len(initial_y)
        # return ot.emd2(a, b, C_y)**0.5
        return ((gen - initial_u)**2).mean()
    device = initial_u.device
    trajectories = model.sample(u_initial=initial_u, 
             y_initial=initial_y,
             device=device,
             n_eval=n_eval,
             return_path=True)
    dyn_wd = compute_dynamic_wd(trajectories)
    static_wd = compute_static_wd(trajectories)
    if print_results:
        print('Dynamic WD:', dyn_wd, '; Static WD:', static_wd)
    return (dyn_wd - static_wd)/static_wd

