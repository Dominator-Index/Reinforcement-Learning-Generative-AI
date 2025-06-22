'''
Code adapted from https://github.com/TADSGroup/ConditionalOT2023

Paper: Hosseini, B., Hsu, A. W., & Taghvaei, A. (2023). Conditional Optimal Transport on Function Spaces.
'''

import sys
sys.path.append('../')
import numpy as np
import mcmc.pCN_sampler_darcy as pcn
import darcySolver as ds
from scipy.interpolate import RectBivariateSpline
import os
os.nice(10)

import torch
from src.util.gaussian_process import make_grid

class PriorSampler():
    def __init__(
        self,
        prior,
        resolution,
        device,
        dtype = torch.float32,
        batch_size = 5000,
        ):
        self.prior = prior
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.query_points = make_grid(*[resolution])
        self.current_batch = self.generate_batch(batch_size)
        self.current_position = 0
        self.batch_size = batch_size

    
    def generate_batch(self,num_samples):
        prior_samples = self.prior.sample(self.query_points, dims=self.resolution, n_samples=num_samples).float()
        return prior_samples.to(self.device).numpy().reshape(num_samples,-1)
    
    def next(self):
        if self.current_position>=self.batch_size:
            self.current_batch = self.generate_batch(self.batch_size)
            self.current_position = 0
        self.current_position += 1
        return self.current_batch[self.current_position - 1]


def interpolate_for_fenics(xgrid,ygrid,function_values):
    interpolated_permeability_field=RectBivariateSpline(
        prior_grid,
        prior_grid,
        function_values,
        kx=2,
        ky=2
        )
    return lambda x:interpolated_permeability_field(x[0],x[1],grid=False)

def constant_rhs(x):
    return x[0]*0+1

def get_solver(
    observation_x_vals,#assume we observe on a regular grid
    observation_y_vals,
    num_cells = 40,
    f_rhs = constant_rhs,
    ):
    """
    Builds another wrapper for darcy solver
    """
    darcy=ds.DarcySolver(num_x_cells=num_cells,num_y_cells=num_cells)
    def solve(permeability_function,):
        sol=darcy.solve(
            permeability=permeability_function,
            f_rhs=f_rhs,
            )
        sol_observed=sol(observation_x_vals,observation_y_vals)
        return sol_observed
    return solve

def get_phi(
    observed_values,
    noise_level,
    grid_num_observed,
    kernel_x_grid,
    kernel_y_grid,
    ):
    observation_x_vals = np.linspace(0,1,grid_num_observed+2)[1:-1]
    observation_y_vals = np.linspace(0,1,grid_num_observed+2)[1:-1]
    solve_darcy = get_solver(
        observation_x_vals,
        observation_y_vals,
        )

    def phi(permeability_field_values):
        permeability_func = interpolate_for_fenics(
            kernel_x_grid,
            kernel_y_grid,
            np.exp(permeability_field_values).reshape(kernel_points,kernel_points)
            )
        solution = solve_darcy(
            permeability_func
            )
        return np.sum(((solution-observed_values)/noise_level)**2)/2
    return phi


def matern_three_half(rho):
    def k(d):
        return (1+np.sqrt(3)*d/rho)*np.exp(-np.sqrt(3)*d/rho)
    return k

def thin_and_summarize(A,thin_rate = 100):
    mean = np.mean(A,axis=0)
    cov = np.cov(A.T)
    A_thinned = A[::thin_rate]
    return A_thinned,mean,cov

from src.util.gaussian_process import GPPrior
import gpytorch

nu = 3/2
kernel_length = 0.5
kernel_variance = 1.

base_kernel = gpytorch.kernels.MaternKernel(nu,eps=1e-10)
base_kernel.lengthscale = kernel_length
covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
covar_module.outputscale = kernel_variance
gp = GPPrior(covar_module)

y_observed = torch.load('../data/train_loader_0.pth')[5][0].numpy()
true_field = torch.load('../data/train_loader_0.pth')[5][1].numpy()
prior_sampler = PriorSampler(gp, true_field.shape[1:], 'cpu')

save_folder = '../MCMC/darcy/sample5'


grid_num_observed = 100
obs_grid = np.linspace(0,1,grid_num_observed)

kernel_points = 40
prior_grid = np.linspace(0,1,kernel_points)

solve = get_solver(obs_grid,obs_grid,num_cells = 40)
permeability = interpolate_for_fenics(
    prior_grid,
    prior_grid,
    np.exp(true_field).reshape(kernel_points,kernel_points)
    )
observed_values = solve(permeability)


phi = get_phi(
    observed_values = observed_values,
    noise_level = 0.025,
    grid_num_observed = grid_num_observed,
    kernel_x_grid = prior_grid,
    kernel_y_grid = prior_grid,
    )

mcmc_sampler = pcn.pCN_sampler(
    prior_sampler,
    phi,
    kernel_points**2
)

burnin_samples,beta_history,acceptance,weight_history,phi_vals = mcmc_sampler.burn_in_tune_beta(
        prior_sampler.next(),
        beta_initial = 0.03,
        target_acceptance = 0.3,
        num_samples = 10000,
        long_alpha = 0.99,
        short_alpha = 0.9,
        adjust_rate = 0.03,
        beta_min = 1e-3,
        beta_max = 1-1e-3,
    )

np.save(f"{save_folder}/burnin_samples.npy",burnin_samples)
np.save(f"{save_folder}/beta_history.npy",beta_history)
np.save(f"{save_folder}/burnin_acceptance.npy",acceptance)
np.save(f"{save_folder}/burnin_phi.npy",phi_vals)

thinning_rate = 100

beta_val = np.mean(beta_history[-500:])
u_initial = burnin_samples[-1]

num_batches = 5000
for i in range(num_batches):
    u_samples,acceptance,phi_vals = mcmc_sampler.get_samples(
        num_samples = int(20000),
        beta = beta_val,
        u0 = u_initial
    )
    path_prefix = f"{save_folder}/batch_{f'{000+i}'.zfill(3)}"
    u_thinned,mean,cov = thin_and_summarize(u_samples,thin_rate = thinning_rate)


    np.save(path_prefix + "u_samples.npy",u_thinned)
    np.save(path_prefix + "acceptance.npy",acceptance)
    np.save(path_prefix + "phi_vals.npy",phi_vals)
    np.save(path_prefix + "mean.npy",mean)
    np.save(path_prefix + "cov.npy",cov)

    u_initial = u_samples[-1]
    print("percent_done", (i+1)/num_batches)
    print("acceptance_rate", np.mean(acceptance))
    for val in phi_vals[::thinning_rate]:
        print("phi", val)
