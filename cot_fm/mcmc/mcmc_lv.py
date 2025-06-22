import numpy as np
import torch

import pymc as pm
from pytensor.compile.ops import as_op
from scipy.integrate import odeint as scipy_odeint
import pytensor.tensor as pt
from numba import njit

# Based heavily on https://www.pymc.io/projects/examples/en/latest/ode_models/ODE_Lotka_Volterra_multiple_ways.html

# define the right hand side of the ODE equations in the Scipy odeint signature
@njit
def rhs(X, t, theta):
    # unpack parameters
    x, y = X
    alpha, beta, gamma, delta = theta
    # equations
    dx_dt = alpha * x - beta * x * y
    dy_dt = -gamma * y + delta * x * y
    return [dx_dt, dy_dt]


# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    t_eval = torch.linspace(0, 20, 11)

    return scipy_odeint(func=rhs, y0=[30, 1], t=t_eval, args=(theta,))


def run_mcmc(log_y_obs):
    model = pm.Model()
    with model:
        sigma_theta = np.sqrt(0.5)
        sigma_y = np.sqrt(0.1)
        
        alpha = pm.LogNormal('alpha', mu=-.125, sigma=sigma_theta)
        beta = pm.LogNormal('beta', mu=-3, sigma=sigma_theta)
        gamma = pm.LogNormal('gamma', mu=-.125, sigma=sigma_theta)
        delta = pm.LogNormal('delta', mu=-3, sigma=sigma_theta)
        
        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta])
        )
        
        log_ode_soln = pm.math.log(ode_solution)
        log_y = pm.Normal("Y_obs", mu=log_ode_soln, sigma=sigma_y, observed=log_y_obs)

    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:-1]
    vars_list

    # Run MCMC
    sampler = "DEMetropolis"
    chains = 5
    tune = 50_000
    draws = 10_000
    with model:
        trace_DEM = pm.sample(step=[pm.DEMetropolis(vars_list)], draws=draws, chains=chains)
    trace = trace_DEM

    samples = np.array(trace.posterior.to_array())
    samples = samples.reshape(4, -1).T
    np.random.shuffle(samples)
    samples = samples[:10_000]

    return samples

if __name__ == '__main__':
    seed = 8927
    rng = np.random.default_rng(seed)

    obs= torch.tensor([[ 22.7758,   1.0570, 176.1085,  26.7437,   1.9323,  38.7372,   2.7494,
                       4.2047,  10.1087,   0.9363,  66.3670,   2.6623,   8.8260,  64.2704,
                       3.6616,   7.7439,   7.9582,   1.9387,  25.7259,   0.7588,  53.5384,
                      46.6839]])

    obs = obs.reshape(1, 11, 2)
    log_y_obs = torch.log(obs)

    samples = run_mcmc(log_y_obs)
    #np.save('./lv_mcmc_samples.npy', samples)