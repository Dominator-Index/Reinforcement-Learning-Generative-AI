import torch
import numpy as np

from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.utils import *

class LotkaVolterraData:

    def __init__(self):
        return
    
    def sample_source(self, batch_size, log=False, scaler=None):
        samples = self.sample_target(batch_size, log=log, scaler=scaler)
        samples[:, -4:] = torch.randn(batch_size, 4)
        
        return samples

    def sample_target(self, batch_size, log=False, scaler=None):
        obs, _, params = simulate(batch_size, n_eval=11, params=None, log=log)
        obs = obs.reshape(batch_size, -1)
        
        yu = torch.hstack([obs, params])
        yu = yu.float()
        
        if scaler:
            yu = scaler.transform(yu)
        
        return yu

def sample_prior_params(n_samples, log=False):
    # Samples [n_samples, 4] parameters from the prior

    # prior is log(U) ~ N(mu, sigma^2)
    mu = torch.tensor([-0.125, -3, -0.125, -3])
    sigma_squared = torch.tensor([0.5])
    
    eps = torch.randn((n_samples, 4))
    out = mu + torch.sqrt(sigma_squared) * eps
    
    if not log:
        out = torch.exp(out)
        
    return out

def lotka_volterra(t, p, params):
    # p: [batch, 2] of current state
    # params: [batch, 4]
    # Returns Lotka-Volterra vector field: [batch, 2]
            
    alpha, beta, gamma, delta = torch.tensor_split(params, 4, dim=1)
    alpha, beta, gamma, delta = torch.squeeze(alpha), torch.squeeze(beta), torch.squeeze(gamma), torch.squeeze(delta)
    
    p1 = p[:, 0]
    p2 = p[:, 1]
        
    out = torch.empty_like(p)
    out[:, 0] = alpha * p1 - beta * p1 * p2
    out[:, 1] = -gamma * p2 + delta * p1 * p2
    
    return out

def simulate(batch_size, n_eval=11, params=None, log=False, clamp_val=1e-4):
    # Simulates noisy observations of the Lotka-Volterra system [batch_size, n_eval, 2]
    # If params is none, simulates from the prior
    
    initial_condition = torch.empty((batch_size, 2))
    initial_condition[:, 0] = 30
    initial_condition[:, 1] = 1
    
    t_eval = torch.linspace(0, 20, n_eval)
    
    if params is None:
        params = sample_prior_params(batch_size, log=False)
    
    # Solve ODE
    ode = lambda t, p: lotka_volterra(t, p, params) 
    ode_soln = odeint(ode, initial_condition, t_eval)  # [n_eval, batch_size, 2]
    
    ode_soln = ode_soln.clamp(min=clamp_val)
    log_ode_soln = torch.log(ode_soln)
    
    # Add log-normal noise
    eps = torch.randn((n_eval, batch_size, 2))
    sigma = np.sqrt(0.1)
    log_obs = log_ode_soln + eps * sigma
    
    if not log:
        obs = torch.exp(log_obs)
    else:
        obs = log_obs
        params = torch.log(params)
        
    ode_soln = ode_soln.permute([1, 0, 2])
    obs = obs.permute([1, 0, 2])
    
    return obs, ode_soln, params

def sample_specific_y(model, log=False, scaler=None, clamp_val=1e-4, n_samples=10_000, device='cuda'):
    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )

    bins = 25
    n_samples = 10_000
    
    # Sample a single observation for this given value of u
    u = torch.tensor([[0.83, 0.041, 1.08, 0.04]])
    
    # Fixed observation (y) used in all experiments
    obs= torch.tensor([[ 22.7758,   1.0570, 176.1085,  26.7437,   1.9323,  38.7372,   2.7494,
                           4.2047,  10.1087,   0.9363,  66.3670,   2.6623,   8.8260,  64.2704,
                           3.6616,   7.7439,   7.9582,   1.9387,  25.7259,   0.7588,  53.5384,
                          46.6839]])

    obs = torch.log(obs)
    
    initial_data = torch.empty((n_samples, 26))    
    initial_data[:, :22] = obs
    if scaler:
        initial_data = scaler.transform(initial_data)
    initial_data[:, 22:] = torch.randn((n_samples, 4))
    initial_data = initial_data.float()

    with torch.no_grad():
        traj = node.trajectory(
            initial_data.to(device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )

        samples = traj[-1].cpu().numpy()

        if scaler:
            samples = scaler.inverse_transform(samples)
        if log:
            samples = torch.exp(samples)

    return samples