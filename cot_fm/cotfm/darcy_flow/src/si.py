import sys
sys.path.append('../')

import numpy as np
from typing import Tuple, Optional, Callable
from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint
from src.util.gaussian_process import GPPrior, make_grid

class SI(nn.Module):
    '''
    Stochastic Interpolant base class.
    '''

    def __init__(
            self,
            model: nn.Module, 
            gamma_type: str = "none", 
            I_type: str = "linear", 
            p: float = 2., 
            a: float = 1.,
            functional: bool = False,
            kernel_length=0.001, 
            kernel_variance=1.0,
            device='cpu',
            sigma=1e-3,
    ):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.make_gamma(gamma_type, a)
        self.make_alpha_beta(I_type, p)
        self.functional = functional
        if functional:
            self.gp = GPPrior(lengthscale=kernel_length, 
                              var=kernel_variance, 
                              device=device)

    def forward(
            self, 
            t: Float[Tensor, "b"],
            u: Float[Tensor, "b s"],
            y: Optional[Float[Tensor, "b s"]] = None
    ) -> Float[Tensor, "b s"]:
        '''
        Forward pass of the model
        '''
        if y is None:
            return self.model(t, u)
        else:
            return self.model(t, u, y)


    def simulate(
            self,
            t: Float[Tensor, "b"],
            u_0: Float[Tensor, "b s"],
            u_1: Float[Tensor, "b s"],
    ) -> Tuple[Float[Tensor, "b s"], Float[Tensor, "b s"], Float[Tensor, "b s"]]:
        '''
        Simulate u_t from the stochastic interpolant, s.t. u_t = I(t, u_0, u_1) + gamma(t)*z, z ~ N(0, I)
        '''
        if self.functional:
            batch_size = u_0.shape[0]
            n_channels = u_0.shape[1]
            dims = u_0.shape[2:]
            # Sample from prior GP
            query_points = make_grid(dims)
            z = self.gp.sample(query_points, dims, n_samples=batch_size, n_channels=n_channels)
        else:
            z = torch.randn_like(u_0)
        batch_size, num_dims = u_0.shape[0], u_0.ndim - 1
        
        gamma = self.gamma(t).view(batch_size, *((1,) * num_dims))
        u_t = self.I(t, u_0, u_1) + gamma*z
        return u_t, z
            
    def make_alpha_beta(
            self, 
            I_type: str="linear", 
            p: float=2.
    ):
        '''
        Initialize deterministic component of a spatially linear stochastic interpolant, i.e. I(t, u_0, u_1) = alpha(t)*u_0 + beta(t)*u_1
        '''
        if I_type == "linear":
            '''
            Linear Interpolant, i.e. x(t) = (1-t)*u_0 + t*u_1
            References: Liu et al. 2023, Flow straight and fast: Learning to generate and transfer data with rectified flow
                        Lipman et al. 2023, Flow matching for generative modeling
            N.B.: If we add noise by make_noisy(self) and define optimal couplings, then this interpolant is the Schrödinger Bridge interpolant from Tong et al. 2023
            '''
            self.alpha = lambda t: 1 - t
            self.beta = lambda t: t
            self.dalpha = lambda t: -torch.ones_like(t)
            self.dbeta = lambda t: torch.ones_like(t)
        elif I_type == "trig":
            '''
            Trigonometric Interpolant, i.e. x(t) = sqrt(1-gamma(t)^2)*sin(pi*t/2)*u_0 + sqrt(1-gamma(t)^2)*cos(pi*t/2)*u_1
            References: Albergo et al. 2023, Stochastic Interpolants: a unifying framework for flows and diffusions
            '''
            self.alpha = lambda t: torch.sqrt(1-self.gamma(t)**2)*torch.sin(np.pi*t/2)
            self.beta = lambda t: torch.sqrt(1-self.gamma(t)**2)*torch.cos(np.pi*t/2)
            self.dalpha = lambda t: - self.gamma(t)*torch.sin(np.pi*t/2)*self.dgamma(t)/torch.sqrt(1-self.gamma(t)**2) + np.pi/2 * torch.cos(np.pi*t/2) * torch.sqrt(1-self.gamma(t)**2)
            self.dbeta = lambda t: - self.gamma(t)*torch.cos(np.pi*t/2)*self.dgamma(t)/torch.sqrt(1-self.gamma(t)**2) - np.pi/2 * torch.sin(np.pi*t/2) * torch.sqrt(1-self.gamma(t)**2)
        elif I_type == "poly":
            '''
            Polynomial Interpolant of order p, i.e. x(t) = (1-t)^p*u_0 + t^p*u_1
            '''
            self.alpha = lambda t: (1-t)**p
            self.beta = lambda t: t**p
            self.dalpha = lambda t: -p*(1-t)**(p-1)
            self.dbeta = lambda t: p*t**(p-1)
        elif I_type == "encdec":
            '''
            Encoder-decoder interpolant, i.e. x(t) = cos^2(pi*t)*1_{[0,.5)}(t)*u_0 + cos^2(pi*t)*1_{[.5,1]}(t)*u_1
            References: Albergo et al. 2023, Stochastic Interpolants: a unifying framework for flows and diffusions
            '''
            self.alpha = lambda t: torch.where(t < .5, torch.cos(np.pi*t)**2, torch.zeros_like(t))
            self.beta = lambda t: torch.where(t >= .5, torch.cos(np.pi*t)**2, torch.zeros_like(t))
            self.dalpha = lambda t: torch.where(t < .5, -2*np.pi*torch.cos(np.pi*t)*torch.sin(np.pi*t), torch.zeros_like(t))
            self.dbeta = lambda t: torch.where(t >= .5, -2*np.pi*torch.cos(np.pi*t)*torch.sin(np.pi*t), torch.zeros_like(t))
        elif I_type == "diffusion":
            '''
            Variance-preserving diffusion interpolant, i.e. x(t) = sqrt(1-4*t^2)*1_{[0,.5)}*u_0 + sqrt(1-4*(1-t)^2)*1_{[.5,1]}*u_1
            '''
            self.alpha = lambda t: torch.where(t < .5, torch.sqrt(1-4*t**2), torch.zeros_like(t))
            self.beta = lambda t: torch.where(t >= .5, torch.sqrt(1-4*(1-t)**2), torch.zeros_like(t))
            self.dalpha = lambda t: torch.where(t < .5, -4*t/torch.sqrt(1-4*t**2), torch.zeros_like(t))
            self.dbeta = lambda t: torch.where(t >= .5, 4*(1-t)/torch.sqrt(1-4*(1-t)**2), torch.zeros_like(t))
        elif I_type == "sigmoid":
            '''
            Sigmoid interpolant, i.e. x(t) = 1/(1+exp(-p*(2*t-1)))*u_0 + 1/(1+exp(p*(2*t-1)))*u_1
            '''
            self.alpha = lambda t: 1/(1+torch.exp(-p*(2*t-1)))
            self.beta = lambda t: 1/(1+torch.exp(p*(2*t-1)))
            self.dalpha = lambda t: 2*p*torch.exp(-p*(2*t-1))/(1+torch.exp(-p*(2*t-1)))**2
            self.dbeta = lambda t: -2*p*torch.exp(p*(2*t-1))/(1+torch.exp(p*(2*t-1)))**2
        else:
            AssertionError("I_type not recognized")

    def make_gamma(
            self, 
            gamma_type: str="sb", 
            a: float=1.
    ):
        '''
        Random component of the stochastic interpolant, i.e. gamma(t)*z, z ~ N(0, I)
        '''
        if gamma_type == "sb":
            self.gamma = lambda t: torch.sqrt(2*t*(1-t))
            self.dgamma = lambda t: (1 - 2*t)/torch.sqrt(2*t*(1-t))
        elif gamma_type == "sinsq":
            self.gamma = lambda t: torch.sin(np.pi*t)**2
            self.dgamma = lambda t: 2*np.pi*torch.sin(np.pi*t)*torch.cos(np.pi*t)
        elif gamma_type == "a-sb":
            self.gamma = lambda t: torch.sqrt(a*t*(1-t))
            self.dgamma = lambda t: (a*(1 - 2*t))/torch.sqrt(a*t*(1-t))
        elif gamma_type == "none":
            self.gamma = lambda t: torch.zeros_like(t)
            self.dgamma = lambda t: torch.zeros_like(t)
        elif gamma_type == "linear":
            self.gamma = lambda t: torch.where(t < .5, 2*t, 2*(1-t))
            self.dgamma = lambda t: torch.where(t < .5, 2*torch.ones_like(t), -2*torch.ones_like(t))
        elif gamma_type == "constant":
            self.gamma = lambda t: torch.ones_like(t)*self.sigma
            self.dgamma = lambda t: torch.zeros_like(t)
        else:
            AssertionError("gamma_type not recognized")
        

    def I(
            self,
            t: Float[Tensor, "b"],
            u_0: Float[Tensor, "b c h w"],
            u_1: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h w"]:
        '''
        Deterministic component of the stochastic interpolant, i.e. I(t, u_0, u_1)
        '''
        batch_size, num_dims = u_0.shape[0], u_0.ndim - 1
        alpha = self.alpha(t).view(batch_size, *((1,) * num_dims))
        beta = self.beta(t).view(batch_size, *((1,) * num_dims))

        return alpha * u_0 + beta * u_1
    def v(
            self, 
            t: Float[Tensor, "b"], 
            u_0: Float[Tensor, "b c h w"],
            u_1: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h w"]:
        '''
        Conditional vector field, i.e. v(t,x) = E[d_t I(t, u_0, u_1) | u_t = x]
        '''
        batch_size, num_dims = u_0.shape[0], u_0.ndim - 1
        dalpha = self.dalpha(t).view(batch_size, *((1,) * num_dims))
        dbeta = self.dbeta(t).view(batch_size, *((1,) * num_dims))
        
        return dalpha * u_0 + dbeta * u_1
    
    def b(self, t, u_0, u_1, z):
        '''
        Drift term, i.e. b(t, u_t) = v(t, u_0, u_1) + dgamma(t)*z
        '''
        batch_size, num_dims = u_0.shape[0], u_0.ndim - 1
        dgamma = self.dgamma(t).view(batch_size, *((1,) * num_dims))

        return self.v(t, u_0, u_1) + dgamma * z
    
    def s(self, t, z):
        '''
        Score term, i.e. s(t,u_t) = - E[z|u_t = x] / gamma(t)
        '''
        # TODO: find best fix when gamma(t) = 0, see section 6.1 of Albergo et al. 2023
        return -z/(self.gamma(t) + 1e-8)
    
    def loss(self, b, v):
        '''
        Loss function, i.e. ||v(t, u_t) - b(t, u_t)||_F^2
        '''
        def mse(b, v):
            return F.mse_loss(b, v,)

        if isinstance(v, tuple):
            v0, v1 = v
            return mse(b, v0)/2 + mse(b, v1)/2
        else:
            return mse(b, v)

        
    
    def straightness(self, b):
        '''
        Straightness of the interpolant, i.e. ||b(t, u_t)||_2^2
        '''
        return b.norm()**2
        
    @torch.no_grad()
    def sample(self, u_initial, y_initial, direction='f', n_samples=None, n_eval=2, device=None,
               return_path=False, rtol=1e-5, atol=1e-5, lower_tlim=0, upper_tlim=1):
        '''
        Sample from the stochastic interpolant using an ODE, starting from u_initial.

        Arguments:
        - u_initial: [batch_size, n_channels, *dims]
        - direction: 'f' or 'r', forward or reverse in time
        - n_samples: number of samples to generate
        - n_eval: number of timesteps to evaluate
        - return_path: if True, return the entire path of samples, otherwise just the final sample
        - rtol, atol: tolerances for odeint
        '''        
        if device is None:
            device = u_initial.device
        if n_samples is None:
            n_samples = u_initial.shape[0]
        if direction == 'f':
            t = torch.linspace(lower_tlim, upper_tlim, n_eval, device=device)
        elif direction == 'r':
            t = torch.linspace(upper_tlim, lower_tlim, n_eval, device=device)
        inital_batch_size = u_initial.shape[0]
        with torch.no_grad():
            def cond_call(t, u):
                batch_size = u.shape[0]
                t = t.repeat(batch_size)
                return self(t, u, y_initial)

            # if n_samples > inital_batch_size:
            #     # if u_initial has less samples than n_samples, repeat u_initial at random to get n_samples
            #     n_new_samples = n_samples - inital_batch_size
            #     u_initial = torch.cat([u_initial, u_initial[torch.randperm(inital_batch_size)[:n_new_samples]]], dim=0)
            self.model = self.model.to(device)
            u_initial = u_initial.to(device)
            y_initial = y_initial.to(device)
            t = t.to(device)

            method = 'dopri5'
            out = odeint(
                cond_call, 
                u_initial, 
                t, 
                method=method, 
                rtol=rtol, 
                atol=atol
            )

        if return_path:
            return out
        else:
            return out[-1]
        
        
    def compute_div(
        self,
        t: torch.tensor,
        u: torch.tensor,
        y: Optional[torch.tensor] = None
    ) -> torch.tensor:
        # only works in 1D
        b = u.shape[0]
        with torch.set_grad_enabled(True):
            u.requires_grad_(True)
            t.requires_grad_(True)
            if y is not None:
                y.requires_grad_(True)
                f_val = self.forward(t, u, y)
            else:
                f_val = self.forward(t, u)
            divergence = 0.0
            for i in range(u.shape[1]):
                divergence += \
                        torch.autograd.grad(
                                f_val[:, i].sum(), u, create_graph=True
                            )[0][:, i]

        return divergence.view(b)


##### REFERENCES #####
# Albergo et al. 2023, Stochastic Interpolants: a unifying framework for flows and diffusions
# Liu et al. 2023, Flow straight and fast: Learning to generate and transfer data with rectified flow
# Lipman et al. 2023, Flow matching for generative modeling
# Tong et al. 2023, Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport