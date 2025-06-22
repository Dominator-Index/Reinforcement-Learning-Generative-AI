from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    cosine_beta_schedule,
    linear_beta_schedule)
from .basic import DiffusionModel, FlowMatching
import matplotlib.pyplot as plt
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
from torchcfm.utils import pad_t_like_x

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.optimal_transport import OTPlanSampler

from typing import Optional, Union, Callable, Dict

class ConditionalFlowMatching(FlowMatching):
    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            # ------------------- DPM Params ------------------- #
            predict_noise: bool = True,
            beta_schedule: str = "cosine",  # or cosine
            beta_schedule_params: Optional[dict] = None,

            device: Union[torch.device, str] = "cpu",
            
            sigma: Union[float, int] = 0.0,  # CFM
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.predict_noise = predict_noise

        if beta_schedule_params is None:
            beta_schedule_params = {}
        beta_schedule_params["T"] = self.diffusion_steps

        if beta_schedule == "linear":
            beta = linear_beta_schedule(**beta_schedule_params)
        elif beta_schedule == "cosine":
            beta = cosine_beta_schedule(**beta_schedule_params)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.beta = torch.tensor(beta, device=self.device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.bar_alpha = torch.cumprod(self.alpha.clone(), 0)
        self.x_max, self.x_min = x_max, x_min
        
        self.sigma = sigma

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)
    
    # ---------------------------------------------------------------------------
    # ConditionalFlowMatching
    
    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0
    
    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma
    
    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)
        
    def sample_source(self, x_target, temperature=1.0):
        # 原来 torch.rand_like → 改成 torch.randn_like
        x_source = torch.randn_like(x_target) * temperature
        x_source = x_source * (1. - self.fix_mask) + x_target * self.fix_mask
        return x_source

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut
    
    # ---------------------------------------------------------------------------
    # Training
    
    def loss(self, x_source, x_target, condition=None):
        t, xt, ut = self.sample_location_and_conditional_flow(x_source, x_target)
        condition = self.model["condition"](condition) if condition is not None else None
        vt = self.model["diffusion"](xt, t, condition)
        
        loss = (vt - ut) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update(self, x_target, condition=None, update_ema=True):
        x_source = self.sample_source(x_target, temperature=1.0)
        x_source = x_source * (1. - self.fix_mask) + x_target * self.fix_mask
        
        loss = self.loss(x_source, x_target, condition)
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema: 
            self.ema_update()
        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log

    def update_classifier(self, x_target, condition):
        x_source = self.sample_source(x_target, temperature=1.0)
        x_source = x_source * (1. - self.fix_mask) + x_target * self.fix_mask
        t, xt, _ = self.sample_location_and_conditional_flow(x_source, x_target, return_noise=False)
        
        log = self.classifier.update(xt, t, condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def predict_function(
        self, x, t, bar_alpha=None,
        use_ema=False, requires_grad=False,
        condition_vec_cfg=None, w_cfg: float = 0.0,
        condition_vec_cg=None, w_cg: float = 1.0,
    ):
        """
        前向预测速度场 u(x,t) 并应用 CFG/CG。
        bar_alpha 参数为兼容老接口，当前实现不使用。
        """
        b = x.shape[0]
        model = self.model_ema if use_ema else self.model

        # —— Classifier-Free Guidance —— 
        with torch.set_grad_enabled(requires_grad):
            if 0 < w_cfg < 1:
                x2 = torch.cat([x, x], dim=0)
                t2 = torch.cat([t, t], dim=0)
                c2 = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], dim=0)
                u2 = model["diffusion"](x2, t2, c2)
                u_cond, u_uncond = u2[:b], u2[b:]
                pred = w_cfg * u_cond + (1 - w_cfg) * u_uncond
            elif w_cfg == 0:
                pred = model["diffusion"](x, t, None)
            else:
                pred = model["diffusion"](x, t, condition_vec_cfg)

        # —— Classifier Guidance —— 
        log_p = None
        if self.classifier and w_cg != 0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x, t, condition_vec_cg)
            pred = pred + w_cg * grad

        return pred, {"log_p": log_p}
    
    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None}

        # choose the model
        model = self.model_ema if use_ema else self.model
        
        # check `sample_steps`
        if sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(f"sample_steps != diffusion_steps, sample_steps will be set to diffusion_steps.")
            sample_steps = self.diffusion_steps
        dt = 1.0 / sample_steps
        
        # 2) 初始化 x₁ ~ N(0,I)
        xt = torch.randn((n_samples, *prior.shape[1:]),
                        device=self.device) * temperature
        xt = xt * (1 - self.fix_mask) + prior * self.fix_mask
        
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()
        
        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg
            
        # 4) 反向 Euler 迭代
        for i in range(sample_steps, 0, -1):
            t_val = torch.full((n_samples,), i * dt,
                           device=self.device, dtype=torch.float32)
            
            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t_val, bar_alpha=None,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)
            
            # Euler 步长更新: x ← x + Δt * u
            xt = xt + dt * pred_theta

            # 保持已知部分不变
            xt = xt * (1 - self.fix_mask) + prior * self.fix_mask
            idx = sample_steps - i + 1
            log["sample_history"][:, idx] = xt.cpu().numpy()
        
        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t_val, condition_vec_cg)
            log["log_p"] = logp

        return xt, log
        
    def sample_x(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            extra_sample_steps: int = 8,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        # initialize logger
        log = {"sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape))
            if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # check sample_steps
        if sample_steps is None or sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(
                f"sample_steps != diffusion_steps, resetting to diffusion_steps.")
            sample_steps = self.diffusion_steps
        dt = 1.0 / sample_steps

        # initialize x at t=1
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg)
                if condition_cfg is not None else None)
            condition_vec_cg = condition_cg

        log_p = None
        # main reverse ODE (Euler)
        for i in range(sample_steps, 0, -1):
            t_val = torch.full((n_samples,), i * dt,
                            device=self.device, dtype=torch.float32)

            # predict flow field u
            pred_theta, out = self.predict_function(
                xt, t_val, None,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)

            # Euler update
            xt = xt + dt * pred_theta

            # fix known portion
            xt = xt * (1 - self.fix_mask) + prior * self.fix_mask

            # record history
            if preserve_history:
                step_idx = sample_steps - i + 1
                log["sample_history"][:, step_idx] = xt.cpu().numpy()

            # record last log_p
            log_p = out.get("log_p", log_p)

        # extra steps at t=0 for refinement
        if extra_sample_steps > 0:
            t_val = torch.zeros((n_samples,), device=self.device, dtype=torch.float32)
            for _ in range(extra_sample_steps):
                pred_theta, out = self.predict_function(
                    xt, t_val, None,
                    use_ema=use_ema,
                    requires_grad=requires_grad,
                    condition_vec_cfg=condition_vec_cfg,
                    condition_vec_cg=condition_vec_cg,
                    w_cfg=w_cfg, w_cg=w_cg)
                
                xt = xt + dt * pred_theta
                xt = xt * (1 - self.fix_mask) + prior * self.fix_mask
                log_p = out.get("log_p", log_p)

        # final log_p if missing
        if log_p is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                log_p = self.classifier.logp(xt, t_val, condition_vec_cg)
        log["log_p"] = log_p

        return xt, log

    
class ExactOptimalTransportConditionalFlowMatching(ConditionalFlowMatching):
    """
    Extend ConditionalFlowMatching for Exact Optimal Transport flow.
    Override methods as needed.
    """
    def __init__(
        self,
        nn_diffusion: BaseNNDiffusion,
        nn_condition: Optional[BaseNNCondition] = None,
        fix_mask=None,
        loss_weight=None,
        classifier: Optional[BaseClassifier] = None,
        grad_clip_norm: Optional[float] = None,
        diffusion_steps: int = 1000,
        ema_rate: float = 0.995,
        optim_params: Optional[dict] = None,
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        predict_noise: bool = True,
        beta_schedule: str = "cosine",
        beta_schedule_params: Optional[dict] = None,
        device: Union[torch.device,str] = "cpu",
        sigma: Union[float,int] = 0.0,
    ):
        super().__init__(
            nn_diffusion,
            nn_condition,
            fix_mask,
            loss_weight,
            classifier,
            grad_clip_norm,
            diffusion_steps,
            ema_rate,
            optim_params,
            x_max,
            x_min,
            predict_noise,
            beta_schedule,
            beta_schedule_params,
            device,
            sigma,
        )
        self.ot_sampler = OTPlanSampler(method="exact")
    
    def update(self, x_target, condition=None, update_ema=True):
        x_source = self.sample_source(x_target, temperature=1.0)
        x_source = x_source * (1. - self.fix_mask) + x_target * self.fix_mask
        
        # 使用OT plan配对采样
        x_source_matched, x_target_matched = self.ot_sampler.sample_plan(x_source, x_target)
        loss = self.loss(x_source_matched, x_target_matched, condition=condition)
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema: 
            self.ema_update()
        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log

    def update_classifier(self, x_target, condition):
        x_source = self.sample_source(x_target, temperature=1.0)
        x_source = x_source * (1. - self.fix_mask) + x_target * self.fix_mask
        
        x_source_matched, x_target_matched = self.ot_sampler.sample_plan(x_source, x_target)
        t, xt, _ = self.sample_location_and_conditional_flow(x_source_matched, x_target_matched, return_noise=False)
        
        log = self.classifier.update(xt, t, condition)
        return log
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        # 采样时也用OT配对
        x0_matched, x1_matched = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0=x0_matched, x1=x1_matched, t=t, return_noise=return_noise)