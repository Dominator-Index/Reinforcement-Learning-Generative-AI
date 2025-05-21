import math
import os
import time
from typing import Tuple 

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.optimal_transport import OTPlanSampler

from typing import Optional, Union, Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)

from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.utils import to_tensor
from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from torchdiffeq import odeint

SUPPORTED_SOLVERS = [
    "ddpm", "ddim",
    "ode_dpmsolver_1", "ode_dpmsolver++_1", "ode_dpmsolver++_2M",
    "sde_dpmsolver_1", "sde_dpmsolver++_1", "sde_dpmsolver++_2M",]

class FlowMatching:

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            # NN backbone for the diffusion model
            nn_diffusion: BaseNNDiffusion,
            # Add a condition-process NN to enable classifier-free-guidance
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

            device: Union[torch.device, str] = "cpu"
    ):
        if optim_params is None:
            optim_params = {"lr": 2e-4, "weight_decay": 1e-5}

        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.diffusion_steps = diffusion_steps
        self.ema_rate = ema_rate

        # nn_condition is None means that the model is not conditioned on any input.
        if nn_condition is None:
            nn_condition = IdentityCondition()

        # In the code implementation of Diffusion models, it is common to maintain an exponential
        # moving average (EMA) version of the model for inference, as it has been observed that
        # this approach can result in more stable generation outcomes.
        self.model = nn.ModuleDict({
            "diffusion": nn_diffusion.to(self.device),
            "condition": nn_condition.to(self.device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        
        self.model.train()
        self.model_ema.eval()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)

        self.classifier = classifier

        self.fix_mask = to_tensor(fix_mask, self.device)[None, ] if fix_mask is not None else 0.
        self.loss_weight = to_tensor(loss_weight, self.device)[None, ] if loss_weight is not None else 1.

    def train(self):
        self.model.train()
        if self.classifier is not None:
            self.classifier.model.train()

    def eval(self):
        self.model.eval()
        if self.classifier is not None:
            self.classifier.model.eval()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])

class BaseFlowMatchingAgent(FlowMatching):
    
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

            # ------------------ Plugins ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- Diffusion Params ------------------- #
            epsilon: float = 1e-3,

            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max = x_max.to(device) if isinstance(x_max, torch.Tensor) else x_max
        self.x_min = x_min.to(device) if isinstance(x_min, torch.Tensor) else x_min

    @property
    def supported_solvers(self):
        return SUPPORTED_SOLVERS

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):
        raise NotImplementedError

    def loss(self, x0, condition=None, **kwargs):

        xt, t, eps = self.add_noise(x0)

        condition = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2
        
        loss = loss * self.loss_weight * (1 - self.fix_mask)
        
        # find weighted_regression_tensor in kwargs
        weighted_regression_tensor = kwargs.get("weighted_regression_tensor", None)
        if weighted_regression_tensor is not None:
            loss *= weighted_regression_tensor.unsqueeze(-1)

        return loss.mean()

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        """One-step gradient update.
        Inputs:
        - x0: torch.Tensor
            Samples from the target distribution.
        - condition: Optional
            Condition of x0. `None` indicates no condition.
        - update_ema: bool
            Whether to update the exponential moving average model.

        Outputs:
        - log: dict
            The log dictionary.
        """
        loss = self.loss(x0, condition, **kwargs)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x0, condition):

        xt, t, eps = self.add_noise(x0)

        log = self.classifier.update(xt, t, condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """
        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    condition = torch.cat([condition, torch.zeros_like(condition)], 0)
                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), condition)
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                pred = model["diffusion"](xt, t, condition)
                pred_uncond = 0.

        if self.predict_noise or not self.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred

        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.predict_noise:
            if self.clip_pred:
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
        else:
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)

        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(self, *args, **kwargs):
        raise NotImplementedError

class ConditionalFlowMatchingAgent(BaseFlowMatchingAgent):
    def __init__(
            self,
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,
            classifier: Optional[BaseClassifier] = None,
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,
            epsilon: float = 1e-3,
            diffusion_steps: int = 1000,
            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,
            predict_noise: bool = True,
            device: Union[torch.device, str] = "cpu",
            sigma: Union[float, int] = 0.0,  # CFM
    ):
        super().__init__(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            fix_mask=fix_mask,
            loss_weight=loss_weight,
            classifier=classifier,
            grad_clip_norm=grad_clip_norm,
            diffusion_steps=diffusion_steps,
            ema_rate=ema_rate,
            optim_params=optim_params,
            device=device
        )
        self.epsilon = epsilon
        self.diffusion_steps = diffusion_steps
        self.discretization = discretization
        self.noise_schedule = noise_schedule
        self.noise_schedule_params = noise_schedule_params
        self.x_max = x_max
        self.x_min = x_min
        self.predict_noise = predict_noise
        self.sigma = sigma
        
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
    
    def loss(self, x0, x1, t=None, return_noise=False, condition=None, **kwargs ):
        
        t, xt, ut = self.sample_location_and_conditional_flow(x0, x1, t, return_noise=return_noise)
        condition = self.model["condition"](condition) if condition is not None else None
        vt = self.model["diffusion"](xt, t, condition)   # 这里可能会有问题 condition的维度
        loss = (vt - ut) ** 2
        
        loss = loss * self.loss_weight * (1 - self.fix_mask)
        
        # find weighted_regression_tensor in kwargs
        weighted_regression_tensor = kwargs.get("weighted_regression_tensor", None)
        if weighted_regression_tensor is not None:
            loss *= weighted_regression_tensor.unsqueeze(-1)

        return loss.mean()  
    
    def update(self, x1, condition=None, update_ema=True, **kwargs):
        """One-step gradient update.
        Inputs:
        - x0: torch.Tensor
            Samples from the source distribution.
        - x1: torch.Tensor
            Samples from the target distribution.
        - condition: Optional
            Condition of x0. `None` indicates no condition.
        - update_ema: bool
            Whether to update the exponential moving average model.

        Outputs:
        - log: dict
            The log dictionary.
        """
        x0 = self.sample_source(x1, temperature=1.0)
        loss = self.loss(x0, x1, condition=condition, **kwargs)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x1, R):
        x0 = self.sample_source(x1, temperature=1.0)
        # 1) 用 flow-matching 的方式，采一个 (t, x_t)
        #    并且把 return_noise=True，拿到 eps，以防你想记录（虽然分类器不需要 eps）
        t, xt, _ = self.sample_location_and_conditional_flow(x0, x1, return_noise=False)

        # 2) 把 x_t 和 t 传给分类器去预测累计回报 R
        #    `noise` 这个位置，用 t
        log = self.classifier.update(xt, t, R)

        return log
    
    def classifier_guidance_flow(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        model: Dict[str, nn.Module],
        condition: Optional[torch.Tensor] = None,
        w: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Flow-Matching 下的 Classifier Guidance:
        v = v_theta(x_t, t)
        if classifier exists and w>0:
            logp, grad = classifier.logp_and_grad(x_t, t, condition)
            v_guided = v + w * grad
        else:
            v_guided = v
        返回 (v_guided, logp)
        """
        # 1) 先算原始的流场 v_theta(x_t,t)
        #    假设你的 flow model 在 model["diffusion"]
        #    接口是 forward(x, t, cond)
        v = model["diffusion"](xt, t, condition) 
        
        # 2) 若无 classifier 或 w==0，直接返回 v
        if self.classifier is None or w == 0.0:
            return v, None
        
        # 3. 用分类器计算 logp 和 grad
        logp, grad = self.classifier.gradients(xt.clone(), t, condition)

        # 4. Flow-guided vector field
        v_guided = v + w * grad
        return v_guided, logp
    
    def classifier_free_guidance_flow(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        model: Dict[str, nn.Module],
        condition: Optional[torch.Tensor] = None,
        w: float = 1.0,
        pred: Optional[torch.Tensor] = None,
        pred_uncond: Optional[torch.Tensor] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Flow Matching 下的 Classifier-Free Guidance:
        bar_v = w * v_cond + (1 - w) * v_uncond

        Inputs:
        - xt:          (B, *dim)         当前样本 x_t
        - t:           (B,)              时间步 t
        - model:       {"diffusion": flow_model}
        - condition:   (B, *c_dim) 或 None
        - w:           guidance scale
        - pred:        v_cond (可选), already computed
        - pred_uncond: v_uncond (可选), already computed
        - requires_grad: 是否需要 grad (一般为 False)
        Output:
        - bar_v:       (B, *dim)   插值后的 vector field
        """

        with torch.set_grad_enabled(requires_grad):
            B = xt.shape[0]

            # 1) 如果既没有传 pred 也没有传 pred_uncond，就一起算
            if pred is None or pred_uncond is None:
                # 构造一个长度 2B 的 batch: 前 B 做有条件，后 B 做无条件
                # xt_repeat: (2B, *dim)
                xt_repeat = xt.repeat(2, *[1 for _ in range(xt.dim()-1)])
                # t_repeat:  (2B,)
                t_repeat = t.repeat(2)
                # condition_repeat: (2B, *c_dim)  — 对后半批填 0
                if condition is not None:
                    cond_uncond = torch.zeros_like(condition)
                    condition_repeat = torch.cat([condition, cond_uncond], dim=0)
                else:
                    condition_repeat = None

                # 一次前向：得到 (2B, *dim) 的流场  model["diffusion"](xt, t, condition)
                all_pred = model["diffusion"](xt_repeat, t_repeat, condition_repeat)
                # 拆分
                pred = all_pred[:B]
                pred_uncond = all_pred[B:]

            # 2) 线性插值
            bar_v = w * pred + (1.0 - w) * pred_uncond

        return bar_v
    
    def clip_prediction(self, pred, *args, **kwargs):
        return pred
    
    def guided_sampling_flow(
        self,
        xt: torch.Tensor,                        # (B, *dim)
        t: torch.Tensor,                         # (B,)
        model: Dict[str, nn.Module],             # {"diffusion": flow_model}
        condition: Optional[torch.Tensor] = None,# (B, *c_dim)
        w_cf: float = 0.0,                       # classifier-free scale
        w_cg: float = 0.0,                       # classifier-guidance scale
        requires_grad: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        统一的 Flow-Matching Guided Sampling：
        1) Classifier-Free Guidance:  v_cf = w_cf⋅v_cond + (1−w_cf)⋅v_uncond
        2) Classifier Guidance:       v = v_cf + w_cg⋅∇_x log p(c|x_t, t)

        Returns:
        - v_guided: (B, *dim)  最终的速度场
        - logp:      (B,)      分类器得分（如果 w_cg>0 且有 classifier）
        """
        # 1) 先算有条件和无条件的流场
        #    model["diffusion"](x, t, cond) -> v_cond, model["diffusion"](x, t, None) -> v_uncond
        v_cond   = model["diffusion"](xt, t, condition)
        v_uncond = model["diffusion"](xt, t, None)

        # 2) Classifier-Free 插值
        if w_cf == 1.0:
            v_cf = v_cond
        elif w_cf == 0.0:
            v_cf = v_uncond
        else:
            v_cf = w_cf * v_cond + (1.0 - w_cf) * v_uncond

        # 3) Classifier Guidance（在 v_cf 基础上加梯度）
        logp = None
        if self.classifier is not None and w_cg != 0.0:
            # 需要对 xt 做梯度追踪
            xtg = xt.clone().detach().requires_grad_(True)
            logp, grad = self.classifier.gradients(xtg, t, condition)
            v_guided = v_cf + w_cg * grad
        else:
            v_guided = v_cf

        return v_guided, logp
    
    def flow_matching_sample(
        self,
        prior,
        n_samples,
        sample_steps,
        condition_cfg=None,
        w_cfg=1.0,
        condition_cg=None,
        w_cg=0.0,
        preserve_history=False,
        warm_start_reference=None,
        warm_start_forward_level=0.8,
        temperature=1.0,
        sample_step_schedule="linear",
        use_ema=True,
        requires_grad=True,
        solver="ode_rk4"
    ):
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape[1:])) if preserve_history else None
        }

        model = self.model_ema if use_ema else self.model

        # ===== 初始化起点 =====
        prior = prior.to(self.device)   # [64, 32, 68]
        if isinstance(warm_start_reference, torch.Tensor):
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask   # [64, 32, 68]
        
        if requires_grad:
            xt.requires_grad_(True)
        else:
            xt = xt.detach()


        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        # ===== 条件处理 =====
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, self.mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg  # classifier 的条件输入

        # ===== 定义 ODE =====
        def ode_dynamics(t_scalar, x):
            # t_scalar: float tensor scalar, shape []
            t = torch.full((x.shape[0],), t_scalar.item(), device=x.device, dtype=x.dtype)
            # 使用 unified guidance 接口
            velocity, _ = self.guided_sampling_flow(
                xt=x,
                t=t,
                model=model,
                condition=condition_vec_cfg,
                w_cf=w_cfg,
                w_cg=w_cg,
                requires_grad=requires_grad
            )
            return velocity

        # ===== 积分时间点 =====
        if sample_step_schedule == "linear":
            t_span = torch.linspace(1.0, 0.0, sample_steps + 1, device=self.device)   # torch.Size([21])
        else:
            raise NotImplementedError(f"Unknown schedule: {sample_step_schedule}")

        # ===== 调用 torchdiffeq.odeint 做 trajectory 计算 =====
        with torch.set_grad_enabled(requires_grad):
            traj = odeint(
                ode_dynamics,
                xt,
                t_span,
                atol=1e-4,
                rtol=1e-4,
                method=solver.replace("ode_", "")
            )  # [steps+1, B, *dim]
            traj = traj.permute(1, 0, *range(2, traj.ndim))  # [B, steps+1, *dim]   # torch.Size([64, 21, 32, 68])

        # ===== 保存轨迹 =====
        if preserve_history:
            log["sample_history"][:] = traj.cpu().numpy()

        # ===== 计算 logp（optional）=====
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.float32, device=self.device)  # torch.Size([3200])
                logp = self.classifier.logp(xt, t, condition_vec_cg)   # 在这个位置报错
            log["log_p"] = logp  # [64, 1]

        # ===== clip 后处理 =====
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log

class ExactOptimalTransportConditionalFlowMatchingAgent(ConditionalFlowMatchingAgent):
    def __init__(
            self,
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,
            classifier: Optional[BaseClassifier] = None,
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,
            epsilon: float = 1e-3,
            diffusion_steps: int = 1000,
            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,
            predict_noise: bool = True,
            device: Union[torch.device, str] = "cpu",
            sigma: Union[float, int] = 0.0,  # CFM
    ):
        super().__init__(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            fix_mask=fix_mask,
            loss_weight=loss_weight,
            classifier=classifier,
            grad_clip_norm=grad_clip_norm,
            ema_rate=ema_rate,
            optim_params=optim_params,
            epsilon=epsilon,
            diffusion_steps=diffusion_steps,
            discretization=discretization,
            noise_schedule=noise_schedule,
            noise_schedule_params=noise_schedule_params,
            x_max=x_max,
            x_min=x_min,
            predict_noise=predict_noise,
            device=device,
            sigma=sigma,
        )
        self.ot_sampler = OTPlanSampler(method="exact")

    def update(self, x1, condition=None, update_ema=True, **kwargs):
        # x1: target batch
        # x0: source batch (noise)
        x0 = self.sample_source(x1, temperature=1.0)
        # 使用OT plan配对采样
        x0_matched, x1_matched = self.ot_sampler.sample_plan(x0, x1)
        loss = self.loss(x0_matched, x1_matched, condition=condition, **kwargs)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log

    def update_classifier(self, x1, R):
        x0 = self.sample_source(x1, temperature=1.0)
        x0_matched, x1_matched = self.ot_sampler.sample_plan(x0, x1)
        t, xt, _ = self.sample_location_and_conditional_flow(x0_matched, x1_matched, return_noise=False)
        log = self.classifier.update(xt, t, R)
        return log

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        # 采样时也用OT配对
        x0_matched, x1_matched = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0_matched, x1_matched, t, return_noise)





