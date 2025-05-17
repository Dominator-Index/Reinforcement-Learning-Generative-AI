import torch
import ot 
import math
import sklearn
from typing import Callable, Optional
from torchcfm.utils import *
from sklearn.datasets import make_s_curve
def compute_w2_squared(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    计算两个经验分布 x0, x1 之间的 2-Wasserstein 距离的平方 W2^2

    参数:
      x0: Tensor of shape [n, d]，源分布样本
      x1: Tensor of shape [n, d]，目标分布样本

    返回:
      W2_sq: 标量 Tensor，表示 W2^2
    """
    # 样本数 n
    n = x0.shape[0]
    # 构造均匀权重向量 a, b，形状 [n]
    a = torch.ones(n, device=x0.device, dtype=x0.dtype) / n
    b = torch.ones(n, device=x1.device, dtype=x1.dtype) / n
    # 计算成本矩阵 C_{ij} = ||x0_i - x1_j||^2，转为 numpy
    # torch.cdist 返回欧氏距离矩阵
    C = torch.cdist(x0, x1, p=2).pow(2).cpu().numpy()
    # 调用 POT 库的 emd 算法，返回传输计划 pi
    pi = ot.emd(a.cpu().numpy(), b.cpu().numpy(), C)
    # 计算 W2^2 = sum_{i,j} pi_{ij} * C_{ij}
    W2_sq = torch.tensor((pi * C).sum(), device=x0.device, dtype=x0.dtype)
    return W2_sq


def compute_model_PE(model: Callable, x0: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """
    使用欧拉法近似计算模型向量场 v = model(x, t) 的路径能量
    PE = ∫_0^1 E_i[ ||v(t, x_i(t))||^2 ] dt

    参数:
      model: 可调用对象，接口 model(x, t) -> Tensor of shape [n, d]
      x0: Tensor [n, d]，batch 初始样本
      steps: int，时间离散步数

    返回:
      PE: 标量 Tensor，batch-平均路径能量
    """
    # batch size n, 维度 d
    n, d = x0.shape
    # 在 [0,1] 上等距采样 steps+1 个时间点
    t_grid = torch.linspace(0, 1, steps + 1, device=x0.device, dtype=x0.dtype)
    # 复制初始状态，以便就地更新
    x = x0.clone()
    # 初始化标量 PE
    PE = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)
    # 时间步长 dt
    dt = 1.0 / steps

    # 遍历每个时间点，使用简单欧拉积分
    for t in t_grid:
        t_expanded = t.unsqueeze(0).expand(n, 1)  # 将标量 t 拓展为形状 [n, 1]
        # 计算速度场 v(t, x)
        # 假设 model 接受 (x, t) 或 torch.cat([x, t]) 形式
        v = model(torch.cat([x, t_expanded], dim=-1))        # 输出形状 [n, d]  
        # 计算每个样本的瞬时能量 ||v||^2
        inst_energy = v.pow(2).sum(dim=1)  # 形状 [n]
        # 对 batch 求平均，再乘 dt 累加到 PE
        PE = PE + inst_energy.mean() * dt
        # 欧拉步推进状态 x
        x = x + v * dt

    return PE

def compute_model_NPE(model: Callable, x0: torch.Tensor, x1: torch.Tensor, steps: int = 50) -> tuple:
    """
    计算归一化路径能量 NPE
    NPE = |PE - W2^2| / W2^2

    参数:
      model: 可调用对象，接口 model(x, t) -> Tensor[n, d]
      x0: Tensor[n, d]，起始分布样本
      x1: Tensor[n, d]，目标分布样本
      steps: int，路径积分步数

    返回:
      W2_sq: 标量 Tensor，2-Wasserstein 距离平方
      PE: 标量 Tensor，路径能量
      NPE: 标量 Tensor，归一化路径能量
    """
    # 1. 计算 2-Wasserstein 距离平方
    W2_sq = compute_w2_squared(x0, x1)
    # 2. 计算路径能量
    PE = compute_model_PE(model, x0, steps)
    # 3. 避免除零
    denom = W2_sq.clamp(min=1e-12)
    # 4. 计算 NPE
    NPE = (PE - W2_sq).abs() / denom
    return W2_sq, PE, NPE

def compute_objective_variance(model, source, target, batch_size, sigma):
    """
    估计 OV ≈ E_[t,x0,x1][ || u_t(x|z) - v_t(x) ||^2 ]
    其中 u_t(x|z) = x1 - x0, v_t(x) ≈ E_z[u_t(x|z)]
    """
    # 1) 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) 一次性采样 x0, x1, 和每个样本自己的 t
    x0 = sample_source(source, batch_size).to(device)       # [n, d]
    x1 = sample_target(target, batch_size).to(device)       # [n, d]
    t  = torch.rand(batch_size, 1, device=device)          # [n, 1]

    # 3) 条件流 (ground truth)
    u_t = x1 - x0                                           # [n, d]

    # 4) 中间样本 x_t ~ p_t(x|x0,x1)
    x_t = sample_conditional_pt(x0, x1, t, sigma).to(device)# [n, d]

    # 5) 用模型估计边际流 v_t(x)
    inp = torch.cat([x_t, t], dim=-1)                       # [n, d+1]
    v_t = model(inp)                                        # [n, d]

    # 6) 计算 OV：batch 均值的平方误差
    ov = torch.mean((u_t - v_t).pow(2))                     # 标量

    return ov

    
def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def sample_scurve(batch_size, noise=0.05, seed=None):
    data, _ = make_s_curve(n_samples=batch_size, noise=noise, random_state=seed)
    data = data[:, [0, 2]]  # 只保留 x 和 z 两个维度，忽略 y
    return torch.tensor(data, dtype=torch.float32)

def sample_data(method, batch_size=None, noise=0.05, seed=None):
    if method == '8gaussians':
        return sample_8gaussians(batch_size)
    elif method == 'moons':
        return sample_moons(batch_size)
    elif method == 'scurve':
        return sample_scurve(batch_size, noise=noise, seed=seed)
    elif method == 'uniform':
        return torch.rand(batch_size, 2)
    elif method == 'gaussian':
        return torch.randn(batch_size, 2)
    else:
        raise ValueError(f"Unknown method: {method}")

# 修改原来调用的 sample_source 和 sample_target 为 sample_data
def sample_source(method, batch_size=None, noise=0.05, seed=None):
    return sample_data(method, batch_size, noise, seed)

def sample_target(method, batch_size=None, noise=0.05, seed=None):
    return sample_data(method, batch_size, noise, seed)
    
    
        
    