import time
import math
import numpy as np
import torch

import math
import os
import time

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

from torchcfm.conditional_flow_matching import (
    VariancePreservingConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    ConditionalFlowMatcher,
    pad_t_like_x,
)
from torchcfm.optimal_transport import OTPlanSampler

# --- 数据集采样函数（示例实现，请根据实际情况替换） ---
def sample_8gaussians(batch_size):
    # 请替换为实际 8gaussians 的采样代码
    return torch.randn(batch_size, 2)

def sample_moons(batch_size):
    # 请替换为实际 moons 的采样代码
    return torch.randn(batch_size, 2)

def sample_moons_8gaussians(batch_size):
    # 请替换为实际 moons-8gaussians 的采样代码
    return torch.randn(batch_size, 2)

def sample_scurve(batch_size):
    # 请替换为实际 scurve 的采样代码
    return torch.randn(batch_size, 2)

# --- 测试文件中通用函数 ---
def random_samples(shape, batch_size=128):
    if isinstance(shape, int):
        shape = [shape]
    return [torch.randn(batch_size, *shape), torch.randn(batch_size, *shape)]

def compute_xt_ut(method, x0, x1, t_given, sigma, epsilon):
    if method == "vp_cfm":
        sigma_t = sigma
        mu_t = torch.cos(math.pi / 2 * t_given) * x0 + torch.sin(math.pi / 2 * t_given) * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (math.pi / 2 *
            (torch.cos(math.pi / 2 * t_given) * x1 - torch.sin(math.pi / 2 * t_given) * x0))
    elif method == "t_cfm":
        sigma_t = 1 - (1 - sigma) * t_given
        mu_t = t_given * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (x1 - (1 - sigma) * computed_xt) / sigma_t
    elif method == "sb_cfm":
        sigma_t = sigma * torch.sqrt(t_given * (1 - t_given))
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = ((1 - 2 * t_given) /
            (2 * t_given * (1 - t_given) + 1e-8) *
            (computed_xt - (t_given * x1 + (1 - t_given) * x0))
            + x1 - x0)
    elif method in ["exact_ot_cfm", "i_cfm"]:
        sigma_t = sigma
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = x1 - x0
    return computed_xt, computed_ut

def get_flow_matcher(method, sigma):
    if method == "vp_cfm":
        fm = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif method == "t_cfm":
        fm = TargetConditionalFlowMatcher(sigma=sigma)
    elif method == "sb_cfm":
        fm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="sinkhorn")
    elif method == "exact_ot_cfm":
        fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif method == "i_cfm":
        fm = ConditionalFlowMatcher(sigma=sigma)
    return fm

def sample_plan(method, x0, x1, sigma):
    if method == "sb_cfm":
        x0, x1 = OTPlanSampler(method="sinkhorn", reg=2 * (sigma**2)).sample_plan(x0, x1)
    elif method == "exact_ot_cfm":
        x0, x1 = OTPlanSampler(method="exact").sample_plan(x0, x1)
    return x0, x1

# --- 评价指标函数（示例实现，请按论文定义替换） ---
def compute_2wasserstein_error(x_true, x_pred):
    # 这里简单用平均欧氏距离代替，请根据论文定义计算 2-Wasserstein Error
    return torch.mean(torch.norm(x_true - x_pred, dim=1)).item()

def compute_normalized_path_energy(flow_field):
    # 这里用平均平方范数代替，请根据论文定义计算 normalized path energy
    return torch.mean(flow_field**2).item()

# --- 实验设置 ---
# 我们在下面的 mapping 中只包含了 test_fm 中支持的三种方法，
# 其对应关系可为：
# OT-CFM -> exact_ot_cfm, CFM -> t_cfm, FM -> vp_cfm
methods = {
    "OT-CFM": "exact_ot_cfm",
    "CFM": "t_cfm",
    "FM": "vp_cfm",
    # 其他方法如 Reg. CNF, CNF, ICNN 需要额外实现
}

datasets = {
    "8gaussians": sample_8gaussians,
    "moons-8gaussians": sample_moons_8gaussians,
    "moons": sample_moons,
    "scurve": sample_scurve,
}

# --- 实验流程函数 ---
def run_experiment(method_name, dataset_name, sigma=0.5, num_steps=1000, batch_size=128):
    # 固定随机种子
    torch.manual_seed(1994)
    np.random.seed(1994)
    
    # 获取数据采样函数，并采样 x0 (源分布) 与 x1 (目标分布)
    dataset_func = datasets[dataset_name]
    x0 = dataset_func(batch_size)
    x1 = dataset_func(batch_size)
    
    method_key = methods[method_name]
    fm = get_flow_matcher(method_key, sigma)
    
    # 模拟训练过程：我们不更新参数，仅调用 forward 模块，记录训练时间
    start_time = time.time()
    for _ in range(num_steps):
        t = torch.rand(batch_size)
        # 执行一次前向传播
        fm.sample_location_and_conditional_flow(x0, x1, return_noise=False)
    train_time = time.time() - start_time
    
    # 评估：在一个 batch 上计算测试指标
    torch.manual_seed(1994)
    t, xt, ut, eps = fm.sample_location_and_conditional_flow(x0, x1, return_noise=True)
    
    # 构造模拟 ground truth：利用 compute_xt_ut 计算 x_t 对应的值
    torch.manual_seed(1994)
    t_given_init = torch.rand(batch_size)
    t_given = t_given_init.reshape(-1, *([1] * (x0.dim() - 1)))
    sigma_pad = pad_t_like_x(sigma, x0)
    epsilon = torch.randn_like(x0)
    x_true, _ = compute_xt_ut(method_key, x0, x1, t_given, sigma_pad, epsilon)
    
    # 计算指标
    w2_error = compute_2wasserstein_error(x_true, xt)
    path_energy = compute_normalized_path_energy(ut)
    
    return w2_error, path_energy, train_time

def main():
    num_runs = 3  # 为了获得均值与标准差
    results = {}
    for method_name in methods.keys():
        results[method_name] = {}
        for dataset_name in datasets.keys():
            w2_errors, energies, times = [], [], []
            for _ in range(num_runs):
                err, energy, t_time = run_experiment(method_name, dataset_name)
                w2_errors.append(err)
                energies.append(energy)
                times.append(t_time)
            results[method_name][dataset_name] = {
                "w2_error": (np.mean(w2_errors), np.std(w2_errors)),
                "energy": (np.mean(energies), np.std(energies)),
                "train_time": (np.mean(times), np.std(times)),
            }
    
    # 打印结果（表格格式）
    print("Method\tDataset\tTest 2-Wasserstein Error\tNormalized Path Energy\tTrain time (s)")
    for method_name in results:
        for dataset_name in results[method_name]:
            r = results[method_name][dataset_name]
            print(f"{method_name}\t{dataset_name}\t"
                  f"{r['w2_error'][0]:.3f} ± {r['w2_error'][1]:.3f}\t"
                  f"{r['energy'][0]:.3f} ± {r['energy'][1]:.3f}\t"
                  f"{r['train_time'][0]:.3f} ± {r['train_time'][1]:.3f}")
    
if __name__ == "__main__":
    main()