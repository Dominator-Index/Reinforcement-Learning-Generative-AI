import math
import os
import time
import sklearn.datasets
import sklearn.datasets
from tqdm import tqdm
import pandas as pd
from sklearn.datasets import make_s_curve
import sys
sys.path.append('/home/nkd/ouyangzl/conditional-flow-matching/runner/src')  # 添加到更高一层路径
import torch
from models.components.distribution_distances import compute_distribution_distances
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
from Metric import *
methods = {'OT-CFM': ExactOptimalTransportConditionalFlowMatcher,
           'CFM': ConditionalFlowMatcher,
           'FM': VariancePreservingConditionalFlowMatcher,
           'Reg. CNF': TargetConditionalFlowMatcher,
           'SB-CFM': SchrodingerBridgeConditionalFlowMatcher,
           'ICNN': ExactOptimalTransportConditionalFlowMatcher}
 
BATCH_SIZE = 256
SIGMA = 0.1

TIME_STEP = 100
DIM = 2
def train_OT_CFM(model_factory, sources, targets, seeds, method='OTCFM', epochs=20000, batch_size=None, sigmas=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Traning {method} starts!')
    results = []
    for sigma in sigmas:
        print(f'Training with sigma: {sigma}')
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(
            sigma=sigma,
        )
        for seed in seeds:
            for source_name, target_name in zip(sources, targets):
                print(f'Training {source_name} to {target_name}...')
                # 设定随机种子
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                # 模型和优化器
                model = model_factory().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                # Start
                beginning = time.time()
                start = beginning
                for epoch in tqdm(range(epochs), desc="Training Epochs"):
                    # 采样源和目标数据
                    source = sample_source(source_name, batch_size, seed=seed).to(device)
                    target = sample_target(target_name, batch_size, seed=seed).to(device)
                    optimizer.zero_grad()
                    # 采样，注意使用 torch.rand 而非 torch.rand_like（source.shape[0]是整数）
                    t = torch.rand(source.shape[0], device=device, dtype=source.dtype)
                    t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(source, target, t)
                    v_t = model(torch.cat([x_t, t[:, None]], dim=-1))
                    # 计算损失
                    loss = torch.mean((v_t - u_t) ** 2)
                    loss.backward()
                    optimizer.step()
                    
                    # 使用 tqdm.write 输出日志
                    if (epoch + 1) % 500 == 0:
                        tqdm.write(f"Epoch {epoch + 1}: loss {loss.item():.4f}, method {method}, seed {seed}, sigma {sigma}")

                    if (epoch + 1) % 5000 == 0:
                        end = time.time()
                        tqdm.write(f"Epoch {epoch + 1}: loss {loss.item():.4f}, time {end - start:.2f}s, method {method}, seed {seed}, sigma {sigma}")
                        start = end
                        node = NeuralODE(
                            torch_wrapper(model),
                            solver="dopri5",
                            sensitivity="adjoint",
                            atol=1e-4,
                            rtol=1e-4
                        )
                        with torch.no_grad():
                            traj = node.trajectory(
                                sample_source(source_name, 1024, seed=seed).to(device),
                                t_span=torch.linspace(0, 1, 100, device=device),
                            )
                        plot_trajectories(traj.cpu().numpy(), save_path=f'/home/nkd/ouyangzl/conditional-flow-matching/images/{method}_{source_name}_{target_name}_{epoch + 1}_{seed}_{sigma}_{batch_size}.png')
                end = time.time()
                # Save model
                torch.save(model.state_dict(), f'/home/nkd/ouyangzl/conditional-flow-matching/checkpoints/{method}_{source_name}_{target_name}_{epoch + 1}_{seed}_{sigma}_{batch_size}.pth')
                
                
                # 重新采样
                eval_bs = batch_size * 2
                eval_src = sample_source(source_name, eval_bs).to(device)
                eval_tgt = sample_target(target_name, eval_bs).to(device)
                # 计算 W2, PE, NPE
                W2_sq, PE, NPE = compute_model_NPE(model, eval_src, eval_tgt, steps=TIME_STEP)
                total_time = time.time() - beginning
                
                print(f'{method} with {sigma} training completed!')
                print(f'W2: {torch.sqrt(W2_sq)}, PE: {torch.sqrt(PE)}, NPE: {NPE}')
                results.append({
                    'source': source_name,
                    'target': target_name,
                    'method': method,
                    'sigma': sigma,
                    "W2": torch.sqrt(W2_sq).item(),
                    'PE': torch.sqrt(PE).item(),
                    'NPE': NPE.item(),
                    'time': total_time,
                    'ov': compute_objective_variance(model, source_name, target_name, eval_bs, sigma).item(),
                    'batch size': batch_size,
                    'seed': seed
                })
    
    df = pd.DataFrame(results)
    df.to_csv(f'./results/{method}_results.csv', index=False)

def train_CFM(model_factory, sources, targets, seeds, method='CFM', epochs=20000, batch_size=None, sigmas=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Traning {method} starts!')
    results = []
    for sigma in sigmas:
        print(f'Training with sigma: {sigma}')
        flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        for seed in seeds:
            for source_name, target_name in zip(sources, targets):
                # 设定随机种子
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                # 模型和优化器
                model = model_factory().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                # Start
                beginning = time.time()
                start = beginning
                for epoch in tqdm(range(epochs), desc="Training Epochs"):
                    source = sample_source(source_name, batch_size, seed=seed).to(device)
                    target = sample_target(target_name, batch_size, seed=seed).to(device)
                    optimizer.zero_grad()
                    # 使用 torch.rand 生成与 batch size 相同数量的时间向量
                    t = torch.rand(source.shape[0], device=device, dtype=source.dtype)
                    t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(source, target, t)
                    # 注意这里将 t 转换成形状 [n, 1]，使用 t[:, None]
                    v_t = model(torch.cat([x_t, t[:, None]], dim=-1))
                    loss = torch.mean((v_t - u_t) ** 2)
                    loss.backward()
                    optimizer.step()
                    
                    # 使用 tqdm.write 输出日志
                    if (epoch + 1) % 500 == 0:
                        tqdm.write(f"Epoch {epoch + 1}: loss {loss.item():.4f}, method {method}, seed {seed}, sigma {sigma}")
                    
                    if (epoch + 1) % 5000 == 0:
                        end = time.time()
                        tqdm.write(f"Epoch {epoch + 1}: loss {loss.item():.4f}, time {end - start:.2f}s, method {method}, seed {seed}, sigma {sigma}")
                        start = end
                        node = NeuralODE(
                            torch_wrapper(model),
                            solver="dopri5",
                            sensitivity="adjoint",
                            atol=1e-4,
                            rtol=1e-4 
                        )
                        with torch.no_grad():
                            traj = node.trajectory(
                                sample_source(source_name, 1024, seed=seed).to(device),
                                t_span=torch.linspace(0, 1, 100, device=device),
                            )
                        plot_trajectories(traj.cpu().numpy(), save_path=f'/home/nkd/ouyangzl/conditional-flow-matching/images/{source_name}_{target_name}_{method}_{epoch + 1}_{seed}_{sigma}_{batch_size}.png')
                end = time.time()
                # Save model
                torch.save(model.state_dict(), f'/home/nkd/ouyangzl/conditional-flow-matching/checkpoints/{method}_{source_name}_{target_name}_{epoch + 1}_{seed}_{sigma}_{batch_size}.pth')
                
                # 重新采样
                eval_bs = batch_size * 2
                eval_src = sample_source(source_name, eval_bs).to(device)
                eval_tgt = sample_target(target_name, eval_bs).to(device)
                # 计算 W2, PE, NPE
                W2_sq, PE, NPE = compute_model_NPE(model, eval_src, eval_tgt, steps=TIME_STEP)
                total_time = time.time() - beginning
                
                print(f'{method} with {sigma} training completed!')
                print(f'W2: {torch.sqrt(W2_sq)}, PE: {torch.sqrt(PE)}, NPE: {NPE}')
                results.append({
                    'source': source_name,
                    'target': target_name,
                    'method': method,
                    'sigma': sigma,
                    "W2": torch.sqrt(W2_sq).item(),
                    'PE': torch.sqrt(PE).item(),
                    'NPE': NPE.item(),
                    'time': total_time,
                    'ov': compute_objective_variance(model, source_name, target_name, eval_bs, sigma).item(),
                    'batch size': batch_size,
                    'seed': seed
                })
                
    df = pd.DataFrame(results)
    df.to_csv(f'./results/{method}_results.csv', index=False)

def sample_scurve(batch_size, noise=0.05, seed=None):
    data, _ = make_s_curve(n_samples=batch_size, noise=noise, random_state=seed)
    data = data[:, [0, 2]]  # 只保留 x 和 z 两个维度，忽略 y
    return torch.tensor(data, dtype=torch.float32)
    
def sample_data(method, batch_size=BATCH_SIZE, noise=0.05, seed=None):
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
def sample_source(method, batch_size=BATCH_SIZE, noise=0.05, seed=None):
    return sample_data(method, batch_size, noise, seed)

def sample_target(method, batch_size=BATCH_SIZE, noise=0.05, seed=None):
    return sample_data(method, batch_size, noise, seed)


def create_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _init():
        return MLP(dim=DIM, time_varying=True).to(device)
    return _init

def main():
    model_factory = create_model()
    seeds = [42, 43, 44, 45]
    sources = ['8gaussians', 'gaussian', 'gaussian', 'gaussian', 'uniform', 'uniform', 'uniform']
    targets = ['moons', 'moons', '8gaussians', 'scurve', 'moons', '8gaussians', 'scurve']
    sigmas = [0.1, 0.3, 1]
    batch_sizes = [256, 512, 1024]
    
    for batch_size in batch_sizes:
        # 训练 CFM
        train_CFM(
            model_factory=model_factory,
            sources=sources,
            targets=targets,
            seeds=seeds,
            method='CFM',
            epochs=20000,
            batch_size=batch_size,
            sigmas=sigmas
        )

        # 训练 OTCFM
        train_OT_CFM(
            model_factory=model_factory,
            sources=sources,
            targets=targets,
            seeds=seeds,
            method='OTCFM',
            epochs=20000,
            batch_size=batch_size,
            sigmas=sigmas
        )
    
if __name__ == "__main__":
    main()
    
    
    





