import os
os.environ["MUJOCO_GL"] = "egl"   # <<< 一定要在下面这些 import 之前！

import numpy as np
import torch
import gymnasium as gym
import minari
from torch.utils.data import DataLoader

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.classifier import CumRewClassifier
from minari.utils import get_normalized_score
from gymnasium.wrappers import RecordVideo

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ------------------ 数据加载（使用minari） ------------------
horizon = 4
terminal_penalty = -100

# 加载数据集
ds = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)
obs_list = []
next_obs_list = []
act_list = []
rew_list = []
timeout_list = []
terminal_list = []

for i, ep in enumerate(ds.iterate_episodes()):
    obs = ep.observations['observation']  # (T+1, obs_dim)

    # 所有数据统一为长度 T
    obs_list.append(obs[:-1])             # (T, obs_dim)
    next_obs_list.append(obs[1:])         # (T, obs_dim)
    act_list.append(ep.actions)           # (T, act_dim)
    rew_list.append(ep.rewards)           # (T,)
    timeout_list.append(ep.truncations)   # (T,)
    terminal_list.append(ep.terminations) # (T,)

print(f"Collected {len(obs_list)} valid episodes")

data_dict = {
    "observations":      np.concatenate(obs_list, axis=0),         # (N, obs_dim)
    "next_observations": np.concatenate(next_obs_list, axis=0),    # (N, obs_dim)
    "actions":           np.concatenate(act_list, axis=0),         # (N, act_dim)
    "rewards":           np.concatenate(rew_list, axis=0),         # (N,)
    "timeouts":          np.concatenate(timeout_list, axis=0),     # (N,)
    "terminals":         np.concatenate(terminal_list, axis=0)     # (N,)
}

# 构造数据集
dataset = D4RLMuJoCoDataset(data_dict, terminal_penalty=terminal_penalty, horizon=horizon)
obs_dim, act_dim = dataset.o_dim, dataset.a_dim
print(f"obs_dim = {obs_dim}, act_dim = {act_dim}")

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# ------------------ 模型构建 ------------------
nn_diffusion = JannerUNet1d(
    obs_dim + act_dim,
    model_dim=128,
    emb_dim=128,
    dim_mult=[1, 2, 3],
    timestep_emb_type="positional",
    attention=False,
    kernel_size=3
).to(device)

nn_classifier = HalfJannerUNet1d(
    horizon=horizon,
    in_dim = obs_dim + act_dim,
    out_dim=1,
    model_dim=128,
    emb_dim=128,
    dim_mult=[1, 2, 3],
    timestep_emb_type="positional",
    kernel_size=3
).to(device)
classifier = CumRewClassifier(nn_classifier, device=device)

fix_mask = torch.zeros((horizon, obs_dim + act_dim), device=device)
loss_weight = torch.ones((horizon, obs_dim + act_dim), device=device)

agent = DiscreteDiffusionSDE(
    nn_diffusion, nn_condition=None,
    fix_mask=fix_mask, loss_weight=loss_weight,
    classifier=classifier, ema_rate=0.999,
    device=device, diffusion_steps=100, predict_noise=True
)
agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=3e-4)

save_path = "/home/nkd/ouyangzl/CleanDiffuser/Checkpoints"
# ------------------ 训练阶段 ------------------
print("Starting training...")
for i, batch in enumerate(loop_dataloader(dataloader)):
    x = torch.cat([batch["obs"]["state"], batch["act"]], dim=-1).to(device)
    loss = agent.update(x)["loss"]
    if i % 10 == 0:
        print(f"Step {i}, loss: {loss:.4f}")
    if i >= 500:
        break

# ------------------ 评测阶段 ------------------
print("Starting evaluation...")
base_env = ds.recover_environment(render_mode="rgb_array")

env_eval = RecordVideo(
    base_env,
    video_folder="videos",
    name_prefix="kitchen",
    episode_trigger=lambda episode_id: True  # 每个 episode 都录
)

num_episodes = 5
all_rewards = []

for ep in range(num_episodes):
    obs, _ = env_eval.reset()
    done = False
    ep_reward = 0.0
    while not done:
        prior = torch.zeros((1, obs_dim + act_dim), device=device)
        sample, _ = agent.sample(prior, solver="ddpm", n_samples=1, sample_steps=10, w_cfg=0.0)  # dim=3
        action = sample.squeeze(0)[:, obs_dim:].clamp(-1, 1).cpu().numpy()[0]
        obs, reward, terminated, truncated, _ = env_eval.step(action)
        done = terminated or truncated
        ep_reward += reward
    all_rewards.append(ep_reward)
    print(f"Episode {ep+1}, reward: {ep_reward}")

all_rewards = np.array(all_rewards)

# 归一化奖励计算
normalized_rewards = get_normalized_score(ds, all_rewards)
print("Normalized rewards:", normalized_rewards)
