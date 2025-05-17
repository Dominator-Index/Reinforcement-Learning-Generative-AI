import numpy as np
import minari
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset

# 设置参数
horizon = 4   # halfcheetah任务中horizon=4即可
terminal_penalty = -100

# 1. 用minari加载数据集（自动下载）
ds = minari.load_dataset("mujoco/halfcheetah/medium-v0", download=True)

# 2. 遍历所有episode，将observations, actions, rewards, truncations和terminations分别收集
obs_list, act_list, rew_list, timeout_list, terminal_list = [], [], [], [], []

for ep in ds.iterate_episodes():
    obs_list.append(ep.observations)
    act_list.append(ep.actions)
    rew_list.append(ep.rewards)
    timeout_list.append(ep.truncations)  # 使用truncations代替原infos["timeouts"]
    terminal_list.append(ep.terminations)

# 3. 拼接所有episode数据
observations = np.concatenate(obs_list, axis=0)
actions      = np.concatenate(act_list, axis=0)
rewards      = np.concatenate(rew_list, axis=0)
timeouts     = np.concatenate(timeout_list, axis=0)
terminals    = np.concatenate(terminal_list, axis=0)

data_dict = {
    "observations": observations,
    "actions":      actions,
    "rewards":      rewards,
    "timeouts":     timeouts,
    "terminals":    terminals
}

# 4. 利用D4RLMuJoCoDataset构造数据集
dataset = D4RLMuJoCoDataset(data_dict, terminal_penalty=terminal_penalty, horizon=horizon)

# 5. 获取数据集的obs_dim和act_dim并打印
obs_dim, act_dim = dataset.o_dim, dataset.a_dim
print(f"obs_dim = {obs_dim}, act_dim = {act_dim}")

# ...后续代码，例如训练、测试等流程可以接入该数据集...
