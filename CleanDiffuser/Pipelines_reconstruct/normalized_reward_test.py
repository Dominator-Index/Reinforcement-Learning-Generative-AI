import gymnasium as gym
import minari
import numpy as np

# 加载 HalfCheetah 的离线数据集
dataset = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)
print(dataset.storage.metadata.keys())

# 恢复与数据集相对应的 Gymnasium 环境
env = dataset.recover_environment()

# 这里定义一个随机策略作为示例（你可以替换为你自己的策略）
def random_policy(obs):
    return env.action_space.sample()

# 执行策略，计算 raw return
def evaluate_policy(policy_fn, env, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns)

# 得到原始得分（未归一化）
raw_score = evaluate_policy(random_policy, env)
print(f"Raw return: {raw_score:.2f}")

# 使用 numpy 数组传入，避免类型错误
normalized_score = minari.get_normalized_score(dataset, np.array([raw_score]))[0] * 100
print(f"Normalized score: {normalized_score:.2f}")
