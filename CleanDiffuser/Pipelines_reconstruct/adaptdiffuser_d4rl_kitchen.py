# 注意：一定要在所有 import 之前设置环境变量
import os
os.environ["MUJOCO_GL"] = "egl"   # 使用 EGL 后端渲染

import time
import numpy as np
import torch
import gymnasium as gym  # 使用 gymnasium 替代原 gym
import minari
from torch.utils.data import DataLoader

import hydra
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.utils import report_parameters
from minari.utils import get_normalized_score
from gymnasium.wrappers import RecordVideo
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import set_seed  # 用户自定义的种子设置函数

@hydra.main(config_path="../configs/adaptdiffuser/kitchen", config_name="kitchen", version_base=None)
def pipeline(args):
    # ------------------ 初始化和配置 ------------------
    set_seed(args.seed)
    device = args.device  # 例如 "cuda:0" 或 "cpu"

    # 保存路径
    save_path = f"results/{args.pipeline_name}/{args.task.env_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ------------------ 数据加载（使用 minari 和 gymnasium） ------------------
    # 加载 kitchen 数据集（minari）替代 d4rl 数据集
    horizon = args.task.horizon       # 数据中统一截取的轨迹长度
    terminal_penalty = args.terminal_penalty  # 终止惩罚值

    ds = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)

    # 构造各种列表，用于存储所有 episode 的数据
    obs_list = []
    next_obs_list = []
    act_list = []
    rew_list = []
    timeout_list = []
    terminal_list = []

    # 遍历所有 episode，注意：每个 episode 的 observation 包含 (T+1) 个状态
    for i, ep in enumerate(ds.iterate_episodes()):
        obs = ep.observations['observation']  # (T+1, obs_dim)
        # 统一截取长度 T
        obs_list.append(obs[:-1])             # (T, obs_dim)
        next_obs_list.append(obs[1:])           # (T, obs_dim)
        act_list.append(ep.actions)             # (T, act_dim)
        rew_list.append(ep.rewards)             # (T,)
        timeout_list.append(ep.truncations)     # (T,)
        terminal_list.append(ep.terminations)   # (T,)

    print(f"Collected {len(obs_list)} valid episodes")

    # 将列表中的数据按维度拼接，保证每个数组至少为 1 维
    data_dict = {
        "observations":      np.concatenate(obs_list, axis=0),      # (N, obs_dim)
        "next_observations": np.concatenate(next_obs_list, axis=0), # (N, obs_dim)
        "actions":           np.concatenate(act_list, axis=0),      # (N, act_dim)
        "rewards":           np.concatenate(rew_list, axis=0),      # (N,)
        "timeouts":          np.concatenate(timeout_list, axis=0),  # (N,)
        "terminals":         np.concatenate(terminal_list, axis=0)  # (N,)
    }

    # 构造 minari 版本的数据集
    dataset = D4RLMuJoCoDataset(data_dict, terminal_penalty=terminal_penalty, horizon=horizon)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    print(f"obs_dim = {obs_dim}, act_dim = {act_dim}")

    # 数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    # ------------------ 模型构建 ------------------
    # 构造扩散模型（JannerUNet1d) 与 分类器（HalfJannerUNet1d）
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,
        model_dim=args.model_dim,      # 例如：128
        emb_dim=args.model_dim,        # 例如：128
        dim_mult=args.task.dim_mult,   # 例如：[1, 2, 3]
        timestep_emb_type="positional",
        attention=False,
        kernel_size=3
    ).to(device)

    nn_classifier = HalfJannerUNet1d(
        horizon=horizon,
        in_dim=obs_dim + act_dim,
        out_dim=1,
        model_dim=args.model_dim,     # 例如：128
        emb_dim=args.model_dim,       # 例如：128
        dim_mult=args.task.dim_mult,  # 例如：[1, 2, 3]
        timestep_emb_type="positional",
        kernel_size=3
    ).to(device)
    # 打印模型参数信息
    print("=============== Diffusion Model Parameters ===============")
    report_parameters(nn_diffusion)
    print("=============== Classifier Model Parameters ===============")
    report_parameters(nn_classifier)

    # 构造累计奖励分类器
    classifier = CumRewClassifier(nn_classifier, device=device)

    # 构造 fix_mask 和 loss_weight（根据轨迹长度与状态动作维度）
    fix_mask = torch.zeros((horizon, obs_dim + act_dim), device=device)
    loss_weight = torch.ones((horizon, obs_dim + act_dim), device=device)

    # ------------------ 构造离散扩散模型代理 ------------------
    agent = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=args.ema_rate,
        device=device,
        diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise
    )
    # 设置扩散模型优化器（例如学习率 3e-4）
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)

    # ------------------ 训练阶段 ------------------
    if args.mode == "train":
        # 使用余弦退火调度器，可根据需要添加
        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.0, "avg_loss_classifier": 0.0}
        start_time = time.time()

        # 无限循环的 dataloader（可自动重复）
        for batch in loop_dataloader(dataloader):
            # 将 batch 中的“obs”与“act”数据转移到 device
            # 假设 dataset 返回的 batch 格式：
            #   batch["obs"]["state"]: (B, obs_dim)
            #   batch["act"]: (B, act_dim)
            x = torch.cat([batch["obs"]["state"], batch["act"]], dim=-1).to(device)

            # 执行扩散模型更新
            cur_loss = agent.update(x)["loss"]
            log["avg_loss_diffusion"] += cur_loss
            diffusion_lr_scheduler.step()

            # 对前面若干步更新分类器
            if n_gradient_step <= args.classifier_gradient_steps:
                # 使用 dummy 条件值，由于原数据集内有“val”
                val = batch["val"].to(device)  # (B, ?)
                cur_loss_cls = agent.update_classifier(x, val)["loss"]
                log["avg_loss_classifier"] += cur_loss_cls
                classifier_lr_scheduler.step()

            # ----------- 日志打印 -----------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(f"Step {log['gradient_steps']} | "
                      f"Avg Diffusion Loss: {log['avg_loss_diffusion']:.4f} | "
                      f"Avg Classifier Loss: {log['avg_loss_classifier']:.4f} | "
                      f"Time: {time.time()-start_time:.2f}s")
                log = {"avg_loss_diffusion": 0.0, "avg_loss_classifier": 0.0}

            # ----------- 模型保存 -----------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(os.path.join(save_path, f"diffusion_ckpt_{n_gradient_step+1}.pt"))
                agent.classifier.save(os.path.join(save_path, f"classifier_ckpt_{n_gradient_step+1}.pt"))

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ------------------ 评测阶段 ------------------
    elif args.mode == "inference":
        # 如果指定了模型路径，加载模型；否则报错
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model path for inference")
        agent.model.eval()
        agent.model_ema.eval()

        # 从 minari 数据集中恢复环境
        base_env = ds.recover_environment(render_mode="rgb_array")
        # 包装环境记录视频，每个 episode 都录
        env_eval = RecordVideo(
            base_env,
            video_folder="videos",
            name_prefix="kitchen",
            episode_trigger=lambda episode_id: True
        )

        num_episodes = args.num_episodes  # 评测 episode 数量
        all_rewards = []

        for ep in range(num_episodes):
            obs, _ = env_eval.reset()
            done = False
            ep_reward = 0.0
            while not done:
                # 构造初始先验（单步输入）
                prior = torch.zeros((1, obs_dim + act_dim), device=device)
                # 使用扩散采样获得下一动作；参数可根据需要调整
                sample, _ = agent.sample(prior, solver="ddpm", n_samples=1, sample_steps=args.sampling_steps, w_cfg=0.0)
                # 从采样结果中提取动作部分，并限制范围
                action = sample.squeeze(0)[:, obs_dim:].clamp(-1, 1).cpu().numpy()[0]
                # 环境一步交互
                obs, reward, terminated, truncated, _ = env_eval.step(action)
                done = terminated or truncated
                ep_reward += reward
            all_rewards.append(ep_reward)
            print(f"Episode {ep+1}, reward: {ep_reward}")

        all_rewards = np.array(all_rewards)
        normalized_rewards = get_normalized_score(ds, all_rewards)
        print("Normalized rewards:", normalized_rewards)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    pipeline()
