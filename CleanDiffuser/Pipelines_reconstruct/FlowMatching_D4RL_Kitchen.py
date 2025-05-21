# 注意：一定要在所有 import 之前设置环境变量
import os
os.environ["MUJOCO_GL"] = "egl"   # 使用 EGL 后端渲染

import logging
import sys
sys.path.append("/home/nkd/ouyangzl/CleanDiffuser/pipelines")
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
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.utils import report_parameters
from minari.utils import get_normalized_score
from gymnasium.wrappers import RecordVideo
from torch.optim.lr_scheduler import CosineAnnealingLR
from flow import ConditionalFlowMatchingAgent

from utils import set_seed  # 用户自定义的种子设置函数

def make_env(ds):
    def _init():
        return ds.recover_environment()
    return _init

# 创建logs目录
os.makedirs("logs", exist_ok=True)

# 配置 logging
logging.basicConfig(
    level=logging.INFO,  # 可设为 DEBUG / INFO / WARNING
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log", mode='w'),
        logging.StreamHandler()  # 同时输出到终端
    ]
)

logger = logging.getLogger(__name__)

@hydra.main(config_path="/home/nkd/ouyangzl/CleanDiffuser/configs_reconstruct/adaptdiffuser/kitchen", config_name="kitchen", version_base=None)
def pipeline(args):
    # ------------------ 初始化和配置 ------------------
    set_seed(args.seed)
    device = args.device  # 例如 "cuda:0" 或 "cpu"
    
    # 保存路径
    save_path = f"/home/nkd/ouyangzl/CleanDiffuser/Checkpoints/{args.pipeline_name}/{args.task.env_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # ------------------ 数据加载（使用 minari 和 gymnasium） ------------------
    # 加载 kitchen 数据集（minari）替代 d4rl 数据集
    horizon = args.task.horizon       # 数据中统一截取的轨迹长度
    # terminal_penalty = args.terminal_penalty  # 终止惩罚值
    
    # 处理 task 得到 task_dir
    task_str = args.task.env_name
    prefix, domain, *rest = task_str.split("-")
    task_dir = f"{prefix}/{domain}/{'-'.join(rest)}"  # 不能嵌套双引号


    ds = minari.load_dataset(task_dir, download=True)
    
    # 构造各种列表，用于存储所有 episode 的数据
    obs_list = []
    next_obs_list = []
    act_list = []
    rew_list = []
    timeout_list = []
    terminal_list = []
    
    logger.info(f'Data collection starts!')
    # 遍历所有 episode，注意：每个 episode 的 observation 包含 (T+1) 个状态
    for i, ep in enumerate(ds.iterate_episodes()):
        obs = ep.observations['observation']
        # 统一截取长度 T
        obs_list.append(obs[:-1])             # (T, obs_dim)
        next_obs_list.append(obs[1:])        # (T, obs_dim)
        act_list.append(ep.actions)          # (T, act_dim)
        rew_list.append(ep.rewards)          # (T,)
        timeout_list.append(ep.truncations)   # (T,)
        terminal_list.append(ep.terminations)  # (T,)
    logger.info(f"Collected {len(obs_list)} valid episodes")
    logger.info(f'Data collection finished!')
    
    logger.info(f'Transforming list to numpy array...')
    # 将列表中的数据按维度拼接，保证每个数组至少为 1 维
    data_dict = {
        "observations":      np.concatenate(obs_list, axis=0),      # (N, obs_dim)
        "next_observations": np.concatenate(next_obs_list, axis=0), # (N, obs_dim)
        "actions":           np.concatenate(act_list, axis=0),      # (N, act_dim)
        "rewards":           np.concatenate(rew_list, axis=0),      # (N,)
        "timeouts":          np.concatenate(timeout_list, axis=0),  # (N,)
        "terminals":         np.concatenate(terminal_list, axis=0)  # (N,)
    }
    logger.info(f"Data processing finished!")
    
    # 构造 minari 版本的数据集
    dataset = D4RLKitchenDataset(data_dict, horizon=horizon)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    logger.info(f"obs_dim = {obs_dim}, act_dim = {act_dim}")
    
    # 数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    
    # ------------------ 模型构建 ------------------
    # 构造扩散模型（JannerUNet1d) 与 分类器（HalfJannerUNet1d）
    logger.info(f'Building models...')
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
    
    # ---------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim), device=args.device)
    fix_mask[0, :obs_dim] = 1.  # 第一时间步保持状态不变
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim), device=args.device)
    loss_weight[0, obs_dim:] = args.action_loss_weight  # 第一时间步动作部分设置指定权重
    
    logger.info(f'Creating agent...')
    # --------------- Diffusion Model --------------------
    agent = ConditionalFlowMatchingAgent(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ---------------------- Training ----------------------
    if args.mode == "train":
        logger.info(f'Training starts...')
        # 设置扩散模型和分类器的余弦退火学习率调度器
        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        # 将模型设置为训练模式
        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}
        start_time = time.time()
        
        logger.info(f"Training {args.diffusion_gradient_steps} steps...")
        # 通过无限循环的 DataLoader 获取 batch 数据
        for batch in loop_dataloader(dataloader):
            # 将 batch 中的观测（状态）、动作、以及附加信息转移至指定设备
            obs = batch["obs"]["state"].to(args.device)   # 形状 (B, obs_dim)
            act = batch["act"].to(args.device)              # 形状 (B, act_dim)
            val = batch["val"].to(args.device)              # 附加条件，可用于分类器更新

            # 拼接状态和动作构成完整输入
            x = torch.cat([obs, act], dim=-1)

            # ----------- 扩散模型梯度更新 -----------
            diff_loss = agent.update(x)["loss"]
            log["avg_loss_diffusion"] += diff_loss
            diffusion_lr_scheduler.step()

            # ----------- 分类器更新（仅在前几步更新） -----------
            if n_gradient_step <= args.classifier_gradient_steps:
                cls_loss = agent.update_classifier(x, val)["loss"]
                log["avg_loss_classifier"] += cls_loss
                classifier_lr_scheduler.step()

            # ----------- 日志打印 -----------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                logger.info(f"[{time.time() - start_time:.2f}s] Step {log['gradient_steps']} | "
                    f"Avg Diffusion Loss: {log['avg_loss_diffusion']:.4f} | "
                    f"Avg Classifier Loss: {log['avg_loss_classifier']:.4f}")
                log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

            # ----------- 模型保存 -----------
            if (n_gradient_step + 1) % args.save_interval == 0:
                logger.info(f'Saving model at step {n_gradient_step + 1}...')
                agent.save(os.path.join(save_path, f"diffusion_ckpt_{n_gradient_step + 1}.pt"))
                agent.classifier.save(os.path.join(save_path, f"classifier_ckpt_{n_gradient_step + 1}.pt"))

            # 更新梯度步数计数器
            n_gradient_step += 1
            # 达到设定梯度步数后退出训练循环
            if n_gradient_step >= args.diffusion_gradient_steps:
                logger.info(f'Training finished!')
                break
    
    elif args.mode == "finetune":
        # 加载预训练的扩散模型和分类器检查点
        agent.load(os.path.join(save_path, f"diffusion_ckpt_{args.ft_ckpt}.pt"))
        agent.classifier.load(os.path.join(save_path, f"classifier_ckpt_{args.ft_ckpt}.pt"))
        
        # 切换到评估模式，方便生成高质量轨迹
        agent.eval()
        
        # 分配缓冲区用于存储选中的 synthetic trajectories，大小为 50000 条轨迹，每条轨迹长度为 horizon
        traj_buffer = torch.empty((50000, args.task.horizon, obs_dim + act_dim), device=args.device)
        sample_bs, preserve_bs, ptr = 20000, 1000, 0

        # 构造 DataLoader，用于遍历整个数据集采样生成轨迹
        gen_dl = DataLoader(
            dataset, batch_size=sample_bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
        )

        # 遍历生成的 DataLoader，生成 synthetic trajectories
        for batch in loop_dataloader(gen_dl):
            # 为当前 batch 创建先验张量，形状为 (sample_bs, horizon, obs_dim+act_dim)
            prior = torch.zeros((sample_bs, args.task.horizon, obs_dim + act_dim), device=args.device)
            # 使用 batch 中每个样本的初始状态填充先验（假设 batch["obs"]["state"] 的形状为 (B, obs_dim)）
            prior[:, 0, :obs_dim] = batch["obs"]["state"][:, 0].to(args.device)
            
            # 利用当前模型采样生成轨迹，返回的 traj 形状为 (sample_bs, horizon, obs_dim+act_dim)
            traj, log = agent.flow_matching_sample(
                prior,
                n_samples=sample_bs,
                sample_steps=args.sampling_steps,
                solver=args.solver,
                use_ema=args.use_ema,
                w_cg=args.task.w_cg,
                temperature=args.temperature
            )
            # log["log_p"] 中存储每个轨迹在第0步的 log_probability (或其它得分)
            logp = log["log_p"]

            # 过滤掉低质量轨迹，选出 logp 第0步大于阈值的轨迹
            selected_traj = traj[logp[:, 0] > args.task.metric_value]
            num_selected = selected_traj.shape[0]
            # 防止选出的轨迹数量超出缓冲区大小
            if ptr + num_selected > 50000:
                num_selected = 50000 - ptr
                selected_traj = selected_traj[:num_selected]
            # 存入轨迹缓冲区，并更新写入指针
            traj_buffer[ptr:ptr + num_selected] = selected_traj
            ptr += num_selected

            print(f'{num_selected} of 10000 trajs have been selected. Progress: {ptr} / 50000')
            if ptr == 50000:
                break

        # 进入自我进化微调阶段
        agent.train()
        agent.optimizer.learning_rate = 1e-5   # 设置微调阶段学习率较低

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "gradient_steps": 0}
        # 在微调阶段，从缓冲区随机采样小批量轨迹进行更新
        while n_gradient_step < 200_000:
            # 从 traj_buffer 中随机选择 32 条轨迹作为训练输入
            x = traj_buffer[torch.randint(0, 50000, (32,))]
            loss_out = agent.update(x)
            log["avg_loss_diffusion"] += loss_out['loss']
            # 每 1000 步打印一次训练日志
            if (n_gradient_step + 1) % 1000 == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= 1000
                print(log)
                log = {"avg_loss_diffusion": 0., "gradient_steps": 0}
            # 每 5000 步保存一次微调后的模型检查点
            if (n_gradient_step + 1) % 5_000 == 0:
                agent.save(os.path.join(save_path, f"finetuned_diffusion_ckpt_{n_gradient_step + 1}.pt"))
                agent.save(os.path.join(save_path, f"finetuned_diffusion_ckpt_latest.pt"))
            n_gradient_step += 1
        
    elif args.mode == "inference":
        # 加载最新的 finetuned 模型检查点
        agent.load(os.path.join(save_path, f"finetuned_diffusion_ckpt_{args.ckpt}.pt"))
        agent.classifier.load(os.path.join(save_path, f"classifier_ckpt_{args.ckpt}.pt"))
        
        # 切换至评估模式
        agent.eval()
        
        # 创建 vectorized 环境，用于并行评测
        env_eval_fns = [make_env(ds) for _ in range(args.num_envs)]
        env_eval = gym.vector.AsyncVectorEnv(env_eval_fns)
        # 获取归一化器，用于将环境观测归一化
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        # 初始化先验张量，形状 (num_envs, horizon, obs_dim+act_dim)，作为采样的输入
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)

        # 评测指定数量的 episode
        for i in range(args.num_episodes):
            # 重置环境：obs shape 为 (num_envs, obs_dim)
            obs, _ = env_eval.reset()
            ep_reward = 0.0
            cum_done = None
            t = 0

            # 循环直到所有环境结束或达到最大时间步 (280+1)
            while not np.all(cum_done) and t < (280 + 1):
                # 对观测进行归一化并转换为 tensor，形状为 (num_envs, obs_dim)
                obs_tensor = torch.tensor(normalizer.normalize(obs), device=args.device)
                # 更新先验张量第一时间步的状态部分：只更新前 obs_dim 个维度
                prior[:, 0, :obs_dim] = obs_tensor

                # 采样候选轨迹：
                # prior.repeat(args.num_candidates, 1, 1) 生成形状 (num_candidates*num_envs, horizon, obs_dim+act_dim)
                # 采样返回的 traj 形状为 (num_candidates*num_envs, horizon, obs_dim+act_dim)
                traj, log = agent.flow_matching_sample(
                    prior.repeat(args.num_candidates, 1, 1),
                    solver=args.solver,
                    n_samples=args.num_candidates * args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema,
                    w_cg=args.task.w_cg,
                    temperature=args.temperature
                )

                # 处理 log_probability：
                # 重塑 log["log_p"] 为 (num_candidates, num_envs, remaining)
                # 然后沿最后一个维度（通常是时间步内的得分）求和，得到 (num_candidates, num_envs)
                logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                # 对每个环境（第二维）选择最佳候选的索引
                idx = logp.argmax(0)  # 形状为 (num_envs,)

                # 将 traj 重塑为 (num_candidates, num_envs, horizon, obs_dim+act_dim)
                traj_reshaped = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)
                # 对每个环境选取最佳候选的第 0 步动作部分，从 obs_dim 开始表示动作
                act_tensor = traj_reshaped[idx, torch.arange(args.num_envs), 0, obs_dim:]
                # 限制动作范围，并转换为 numpy 数组（形状 (num_envs, act_dim)）
                act = act_tensor.clip(-1., 1.).cpu().numpy()

                # 使用选定的动作与环境交互，返回新的观测、奖励、完成状态等
                obs, rew, done, info = env_eval.step(act)
                t += 1
                # 如果 cum_done 尚未初始化则设为 done，否则逐步进行逻辑或
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew

                print(f"[t={t}] cum_rew: {ep_reward}, logp: {logp[idx, torch.arange(args.num_envs)]}")

            # 限制每个 episode 的累计奖励在 [0,4] 范围内（根据任务最大奖励设定）
            episode_rewards.append(np.clip(ep_reward, 0., 4.))

        # 将所有 episode 的奖励转为 numpy 数组，并计算均值和标准差
        episode_rewards = get_normalized_score(ds, episode_rewards)
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    elif args.mode == "video":
        # 加载最新的 finetuned 模型检查点
        agent.load(os.path.join(save_path, f"finetuned_diffusion_ckpt_{args.ckpt}.pt"))
        agent.classifier.load(os.path.join(save_path, f"classifier_ckpt_{args.ckpt}.pt"))
        
        # 切换至评估模式
        agent.eval()
        
        # 创建 single env，用于录制视频
        base_env = ds.recover_environment(render_mode="rgb_array")
        
        env_eval = RecordVideo(
            base_env,
            video_folder="videos",
            name_prefix="kitchen",
            episode_trigger=lambda episode_id: True  # 每个 episode 都录
        )
        # 获取归一化器，用于将环境观测归一化
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        # 初始化先验张量，形状 (num_envs, horizon, obs_dim+act_dim)，作为采样的输入
        prior = torch.zeros((1, args.task.horizon, obs_dim + act_dim), device=args.device)

        # 评测指定数量的 episode
        for i in range(args.num_episodes):
            # 重置环境：obs shape 为 (num_envs, obs_dim)
            obs, _ = env_eval.reset()
            ep_reward = 0.0
            cum_done = None
            t = 0

            # 循环直到所有环境结束或达到最大时间步 (280+1)
            while not np.all(cum_done) and t < (280 + 1):
                # 对观测进行归一化并转换为 tensor，形状为 (num_envs, obs_dim)
                obs_tensor = torch.tensor(normalizer.normalize(obs), device=args.device)
                # 更新先验张量第一时间步的状态部分：只更新前 obs_dim 个维度
                prior[:, 0, :obs_dim] = obs_tensor

                # 采样候选轨迹：
                # prior.repeat(args.num_candidates, 1, 1) 生成形状 (num_candidates*num_envs, horizon, obs_dim+act_dim)
                # 采样返回的 traj 形状为 (num_candidates*num_envs, horizon, obs_dim+act_dim)
                traj, log = agent.flow_matching_sample(
                    prior.repeat(args.num_candidates, 1, 1),
                    solver=args.solver,
                    n_samples=args.num_candidates,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema,
                    w_cg=args.task.w_cg,
                    temperature=args.temperature
                )

                # 处理 log_probability：
                # 重塑 log["log_p"] 为 (num_candidates, num_envs, remaining)
                # 然后沿最后一个维度（通常是时间步内的得分）求和，得到 (num_candidates, num_envs)
                logp = log["log_p"].view(args.num_candidates, 1, -1).sum(-1)
                # 对每个环境（第二维）选择最佳候选的索引
                idx = logp.argmax(0)  # 形状为 (num_envs,)

                # 将 traj 重塑为 (num_candidates, num_envs, horizon, obs_dim+act_dim)
                traj_reshaped = traj.view(args.num_candidates, 1, args.task.horizon, -1)
                # 对每个环境选取最佳候选的第 0 步动作部分，从 obs_dim 开始表示动作
                act_tensor = traj_reshaped[idx, torch.arange(1), 0, obs_dim:]
                # 限制动作范围，并转换为 numpy 数组（形状 (num_envs, act_dim)）
                act = act_tensor.clip(-1., 1.).cpu().numpy()

                # 使用选定的动作与环境交互，返回新的观测、奖励、完成状态等
                obs, rew, done, info = env_eval.step(act)
                t += 1
                # 如果 cum_done 尚未初始化则设为 done，否则逐步进行逻辑或
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew

                print(f"[t={t}] cum_rew: {ep_reward}, logp: {logp[idx, torch.arange(1)]}")

            # 限制每个 episode 的累计奖励在 [0,4] 范围内（根据任务最大奖励设定）
            episode_rewards.append(np.clip(ep_reward, 0., 4.))

        # 将所有 episode 的奖励转为 numpy 数组，并计算均值和标准差
        episode_rewards = get_normalized_score(ds, episode_rewards)
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
        
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    pipeline()

        