import os
import argparse
import gym
import numpy as np
from DDPG import DDPG
import logging
from utils import  create_directory, plot_learning_curve, scale_action

from utils import setup_logger
logger = setup_logger(log_dir="/home/nkd/ouyangzl/DQN/log", log_filename='ddpg_train.log', level=logging.DEBUG)
            
def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPG on Environment")
    parser.add_argument("--max_episodes", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint/DDPG/')
    parser.add_argument('--figure_file', type=str, default='/output_images/reward.png')
    
    return parser.parse_args()

def main():
    logger.info("Start Training !")
    args = parse_args()
    logger.info("Arguments are parsed !")
    env = gym.make('MountainCarContinuous-v0')
    logger.info("Making Environments")
    agent = DDPG(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                 critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.checkpoint_dir,
                 batch_size=256)
    create_directory(args.checkpoint_dir, sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

    
    reward_history = []
    avg_reward_history = []
    
    for episode in range(args.max_episodes):
        logger.info(f'Epoch: {episode}')
        done = False
        total_reward = 0
        observation, _ = env.reset()
        while not done:
            action = agent.take_action(observation, train=True)
            
            # === 判断 action 合法性 ===
            if action is None or not isinstance(action, np.ndarray) or action.size == 0:
                logger.error('Action is not taken or invalid!')
            else:
                logger.info(f'Action is {action} with shape {action.shape}')

            # === 执行动作 ===
            clipped_action = scale_action(action.copy(), env.action_space.high, env.action_space.low)
            next_observation, reward, terminated, truncated, info = env.step(clipped_action)

            # === 判断 next_observation 和 reward 合法性 ===
            if next_observation is None or not isinstance(next_observation, np.ndarray) or next_observation.size == 0:
                logger.error('Next observation is invalid!')
            else:
                logger.info(f'Next observation is {next_observation} with shape {next_observation.shape}')

            # === 判断 reward 合法性 ===
            if reward is None or (isinstance(reward, np.ndarray) and reward.size == 0):
                logger.error('Reward is invalid!')
            else:
                logger.info(f'Reward is {reward} with type {type(reward)}')
                
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            total_reward += reward
            observation = next_observation
        
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_history.append(avg_reward)
        logger.info('Ep: {} Reward: {:.1f} AvgReward: {:.1f}'.format(episode + 1, total_reward, avg_reward))
        
        if (episode + 1) % 200 == 0:
            agent.save_models(episode + 1)
    
    episodes = [i+1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward', ylabel='reward', figure_file=args.figure_file)

if __name__ == '__main__':
    args = parse_args()
    main()
