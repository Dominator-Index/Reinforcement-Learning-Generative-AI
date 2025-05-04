import gym
import argparse
import logging
import numpy as np
from DQN import DQN
from utils import plot_learning_curve, create_directory, setup_logger

logger = setup_logger(log_dir="/home/nkd/ouyangzl/DQN/log", log_filename="dqn_train.log", level=logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description="Training DQN")
    parser.add_argument('--max_episodes', type=int, default=500)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DQN/')
    parser.add_argument('--reward_path', type=str, default='./output_images/avg_reward.png')
    parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')
    
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Arguments parsed !")
    env = gym.make('CartPole-v1')
    logger.info("Environment created !")
    agent = DQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []
    
    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation, _ = env.reset()
        if isinstance(observation, np.ndarray) and observation.size > 0:
            logger.info(f"Reset successful! Initial observation: {observation}")
        else:
            logger.error(f"Reset failed! Invalid observation: {observation}")
        
        while not done:
            action = agent.take_action(observation, isTrain=True)
            # === 判断 action 合法性 ===
            if action is None or not isinstance(action, np.ndarray) or action.size == 0:
                logger.error('Action is not taken or invalid!')
            else:
                logger.info(f'Action is {action} with shape {action.shape}')
                
            next_observation, reward, terminateds, truncated, info = env.step(action)
            
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
        
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        logger.info(f'EP: {episode + 1} reward: {total_reward} avg_reward: {avg_reward} epsilon: {agent.epsilon}')
        
        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)
    
    episodes = [i for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)

if __name__ == '__main__':
    main()



