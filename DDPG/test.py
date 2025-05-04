import argparse
import gym
import imageio
from DDPG import DDPG
from utils import scale_action

def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPG on Environment")
    parser.add_argument('--filename', type=str, default='./output_images/LunarLander.gif')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/DDPG/')
    parser.add_argument('--fps', type=int, default=True)
    parser.add_argument('--render', type=bool, default=True)

def main():
    
    args = parse_args()
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPG(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                 critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.checkpoint_dir,
                 batch_size=256)
    agent.load_modles(1000)
    video = imageio.get_writer(args.filename)
    
    done = False
    observation = env.reset()
    while not done:
        if args.render:
            env.render()
        action = agent.choose_action(observation, train=True)
        clipped_action = scale_action(action.copy(), env.action_space.highm, env.action_space.low)
        next_observation, reward, done, info = env.step(clipped_action)
        observation = next_observation
        if args.save_video:
            video.append_data(env.render(mode='rgb_array'))
if __name__ == "__main__":
    main()


    