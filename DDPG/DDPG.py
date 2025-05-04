import torch 
import torch.nn.functional as F
import numpy as np
from networks import Actor, Critic
from buffer import ReplayBuffer
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
class DDPG:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim,
                 actor_fc2_dim, critic_fc1_dim, critic_fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, action_noise=0.1, max_size=1000000,
                 batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.checkpoint_dir = ckpt_dir
        
        self.actor = Actor(alpha=alpha, state_dim=state_dim, action_dim=action_dim, fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_actor = Actor(alpha=alpha, state_dim=state_dim, action_dim=action_dim, fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic = Critic(beta=beta, state_dim=state_dim, action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic = Critic(beta=beta, state_dim=state_dim, action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim, batch_size=batch_size)
        self.update_network_parameters(tau=1.0)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for actor_params, target_actor_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)  # 与 = 不同，copy_() 是真正修改张量内容本身 .data 能够绕过 autograd，就地操作，不会影响反向传播图
        
        for critic_params, target_critic_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def take_action(self, observation, train=True):
        self.actor.eval()
        with torch.no_grad():
            # state = torch.tensor([observation], dtype=torch.float).to(device) # 会变成 shape 为 (1, obs_dim) 的张量，即加了一个 batch 维度：
            # state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(device)  # unsqueeze_(0) 原地操作，否则会开辟新内存
            state = torch.from_numpy(np.array(observation, dtype=np.float32)).unsqueeze(0).to(device)
            action = self.actor.forward(state).squeeze()
        
        if train:
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise), dtype=torch.float).to(device)
            action = torch.clamp(action + noise, -1, 1)
        self.actor.train()
        
        return action.detach().cpu().numpy()
    
    def learn(self):
        if not self.memory.ready():
            return
        
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(device)
        terminals_tensor = torch.tensor(terminals,).to(device)
        
        with torch.no_grad():  # 不需要改 target，到时候延迟跟新即可，copy. 因此关闭求梯度
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            q_target = self.target_critic.forward(next_states_tensor, next_actions_tensor).view(-1)
            # 布尔索引语法，是 PyTorch 的高级张量操作。
            q_target[terminals_tensor] = 0.0   # terminals_tensor = torch.tensor([False, False, True, False]) 将所有done = True （终止） 元素设置为0.0
            target = rewards_tensor + self.gamma * q_target
        q = self.critic.forward(states_tensor, actions_tensor).view(-1)
        
        critic_loss = F.mse_loss(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        
        new_actions_tensor = self.actor.forward(states_tensor)
        actor_loss = -torch.mean(self.critic(states_tensor, new_actions_tensor))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
    
    def save_model(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor{}.pth'.format(episode))
        logger.info('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir + 'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic.save_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}.pth'.format(episode))
        print("Saving critic network successfully!")
        self.target_critic.save_checkpoint(self.checkpoint_dir + 'Target_critic/DDPG_target_critic_{}.pth'.format(episode))
        print("Saving target critic network successfully!")
        
    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir + 'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic.load_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}.pth'.format(episode))
        print('Loading critic network successfully!')
        self.target_critic.load_checkpoint(self.checkpoint_dir + 'Target_critic/DDPG_target_critic_{}.pth'.format(episode))
        print('Loading target critic network successfully!')
        
        
        
            
            
        
        
        
            
        
        
