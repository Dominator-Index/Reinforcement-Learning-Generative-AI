import torch
import torch.nn.functional as F
import numpy as np
from networks import DeepQNetwork
from buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-4, max_size=100000, batch_size=256):
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.action_space = [i for i in range(action_dim)]
        self.checkpoint_dir = ckpt_dir
        
        self.q_eval = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DeepQNetwork(alpha, state_dim=state_dim, action_dim=action_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=max_size, batch_size=batch_size)
        
        self.update_network_parameters(tau=1.0)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1- tau) * q_target_params)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def take_action(self, observation, isTrain=True):
        state = torch.from_numpy(np.array(observation, dtype=np.float32)).unsqueeze(0).to(device)
        qs = self.q_eval.forward(state)
        action = torch.argmax(qs).item()
        
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)
            
        # 如果action是单一的整数，转化为numpy数组
        # action = np.array([action], dtype=np.float32)  # 确保返回一个数组
        
        # 将 action 转换为 PyTorch 张量，再调用 detach() 和 cpu()
        # action_tensor = torch.tensor(action)  # 转换为张量
    
        return action
    
    def learn(self):
        if not self.memory.ready():
            return 
        
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = np.arange(self.batch_size)
        
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
        terminals_tensor = torch.tensor(terminals).to(device)
        
        # Convert actions to torch.Tensor before unsqueeze
        actions_tensor = torch.tensor(actions).to(device).long() # Convert actions to torch.Tensor and move to device
        
        with torch.no_grad():
            q_target = self.q_target.forward(next_states_tensor)
            q_target[terminals_tensor] = 0.0 # 索引操作，为true设1 false 设0
            target = rewards_tensor + self.gamma * torch.max(q_target, dim=-1)[0]  # 返回元组 (value, index)
            
        q_values = self.q_eval.forward(states_tensor)  # actions 就是动作索引
        
        # Use gather to select Q-values corresponding to the actions taken
        q = torch.gather(q_values, dim=1, index=actions_tensor.unsqueeze(1))  # Shape: [batch_size, 1]
        
        loss = F.mse_loss(q, target.detach())  # detach() 防止被计算图追踪
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
        
        self.update_network_parameters()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')
 
    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')

        
        
        
        
            
            