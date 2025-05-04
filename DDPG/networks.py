import torch
import torch.nn as nn
import torch.optim as optim
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

def weight_init(net):
    if isinstance(net, nn.Linear):
        nn.init.xavier_normal_(net.weight)
        if net.bias is not None:
            nn.init.constant_(net.bias, 0.0)
    elif isinstance(net, nn.BatchNorm1d):
        nn.init.constant_(net.weight, 1.0)
        nn.init.constant_(net.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(weight_init)
        self.to(device)
        
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.action(x))
        
        return action
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
    
class Critic(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(action_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)
    
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.001)
        self.apply(weight_init)
        self.to(device)
        
    def forward(self, state, action):
        x_s = torch.relu(self.ln1(self.fc1(state)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(action)
        x = torch.relu(x_s + x_a)
        q = self.q(x)
        
        return q
    
    def save_checkpoint(self, checktpoint_file):
        torch.save(self.state_dict(), checktpoint_file)
    
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
    
    
    
        
        
        