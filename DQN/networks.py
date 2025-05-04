import torch
import torch.nn as nn
import logging
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q = self.q(x)
        
        logger.info(f'Q value is {q} with shape {q.shape}')
        
        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
        
        
        
