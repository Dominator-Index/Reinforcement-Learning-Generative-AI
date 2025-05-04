import numpy as np
import logging

logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.poniter = 0
        
        self.state_memory = np.zeros((self.max_size, state_dim))
        self.action_memory = np.zeros((self.max_size, ))
        self.reward_memory = np.zeros((self.max_size,))
        self.next_state_memory = np.zeros((self.max_size, state_dim))
        self.terminal_memory = np.zeros((self.max_size, ), dtype=bool)
    
    def store_transition(self, state, action, reward, next_state, done):
        index = self.poniter % self.max_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        
        self.poniter += 1
    
    def sample_buffer(self):
        buffer_len = min(self.max_size, self.poniter)
        
        batch = np.random.choice(buffer_len, self.batch_size, replace=True, p=None)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]
        
        return states, actions, rewards, next_states, terminals
    
    def ready(self):
        return self.poniter > self.batch_size
        
    
    