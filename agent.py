import torch
import numpy as np
from neural import DeepQNetwork

class Agent():
    def __init__(self,
                 gamma, 
                 epsilon,
                 lr, 
                 input_dims,
                 batch_size, 
                 n_actions,
                 max_memory_size=100_000, 
                 eps_end = 0.01,
                 eps_decay = 5e-4
             ):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_memory_size = max_memory_size
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.action_space = [B for in range(n_actions)]
        self.mem_cntr = 0 
        self.Q_eval = DeepQNetwork(
            self.lr, 
            n_actions=n_actions, 
            inputs_dims=input_dims,
            fc1_dims=256, fc2_dims= 256
            )
        
        self.memory_state = np.zeros((self.max_memory_size, *input_dims), dtype=np.float32)
        self.new_memory_state = np.zeros(
            (self.memory_state, *input_dims), 
            dtype=np.float32)

        self.memory_action = np.zeros(self.max_memory_size, dtype=np.int32)
        self.memory_reward = np.zeros(self.max_memory_size, dtype=np.float32)
        self.memory_terminal = np.zeros(self.max_memory_size, dtype=np.bool)

        