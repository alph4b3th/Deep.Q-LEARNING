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
        self.last_loss = float('inf')
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_memory_size = max_memory_size
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.action_space = [B for B in range(n_actions)]
        self.mem_cntr = 0 
        self.Q_eval = DeepQNetwork(
            self.lr, 
            n_actions=n_actions, 
            inputs_dims=input_dims,
            fc1_dims=256, fc2_dims= 256
            )
        
        self.memory_state = np.zeros((self.max_memory_size, *input_dims), dtype=np.float32)
        self.new_memory_state = np.zeros((self.max_memory_size, *input_dims), dtype=np.float32)
       # self.new_memory_state = np.zeros((self.memory_state.shape[0], *input_dims), dtype=np.float32)

        self.memory_action = np.zeros(self.max_memory_size, dtype=np.int32)
        self.memory_reward = np.zeros(self.max_memory_size, dtype=np.float32)
        self.memory_terminal = np.zeros(self.max_memory_size, dtype=np.bool_)

    def store_transaction(
            self,
            state,
            action,
            reward,
            state_,
            done
    ):
      

        index = self.mem_cntr %  self.max_memory_size   
        self.memory_state[index] = state
        self.new_memory_state[index] = state_
        self.memory_reward[index] = reward
        self.memory_action[index] = action
        self.memory_terminal[index] = done

        self.mem_cntr+=1

    def choose_action(self, observation ):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            action = self.Q_eval.forward(state)
            action = torch.argmax(action).item()
            return action
        
        action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        mem_max = min(self.mem_cntr, self.max_memory_size)
        batch = np.random.choice(mem_max, self.batch_size, replace=False)

        batch_idx = np.arange(self.batch_size, dtype=np.int32   )

        batch_state = torch.tensor(self.memory_state[batch]).to(self.Q_eval.device)
        batch_new_state = torch.tensor(self.new_memory_state[batch]).to(self.Q_eval.device)
        batch_reward = torch.tensor(self.memory_reward[batch]).to(self.Q_eval.device)
        batch_terminal = torch.tensor(self.memory_terminal[batch]).to(self.Q_eval.device)

        batch_action = self.memory_action[batch]

        q_eval = self.Q_eval.forward(batch_state)[batch_idx, batch_action]
        q_next = self.Q_eval.forward(batch_new_state)
        q_next[batch_terminal] = 0.0

        q_target = batch_reward + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss_fn(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
      
        self.Q_eval.optimizer.step()

        self.epsilon -= self.eps_decay if self.epsilon > self.eps_end \
                                       else self.eps_end
        
        if loss < self.last_loss:
            self.last_loss = loss
            torch.save(q_eval, "deep-q.pth")

        


        