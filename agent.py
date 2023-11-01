import torch
import math
import numpy as np
from neural import DeepQNetwork
from memory import ReplayMemory, Transition

class Agent():
    def __init__(self,
                 observations,
                 n_actions,
                 MEMORY_SIZE=10000,
                 BATCH_SIZE = 128,
                 GAMMA = 0.99,
                 EPS_START = 0.9,
                 EPS_END = 5e-2,
                 EPS_DECAY = 1000,
                 TAU = 5e-3,
                 LR = 1e-4,
             ):
        
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.epsilon = 0
        self.steps_done = 0
        self.tau = TAU
        self.lr =LR
        
        self.last_loss = float('inf')

        self.policy_net = DeepQNetwork(observations,  n_actions)
        self.target_net = DeepQNetwork(observations,  n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(MEMORY_SIZE)
        
        self.optimizer = \
            torch.optim.\
            AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        
        self.load_model()
        

    def load_model(self):
        try:
            self.policy_net = torch.load("policy.pth")
            print("model is loaded with sucefull")
        except:
            print("fail to load model. Model weights exits ?")

    def save_model(self):
        torch.save(self.policy_net, "policy.pth")
        torch.save(self.target_net, "target.pth")
         

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

    def decay_eps(self):
        self.epsilon = self.eps_end + (self.eps_start -\
                                   self.eps_end)  *\
        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done +=1

    def choose_action(self, state, env):
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        
        action = torch.tensor(
                    [[env.action_space.sample()]],
                    device=self.policy_net.device, 
                    dtype=torch.long
                            )

        return action
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
      
        batch = Transition(*zip(*transitions))

   
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), 
                                            device=self.policy_net.device, 
                                            dtype=torch.bool
                                    )
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.policy_net.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        if loss < self.last_loss:
            self.last_loss = loss
            self.save_model()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        
       

        


        