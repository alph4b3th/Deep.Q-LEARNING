import torch

class DeepQNetwork(torch.nn.Module):
    def __init__(self, observations, n_actions):
        super(DeepQNetwork, self).__init__()
        self.observations = observations
        self.n_actions = n_actions

        self.activation = torch.nn.ReLU()
        self.network = torch.nn.Sequential(
             self.activation,
             torch.nn.Linear(self.observations, 500),
             torch.nn.Dropout(0.2),
             self.activation,
             torch.nn.Linear(500, 512),
             torch.nn.Dropout(0.2),
             self.activation,
             torch.nn.Linear(512, 256),
             torch.nn.Dropout(0.2),
             self.activation,
             torch.nn.Linear(256, 256),
             self.activation,
             torch.nn.Linear(256, n_actions)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)
   
    def forward(self, state):
        actions = self.network(state)
        return actions

