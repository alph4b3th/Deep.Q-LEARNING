import torch

class DeepQNetwork(torch.nn.Module):
    def __init__(self, lr, inputs_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.inputs_dims = inputs_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = torch.nn.Linear(*self.inputs_dims, self.fc1_dims)
        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = torch.nn.Linear(self.fc2_dims, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.activation = torch.nn.ReLU()

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        actions = self.fc3(x)
        return actions

