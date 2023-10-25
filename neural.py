import torch

class DeepQNetwork(torch.nn.Module):
    def __init__(self, lr, inputs_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.inputs_dims = inputs_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1_dims = torch.nn.Linear(*self.inputs_dims, self.fc1_dims)
        self.fc2_dims = torch.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3_dims = torch.nn.Linear(self.fc2_dims, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, state):
        x = torch.nn.ReLU(self.fc1_dims(state))
        x = torch.nn.ReLU(self.fc2_dims(x))
        actions = torch.nn.ReLU(self.fc3_dims(x))
        return actions

