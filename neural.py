import torch

class DeepQNetwork(torch.nn.Module):
    def __init__(self, lr, inputs_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.inputs_dims = inputs_dims
        self.n_actions = n_actions

        self.activation = torch.nn.ReLU()
        self.network = torch.nn.Sequential(
             self.activation,
             torch.nn.Linear(*self.inputs_dims, 128),
             self.activation,
             torch.nn.Linear(128, 128),
             self.activation,
             torch.nn.Linear(128, 128),
             self.activation,
             torch.nn.Linear(128, n_actions)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)

        try:
            torch.load("deep-q.pth")
            print("model is loaded with sucefull")
        except:
            print("fail to load model. Model weights exits ?")

    def forward(self, state):
        actions = self.network(state)
        return actions

