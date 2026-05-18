from torch import nn

class SinNet(nn.Module):
    def __init__(self, hidden_neurons):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, 1)
        )
    
    def forward(self, x):
        return self.net(x)
