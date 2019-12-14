import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, seed, fc1_units=64, fc2_units=64):
        super(DQN, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, output_dim)
        )
        
    def forward(self, state):
        qvals = self.layers(state)
        return qvals


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, seed, hidden_units=128):
        super(DuelingDQN, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals