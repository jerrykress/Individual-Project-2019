import torch
import torch.nn as nn
import torch.nn.functional as F 


class ACNetwork(nn.Module):

    def __init__(self, observ_dim, action_dim):
        super(ACNetwork, self).__init__()
        #policy network
        self.policy1 = nn.Linear(observ_dim, 256) 
        self.policy2 = nn.Linear(256, action_dim)
        #value network
        self.value1 = nn.Linear(observ_dim, 256)
        self.value2 = nn.Linear(256, 1)
        
    def forward(self, state):
        #policy network
        dist = F.relu(self.policy1(state))
        dist = self.policy2(dist)
        #value network
        value = F.relu(self.value1(state))
        value = self.value2(value)

        return dist, value