import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)  #observation space to hidden layer
        self.fc2 = nn.Linear(128, 2)  #hidden layer to action space

        self.probs = []  #saved log space probability
        self.rewards = []  #saved reward values

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return F.softmax(y, dim=1)