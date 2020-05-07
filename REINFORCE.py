import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


gamma = 0.9
seed = 1
render = False
log_interval = 10 #interval between training status logs (default: 10)


env = gym.make('CartPole-v0')
env.seed(seed)
torch.manual_seed(seed)
live_rewards = []


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


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()
reward_global = []


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.probs[:]

def plot(live_time):
    plt.ion()
    plt.grid()
    plt.plot(live_time, 'g-')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.pause(0.000001)


if __name__ == '__main__':
    
    for i_episode in range(1000):
        state = env.reset()
        for t in range(200): #max time step
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            policy.rewards.append(reward)
            if done:
                live_rewards.append(t)
                break
        
        plot(live_rewards)

        finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tReward: {:5d}\t'.format(i_episode, t+1))
