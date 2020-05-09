import gym
import numpy as np
import time
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


MAX_EPISODE = 1000

gamma = 0.99 #discount factor
lr = 1e-3
seed = 1
render = False
log_interval = 10 #interval between training status logs (default: 10)


env = gym.make('CartPole-v0')
env.seed(seed)
torch.manual_seed(seed)

episode_rewards = []
average_rewards = []
episode_runtime = []
total_runtime = 0
total_rewards = 0


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
optimser = optim.Adam(policy.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()


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
    optimser.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimser.step()
    del policy.rewards[:]
    del policy.probs[:]

def plot():
    plt.ion()
    plt.grid()
    plt.subplots_adjust(hspace = 0.5)

    plt.subplot(311)
    plt.title("Total Runtime: " + "{:.2f}".format(total_runtime) + " s")
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.plot(episode_rewards, 'b-')

    plt.subplot(312)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.plot(average_rewards, 'm-')

    plt.subplot(313)
    plt.xlabel('Episode')
    plt.ylabel('Runtime')
    plt.plot(episode_runtime, 'g-')
    plt.pause(0.000001)
    plt.savefig("REINFORCE.png")


if __name__ == '__main__':
    
    for i_episode in range(MAX_EPISODE):
        tic = time.time()

        state = env.reset()
        for t in range(200): #max time step
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            policy.rewards.append(reward)
            if done:
                episode_rewards.append(t)
                total_rewards += t
                average_rewards.append(total_rewards/(i_episode+1))
                break
        
        plot()
        par = [param.data for param in policy.parameters()]
        # print(i_episode, par[0], "\n", par[1], "\n", par[2], "\n", par[3], "\n")
        # print(len(par), len(par[0]), len(par[1]), len(par[2]), len(par[3]))
        # for name, param in policy.named_parameters():
        #     print(name)

        finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tReward: {:5d}\t'.format(i_episode, t+1))

        toc = time.time()
        episode_runtime.append(toc - tic)
        total_runtime += (toc - tic)

