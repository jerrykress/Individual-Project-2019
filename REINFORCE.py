import time
from itertools import count
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


MAX_EPISODE = 1000
MAX_STEPS = 200
GAMMA = 0.99
LR = 1e-3

LIVE_PLOTTING = True
RENDER = False

class PolicyNetwork(nn.Module):
    def __init__(self, observ_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(observ_dim, 128)  #observation space to hidden layer
        self.fc2 = nn.Linear(128, action_dim)  #hidden layer to action space

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=0)

class REINFORCEAgent():

    def __init__(self, env, gamma, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.observ_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.gamma = gamma
        self.lr = lr
        
        self.model = PolicyNetwork(self.observ_dim, self.action_dim)
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

        self.saved_probs = []

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dist = self.model(state)
        probs = Categorical(dist)
        action = probs.sample()
        self.saved_probs.append(probs.log_prob(action))

        return action.cpu().detach().item()

    def compute_loss(self, trajectory):
        rewards = [sars[2] for sars in trajectory]
        discounted_rewards = torch.tensor([sum([0.99**(n-i)*rewards[i] for i in range(n+1)]) for n in list(reversed(range(len(rewards))))])
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())
        policy_loss = [-log_prob * reward for log_prob, reward in zip(self.saved_probs, discounted_rewards)]
        
        return torch.stack(policy_loss).sum()

    def update(self, trajectory):
        loss = self.compute_loss(trajectory)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del self.saved_probs[:]

def plot(total_runtime, episode_rewards, average_rewards, episode_runtime):
    if LIVE_PLOTTING:
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
    plt.show()


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    agent = REINFORCEAgent(env, GAMMA, LR)

    episode_rewards = []
    average_rewards = []
    episode_runtime = []
    total_runtime = 0
    total_rewards = 0
    
    for episode in range(MAX_EPISODE):
        tic = time.time()
        episode_reward = 0
        trajectory = [] # [[s, a, r, s', done]]
        state = env.reset()

        for step in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward

            if RENDER:
                env.render()

            if done:
                episode_rewards.append(step)
                total_rewards += step
                average_rewards.append(total_rewards/(episode+1))
                break

            state = next_state
        
        if LIVE_PLOTTING:
            plot(total_runtime, episode_rewards, average_rewards, episode_runtime)

        print('Episode {}\tReward: {:5f}\t'.format(episode, episode_reward))
        agent.update(trajectory)
        toc = time.time()
        episode_runtime.append(toc - tic)
        total_runtime += (toc - tic)

    plot(total_runtime, episode_rewards, average_rewards, episode_runtime)
