import time
import gym 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
from models import ACNetwork

MAX_EPISODE = 1000
MAX_STEPS = 200
GAMMA = 0.99
LR = 1e-3

LIVE_PLOTTING = False
RENDER = False

class A2CAgent():

    def __init__(self, env, gamma, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.observ_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.gamma = gamma
        self.lr = lr
        
        self.model = ACNetwork(self.observ_dim, self.action_dim)
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
    
    def get_action(self, state): #sample action from policy
        state = torch.FloatTensor(state).to(self.device)
        logits, _ = self.model.forward(state)
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()
    
    def compute_loss(self, trajectory):
        states =  torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor( [sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        
        # compute discounted rewards
        discounted_rewards = torch.FloatTensor([sum([self.gamma**(n-i)*rewards[i] for i in range(n+1)]) for n in list(reversed(range(len(rewards))))])
        target_values = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        
        logits, actual_values = self.model.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        value_loss = F.mse_loss(actual_values, target_values.detach())
        
        # compute entropy bonus
        entropy = torch.stack([-torch.sum(dist.mean() * torch.log(dist)) for dist in dists]).sum()
        
        # compute policy loss
        advantage = target_values - actual_values
        policy_loss = (-probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()).mean()
        
        return policy_loss + value_loss - 0.001 * entropy 
        
    def update(self, trajectory):
        loss = self.compute_loss(trajectory)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


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
    plt.savefig("a2c.png")
    plt.show()


if __name__ == '__main__':

    env = gym.make("CartPole-v0")
    agent = A2CAgent(env, GAMMA, LR)

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

        for steps in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward

            if RENDER:
                env.render()

            if done:
                episode_rewards.append(steps)
                total_rewards += steps
                average_rewards.append(total_rewards/(episode+1))
                break
                
            state = next_state

        if LIVE_PLOTTING:
             plot(total_runtime, episode_rewards, average_rewards, episode_runtime)
 
        print('Episode {}\tReward: {:5f}\t'.format(episode, episode_reward))
        agent.update(trajectory)
        toc = time.time()
        episode_runtime.append(toc-tic)
        total_runtime += (toc - tic)
    
    plot(total_runtime, episode_rewards, average_rewards, episode_runtime)