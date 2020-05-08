import torch
import torch.optim as optim
import torch.multiprocessing as mp  
import gym
import matplotlib.pyplot as plt

from models import TwoHeadNetwork, ValueNetwork, PolicyNetwork
from worker import Worker, DecoupledWorker

mp.set_start_method('spawn', force=True)

class A3CAgent:
    
    def __init__(self, env, gamma, lr, global_max_episode):
        self.env = env

        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.global_rewards = mp.Manager().dict()
        self.global_network = TwoHeadNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr) 
        self.workers = [Worker(i, env, self.gamma, self.global_network, self.global_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE, self.global_rewards) for i in range(mp.cpu_count())]
    
    def train(self):
        print("Training on {} cores".format(mp.cpu_count()))

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
        print(self.global_rewards)
        plt.grid()
        plt.plot(self.global_rewards.values(), 'b-')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig("a3c.png")
    
    def save_model(self):
        torch.save(self.global_network.state_dict(), "a3c_model.pth")


class DecoupledA3CAgent:
    
    def __init__(self, env, gamma, lr, global_max_episode):
        self.env = env

        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.global_value_network = ValueNetwork(self.env.observation_space.shape[0], 1)
        self.global_policy_network = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.global_value_optimizer = optim.Adam(self.global_value_network.parameters(), lr=lr) 
        self.global_policy_optimizer = optim.Adam(self.global_policy_network.parameters(), lr=lr) 
        
        self.workers = [DecoupledWorker(i, env, self.gamma, self.global_value_network, self.global_policy_network,\
             self.global_value_optimizer, self.global_policy_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE) for i in range(mp.cpu_count())]
    
    def train(self):
        print("Training on {} cores".format(mp.cpu_count()))
        input("Enter to start")

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
    
    def save_model(self):
        torch.save(self.global_value_network.state_dict(), "a3c_value_model.pth")
        torch.save(self.global_policy_network.state_dict(), "a3c_policy_model.pth")

