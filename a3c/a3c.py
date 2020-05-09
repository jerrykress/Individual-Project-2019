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
        self.global_runtime = mp.Manager().dict()
        self.global_network = TwoHeadNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr) 
        self.workers = [Worker(i, env, self.gamma, self.global_network, self.global_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE, self.global_rewards, self.global_runtime) for i in range(mp.cpu_count())]
    
    def train(self):
        print("Training on {} cores".format(mp.cpu_count()))

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]

        #Plotting
        plt.grid()
        plt.subplots_adjust(hspace = 0.5)

        plt.subplot(311)
        plt.title("Total Runtime: " + "{:.2f}".format(sum(self.global_runtime.values())) + " s")
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.plot(self.global_rewards.values(), 'b-')

        plt.subplot(312)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.plot([sum(self.global_rewards.values()[0:i+1])/(i+1) for i in range(len(self.global_rewards.values())) ], 'm-')

        plt.subplot(313)
        plt.xlabel('Episode')
        plt.ylabel('Runtime')
        plt.plot(self.global_runtime.values(), 'g-')
        plt.savefig("a3c.png")
        # plt.show()


    
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

