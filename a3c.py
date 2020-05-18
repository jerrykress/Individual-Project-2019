import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp  
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torch.distributions import Categorical
from models import ACNetwork

# mp.set_start_method('spawn', force=True)
MAX_EPISODE = 1000
MAX_STEPS = 200
GAMMA = 0.99
LR = 1e-3

LIVE_PLOTTING = False
RENDER = False

class A3CAgent:
    
    def __init__(self, env, gamma, lr):
        self.env = env

        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)

        self.global_rewards = mp.Manager().dict()
        self.global_runtime = mp.Manager().dict()
        self.global_network = ACNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr) 
        self.workers = [Worker(i, env, self.gamma, self.global_network, self.global_optimizer, self.global_episode, self.global_rewards, self.global_runtime) for i in range(mp.cpu_count())]
    
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
        plt.show()


class Worker(mp.Process):

    def __init__(self, id, env, gamma, global_network, global_optimizer, global_episode, global_rewards, global_runtime):
        super(Worker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "Thread%i" % id
        
        self.env = env
        self.env.seed(id)
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = gamma
        self.local_network = ACNetwork(self.obs_dim, self.action_dim) 

        self.global_runtime = global_runtime
        self.global_rewards = global_rewards
        self.global_network = global_network
        self.global_episode = global_episode
        self.global_optimizer = global_optimizer
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logits, _ = self.local_network.forward(state)
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()
    
    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        
        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]) * rewards[j:]) for j in range(rewards.size(0))]
        
        logits, values = self.local_network.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        value_loss = F.mse_loss(values, value_targets.detach())
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()
        
        # compute policy loss
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        
        total_loss = policy_loss + value_loss - 0.001 * entropy 
        return total_loss

    def update_global(self, trajectory):
        loss = self.compute_loss(trajectory)
        
        self.global_optimizer.zero_grad()
        loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_network.parameters(), self.global_network.parameters()):
            global_params._grad = local_params._grad
        self.global_optimizer.step()

    def sync_with_global(self):
        self.local_network.load_state_dict(self.global_network.state_dict())

    def run(self):
        lock = mp.Lock()
        state = self.env.reset()
        trajectory = [] # [[s, a, r, s', done]]
        episode_reward = 0
        
        while self.global_episode.value < MAX_EPISODE:

            tic = time.time()

            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward

            if done:
                with self.global_episode.get_lock():
                    self.global_episode.value += 1
                    self.global_rewards[self.global_episode.value] = episode_reward
                
                print(self.name + " | episode: "+ str(self.global_episode.value) + " -> Reward: " + str(episode_reward))

                self.update_global(trajectory)
                self.sync_with_global()

                trajectory = []
                episode_reward = 0
                state = self.env.reset()
            
            state = next_state

            toc = time.time()

            with lock:
                self.global_runtime[self.global_episode.value] = toc - tic


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = A3CAgent(env, GAMMA, LR)
    agent.train()