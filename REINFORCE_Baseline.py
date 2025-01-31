import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

render = False


class Policy(nn.Module):
    def __init__(self,n_states, n_hidden, n_output):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)

        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []

    def forward(self, x):
        x = F.relu(self.linear1(x))
        output = F.softmax(self.linear2(x), dim= 1)
        return output



env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = Policy(n_states, 128, n_actions)
s_value_func = Policy(n_states, 128, 1)


optimiser = optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99



seed = 1
env.seed(seed)
torch.manual_seed(seed)
live_time = []

def loop_episode():

    state = env.reset()
    if render: env.render()
    policy_loss = []
    s_value = []
    state_sequence = []
    log_act_prob = []
    for t in range(1000):
        state = torch.from_numpy(state).unsqueeze(0).float()  # 在第0维增加一个维度，将数据组织成[N , .....] 形式
        state_sequence.append(deepcopy(state))
        action_probs = policy(state)
        m = Categorical(action_probs)
        action = m.sample()
        m_log_prob = m.log_prob(action)
        log_act_prob.append(m_log_prob)
        # policy.log_act_probs.append(m_log_prob)
        action = action.item()
        next_state, re, done, _ = env.step(action)
        if render: env.render()
        policy.reward.append(re)
        if done:
            live_time.append(t)
            break
        state = next_state

    R = 0
    Gt = []

    # get Gt value
    for r in policy.reward[::-1]:
        R = r + gamma * R
        Gt.insert(0, R)
        # s_value_func.sigma.insert(0,sigma)
        # policy.Gt.insert(0,R)


    # update step by step
    for i in range(len(Gt)):

        G = Gt[i]
        V = s_value_func(state_sequence[i])
        delta = G - V

        # update value network
        alpha_w = 1e-3 

        optimizer_w = optim.Adam(policy.parameters(), lr=alpha_w)
        optimizer_w.zero_grad()
        policy_loss_w =-delta
        policy_loss_w.backward(retain_graph = True)
        clip_grad_norm_(policy_loss_w, 0.1)
        optimizer_w.step()

        # update policy network
        optimiser.zero_grad()
        policy_loss_theta = - log_act_prob[i] * delta
        policy_loss_theta.backward(retain_graph = True)
        clip_grad_norm_(policy_loss_theta, 0.1)
        optimiser.step()

    del policy.log_act_probs[:]
    del policy.reward[:]



if __name__ == '__main__':
    for i_episode in range(1000):
        # print(i_episode)
        loop_episode()
        plot(live_time)
        print(live_time[-1])
    # policy.plot(live_time)
