import time
import random
import numpy as np
import matplotlib.pyplot as plt

import gym
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam

MAX_EPISODE = 1000
MAX_STEPS = 200

BATCH = 32
GAMMA = 0.99
DECAY = 0.99
LR = 1e-3

LIVE_PLOTTING = True
RENDER = False

class DQN:

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space

        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=LR))

        self.model = model
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=10000)

    def get_action(self, state): #sample action from policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def replay(self): #memory replay
        if len(self.memory) < BATCH:
            return

        batch =        random.sample(self.memory, BATCH)
        states =       np.array([sars[0] for sars in batch])
        actions =      np.array([sars[1] for sars in batch])
        rewards =      np.array([sars[2] for sars in batch])
        next_states =  np.array([sars[3] for sars in batch])
        dones =        np.array([sars[4] for sars in batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + GAMMA * (np.max(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        targets_full[[range(BATCH)], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= DECAY

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
    plt.savefig("dqn.png")
    plt.show()
 

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    np.random.seed(0)

    episode_rewards = []
    episode_runtime = []
    average_rewards = []
    total_rewards = 0
    total_runtime = 0

    agent = DQN(env.action_space.n, env.observation_space.shape[0])

    for episode in range(MAX_EPISODE):
        tic = time.time()
        state = env.reset()
        state = np.reshape(state, (1, 4))

        for i in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, 4))
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            agent.replay()

            if RENDER:
                env.render()
                
            if done:
                print("Episode {}: {}".format(episode, i+1))
                episode_rewards.append(i+1)
                total_rewards += (i+1)
                average_rewards.append(total_rewards/(episode+1))
                
                break

        if LIVE_PLOTTING:
            plot(total_runtime, episode_rewards, average_rewards, episode_runtime)

        toc = time.time()

        episode_runtime.append(toc-tic)
        total_runtime += (toc-tic)
    
    plot(total_runtime, episode_rewards, average_rewards, episode_runtime)
