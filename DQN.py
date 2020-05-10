#  A pole is attached by an un-actuated joint to a cart,
#  which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart.
#  The pendulum starts upright, and the goal is to prevent it from falling over.
#  A reward of +1 is provided for every timestep that the pole remains upright.
#  The episode ends when the pole is more than 15 degrees from vertical,
#  or the cart moves more than 2.4 units from the center.

import time
import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
import numpy as np

MAX_EPISODE = 1000

env = gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)

episode_rewards = []
episode_runtime = []
average_rewards = []
total_rewards = 0
total_runtime = 0


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .99
        self.batch_size = 50
        self.epsilon_min = .01
        self.epsilon_decay = .99
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        return model

    def memorise(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
    plt.savefig("dqn.png")
 

if __name__ == '__main__':

    agent = DQN(env.action_space.n, env.observation_space.shape[0])

    for episode in range(MAX_EPISODE):
        tic = time.time()

        state = env.reset()
        state = np.reshape(state, (1, 4))
        max_steps = 1000
        for i in range(max_steps):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, 4))
            agent.memorise(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("Episode {}: {}".format(episode, i+1))
                episode_rewards.append(i+1)
                total_rewards += (i+1)
                average_rewards.append(total_rewards/(episode+1))
                # plot()
                break
        
        toc = time.time()

        episode_runtime.append(toc-tic)
        total_runtime += (toc-tic)
    
    plot()
