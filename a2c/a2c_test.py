import time
import gym 
from a2c import A2CAgent
import matplotlib.pyplot as plt


env = gym.make("CartPole-v0")
# obs_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
MAX_EPISODE = 50
MAX_STEPS = 200

lr = 1e-3
gamma = 0.99

agent = A2CAgent(env, gamma, lr)

episode_rewards = []
average_rewards = []
episode_runtime = []
total_runtime = 0
total_rewards = 0

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
    plt.savefig("a2c.png")

if __name__ == '__main__':
    for episode in range(MAX_EPISODE):

        tic = time.time()

        state = env.reset()
        trajectory = [] # [[s, a, r, s', done], [], ...]
        episode_reward = 0
        for steps in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward

            # env.render()

            if done:
                episode_rewards.append(steps)
                total_rewards += steps
                average_rewards.append(total_rewards/(episode+1))
                break
                
            state = next_state

        plot()

        if episode % 10 == 0:
            print("Episode " + str(episode) + ": " + str(episode_reward))
        agent.update(trajectory)

        toc = time.time()
        episode_runtime.append(toc-tic)
        total_runtime += (toc - tic)
