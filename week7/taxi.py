import matplotlib.pyplot as plt
import random
import numpy as np
import gymnasium as gym

BASE_RANDOM_SEED = 10003

def g_learing(
        env,
        episodes: int = 5000,
        seed: int = BASE_RANDOM_SEED,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
) -> dict:

    np.random.random(seed)
    random.seed()
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))


    episode_rewards = []

    for episode in range(episodes):

        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):

            if np.random.random() < epsilon:
                action = np.random.randint(0, n_actions)

            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            next_max = np.max(q_table[next_state])

            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * next_max - q_table[state, action])

            state = next_state

        episode_rewards.append(total_reward)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": np.mean(episode_rewards),
        "std_rewards": np.std(episode_rewards)
    }
    

def main():
    learning_rate = 0.1
    dicount_factor = 0.95
    epsilon = 0.1
    n_runs = 10

    seeds = [BASE_RANDOM_SEED + i for i in range(n_runs)]
    stats_list = []

    for i, seed in enumerate(seeds):
        env = gym.make("Taxi-v3")
        print(f"Run {i+ 1}/{n_runs} with seed {seed}")

        stats = g_learing(
            env,
            seed=seed,
            learning_rate=learning_rate,
            discount_factor=dicount_factor,
            epsilon=epsilon
        )

        env.close()
        stats_list.append(stats)
        

    for rewards in stats_list:
        plt.plot(
            rewards["episode_rewards"],
            color="blue",
            alpha=0.1
        )
        
    plt.title("Episode Rewards over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

if __name__ == "__main__":
    main()
