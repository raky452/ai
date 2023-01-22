import gym
import numpy as np

# Define the environment
env = gym.make("FrozenLake-v1")

# Define the Q-table and the learning rate
Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.8

# Define the number of episodes
num_episodes = 2000

# Run the Q-learning algorithm
for i in range(num_episodes):
    # Reset the environment
    state,_ = env.reset()
    done = False
    while not done:
        # Choose an action
        q_values = Q[state, :]
        action = np.argmax(q_values + (np.random.randn(1) / (i + 1)))
        # Take the action and observe the next state and reward
        next_state, reward, terminated,truncated, _ = env.step(action)
        done = terminated or truncated
        # Update the Q-table
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + np.max(Q[next_state, :]))
        state = next_state

# Print the final Q-table
print(Q)
