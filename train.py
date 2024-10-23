import yaml
import gymnasium as gym
from ale_py import ALEInterface
from dqn import DQN
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract DQN hyperparameters
dqn_params = config['dqn']
gamma = dqn_params['gamma']
learning_rate = dqn_params['learning_rate']
epsilon_start = dqn_params['epsilon_start']
epsilon_end = dqn_params['epsilon_end']
epsilon_decay = dqn_params['epsilon_decay']
target_update = dqn_params['target_update']
batch_size = dqn_params['batch_size']
memory_capacity = dqn_params['memory_capacity']
num_episodes = dqn_params['num_episodes']
max_steps_per_episode = dqn_params['max_steps_per_episode']

# Extract environment settings
env_params = config['env']
env_name = env_params['name']
render = env_params['render']

# Initialize the Arcade Learning Environment
ale = ALEInterface()
gym.register_envs(ale)

# Create the environment
env = gym.make(env_name, render_mode="human" if render else None)
print("Action Space:", env.action_space)
print("Number of Actions:", env.action_space.n)


# Reset the environment to get the first observation
observation, info = env.reset()

# Check the shape of the observation
print("Observation shape:", observation.shape)

# Set input dimension based on the shape of the observation
input_dim = 210 * 160 * 3  # Flattened input size
output_dim = env.action_space.n  # Get the number of possible actions

# Initialize the DQN model with the correct input dimension and hyperparameters
dqn = DQN(input_dim=input_dim, output_dim=output_dim, epsilon=epsilon_start)

# Load the saved model if it exists
model_save_path = '/Users/25lim/ADV ML AI/Project02_PygAImev2.1/dqn_model.pth'
if os.path.exists(model_save_path):
    dqn.load_state_dict(torch.load(model_save_path, weights_only = True))
    print(f"Loaded saved model from {model_save_path}")
else:
    print("No saved model found, starting from scratch.")

# Initialize a counter for actions
action_counter = 0
log_interval = 25  # Set the interval for logging actions

# List to store total rewards for each episode
episode_rewards = []

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')  # Initialize an empty line
ax.set_xlim(0, num_episodes)  # Set x-axis limits
ax.set_ylim(0, 500)  # Initial y-axis limits (will be updated)
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Training Progress')


# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract environment settings
env_params = config['env']
env_name = env_params['name']
render = env_params['render']

# Example training loop
for episode in range(num_episodes):
    observation, info = env.reset()
    observation = observation.flatten().astype(np.float32)  # Use float32 for observations
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < max_steps_per_episode:
        action = dqn.select_action(observation)  # Use your DQN to select an action
        next_observation, reward, terminated, truncated, info = env.step(action)  # Step in the environment

        next_observation = next_observation.flatten().astype(np.float32)  # Use float32 for observations

        # Store experience in memory
        dqn.remember(observation, action, reward, next_observation, terminated)

        # Train the DQN
        dqn.replay(batch_size)

        # Update the current observation
        observation = next_observation
        total_reward += reward
        done = terminated or truncated
        steps += 1

        # Log the action every 50 steps
        if steps % 50 == 0:
            print(f"Episode {episode + 1}, Step {steps}, Action: {action}")

    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

    # Store total reward for this episode
    episode_rewards.append(total_reward)

    # Save the model at regular intervals
    if (episode + 1) % target_update == 0:
        try:
            torch.save(dqn.state_dict(), model_save_path)
            print(f"Model saved after episode {episode + 1}")
        except Exception as e:
            print(f"Error saving model: {e}")
            sys.exit("Exiting program due to model save failure.")

# Close the environment at the end
env.close()

# Plot the rewards
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards, label='Total Reward', color='blue')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.grid()
plt.ylim(0, max(episode_rewards) + 100)  # Set y-limit 100 higher than the current max reward
plt.savefig('reward_plot.png1')  # Save the plot as a .png file
plt.show()  # Display the plot