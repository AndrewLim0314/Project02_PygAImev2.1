import yaml
import gymnasium as gym
from ale_py import ALEInterface
from NNnSPDDDQN import DQNCNN, FrameStacker, preprocess_observation
import torch
import os
import matplotlib.pyplot as plt
import random
import numpy as np

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract DQN hyperparameters and environment settings from YAML
dqn_params = config['dqn']
training_params = config['training']
env_params = config['env']

# Initialize hyperparameters
gamma = dqn_params['gamma']
learning_rate = training_params['learning_rate']
target_update_freq = dqn_params['target_update']
batch_size = training_params['batch_size']
memory_capacity = dqn_params['memory_capacity']
num_episodes = dqn_params['num_episodes']
max_steps_per_episode = dqn_params['max_steps_per_episode']
model_update_freq = dqn_params['model_update']

# Initialize environment
env = gym.make(env_params['name'], render_mode="rgb_array" if env_params['render'] else None)
print("Action Space:", env.action_space)
print("Number of Actions:", env.action_space.n)

# Initialize DQN and target network with prioritized memory and noisy nets
input_dim = (4, 84, 84)
output_dim = env.action_space.n
dqn = DQNCNN(input_dim, output_dim, noisy=True, n_step=training_params.get('n_step', 1))
dqn_target = DQNCNN(input_dim, output_dim, noisy=True, n_step=training_params.get('n_step', 1))
dqn_target.load_state_dict(dqn.state_dict())
dqn_target.eval()

# Load a saved model if it exists
model_save_path = 'Models/nndqn_model.pth'
if os.path.exists(model_save_path):
    dqn.load_state_dict(torch.load(model_save_path, weights_only= True))
    print(f"Loaded saved model from {model_save_path}")
else:
    print("No saved model found, starting from scratch.")

# Initialize memory and frame stacker
stacker = FrameStacker(4)
episode_rewards = []
loss_values = []

# Setup plotting for reward visualization
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
line_reward, = ax1.plot([], [], label='Total Reward', color='blue')
line_loss, = ax2.plot([], [], label='Loss', color='red')
ax1.set_xlim(0, num_episodes)
ax1.set_ylim(0, 500)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.set_title('Training Progress')
ax1.legend()
ax1.grid()

ax2.set_xlim(0, num_episodes)
ax2.set_ylim(0, 10)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.legend()
ax2.grid()

global_step = 0

for episode in range(num_episodes):
    # Environment reset
    observation, info = env.reset()
    if observation is None or observation.size == 0:
        print("Invalid initial observation.")
        continue

    preprocessed_observation = preprocess_observation(observation)
    observation = stacker.reset(preprocessed_observation)

    total_reward = 0
    done = False
    steps = 0

    # Initialize rolling average for loss with a smoothing factor (e.g., alpha)
    rolling_avg_loss = None
    alpha = 0.1  # Weight for rolling average (higher alpha means faster adaptation)

    # Track the number of lives
    prev_lives = info.get('lives', 3)  # Assume the game starts with 3 lives

    while not done and steps < max_steps_per_episode:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action = dqn.select_action(state_tensor)

        # Print episode and action taken every 50 steps
        if steps % 50 == 0:
            print(f"Episode {episode + 1}, Step {steps}, Action: {action}")

        # Execute action in environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        if next_observation is None or next_observation.size == 0:
            print("Invalid next observation.")
            break

        preprocessed_next_observation = preprocess_observation(next_observation)
        next_observation = stacker.add_frame(preprocessed_next_observation)

        # Get the current number of lives
        current_lives = info.get('lives', prev_lives)

        # Penalize the agent for losing a life
        if current_lives < prev_lives:
            reward -= 1  # Subtract an additional point for dying

        # Clip the reward to be within the range [-1, 1]
        reward = np.clip(reward, -1, 1)

        # Update previous lives to the current lives
        prev_lives = current_lives

        # Store transition with prioritized replay
        dqn.remember(observation, action, reward, next_observation, terminated)

        # Perform replay after warmup
        if len(dqn.memory) >= training_params['warmup_steps']:
            beta = min(1.0, training_params['beta_start'] + global_step * (1.0 - training_params['beta_start']) /
                       training_params['beta_frames'])
            loss = dqn.replay(batch_size, step=global_step, beta=beta)
            if loss is not None:
                # Update the rolling average loss
                if rolling_avg_loss is None:
                    rolling_avg_loss = loss.item()  # Initialize on first loss
                else:
                    rolling_avg_loss = alpha * loss.item() + (1 - alpha) * rolling_avg_loss

        if global_step % target_update_freq == 0:
            dqn.update_target_network(tau=0.005)

        total_reward += reward
        done = terminated or truncated
        observation = next_observation
        steps += 1
        global_step += 1

    # Record the rolling average loss for the episode if loss was computed
    if rolling_avg_loss is not None:
        loss_values.append(rolling_avg_loss)

    # Record the total reward for the episode
    episode_rewards.append(total_reward)
    line_reward.set_xdata(range(1, len(episode_rewards) + 1))
    line_reward.set_ydata(episode_rewards)
    line_loss.set_xdata(range(1, len(loss_values) + 1))
    if loss_values:
        line_loss.set_ydata(loss_values)
    ax1.set_ylim(0, max(episode_rewards) + 100)
    ax2.set_ylim(0, max(loss_values) + 1) if loss_values else ax2.set_ylim(0, 10)
    fig.canvas.draw()
    fig.canvas.flush_events()
    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward} - Rolling Average Loss: {rolling_avg_loss if rolling_avg_loss is not None else 'N/A'}")

    if (episode + 1) % model_update_freq == 0:
        torch.save(dqn.state_dict(), model_save_path)
        print(f"Model saved after episode {episode + 1}")

env.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards, label='Total Reward', color='blue')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.grid()
plt.ylim(0, max(episode_rewards) + 100)
plt.savefig('reward_plot_nnddqn.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_values) + 1), loss_values, label='Loss', color='red')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid()
plt.ylim(0, max(loss_values) + 1) if loss_values else plt.ylim(0, 10)
plt.savefig('loss_plot_ddqn.png')
plt.show()
