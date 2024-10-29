import yaml
import gymnasium as gym
from PDDDQN import DuelingDQNCNN, FrameStacker, preprocess_observation, RewardNormalizer
from ale_py import ALEInterface
import torch
import os
import matplotlib.pyplot as plt
import random
import gc

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
training_params = config['training']

# Extract DQN hyperparameters
dqn_params = config['dqn']
gamma = dqn_params['gamma']
learning_rate = dqn_params['learning_rate']
epsilon_start = dqn_params['epsilon_start']
epsilon_end = dqn_params['epsilon_end']
epsilon_decay = dqn_params['epsilon_decay']
target_update_freq = dqn_params['target_update']
batch_size = dqn_params['batch_size']
memory_capacity = dqn_params['memory_capacity']
num_episodes = dqn_params['num_episodes']
max_steps_per_episode = dqn_params['max_steps_per_episode']
model_update_frequency = 250

# Extract environment settings
env_params = config['env']
env_name = env_params['name']
render = env_params['render']

# Initialize the environment
env = gym.make(env_name, render_mode="rgb_array" if render else None)
print("Action Space:", env.action_space)
print("Number of Actions:", env.action_space.n)

# Reset the environment to get the first observation
observation, info = env.reset()

print("Observation shape:", observation.shape)

# Initialize the DQN and target network
input_dim = (4, 84, 84)
output_dim = env.action_space.n
dqn = DuelingDQNCNN(input_dim, output_dim)
dqn_target = DuelingDQNCNN(input_dim, output_dim)  # Target network
dqn_target.load_state_dict(dqn.state_dict())  # Sync with main DQN
dqn_target.eval()  # Target network is used only for inference

# Load a saved model if it exists
model_save_path = 'pdddqn_model.pth'
if os.path.exists(model_save_path):
    dqn.load_state_dict(torch.load(model_save_path))
    print(f"Loaded saved model from {model_save_path}")
else:
    print("No saved model found, starting from scratch.")

# Frame stacker and replay memory
stacker = FrameStacker(4)
episode_rewards = []
loss_values = []

# Plot setup for training progress
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Total Reward', color='blue')
ax.set_xlim(0, num_episodes)
ax.set_ylim(0, 500)
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Training Progress')
ax.legend()
ax.grid()

# Initialize rolling average for loss with a smoothing factor (e.g., alpha)
rolling_avg_loss = None
alpha = 0.1  # Weight for rolling average (higher alpha means faster adaptation)

# Initialize reward normalizer
reward_normalizer = RewardNormalizer()

for episode in range(num_episodes):
    # Reset the environment and get the initial observation
    observation, info = env.reset()
    if observation is None or observation.size == 0:
        print("Invalid initial observation.")
        continue  # Skip episode if invalid

    # Preprocess the initial observation and reset the frame stacker
    preprocessed_observation = preprocess_observation(observation)
    observation = stacker.reset(preprocessed_observation)  # Reset stacker with the preprocessed observation

    total_reward = 0
    done = False
    steps = 0
    previous_lives = info.get('lives', 3)  # Track the number of lives, default to 3 if not provided

    while not done and steps < max_steps_per_episode:
        # Epsilon-greedy action selection
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                # Convert observation to tensor and ensure correct shape
                state_tensor = torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension
                action = dqn.select_action(state_tensor)

        # Take a step in the environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        if next_observation is None or next_observation.size == 0:
            print("Invalid next observation.")
            break

        # Check if a life was lost and apply -100 penalty
        current_lives = info.get('lives', previous_lives)
        if current_lives < previous_lives:
            reward -= 100
        previous_lives = current_lives

        # Update the normalizer with the new reward value
        reward_normalizer.update(reward)

        # Normalize the reward
        reward = reward_normalizer.normalize(reward)

        # Preprocess the next observation and stack frames
        preprocessed_next_observation = preprocess_observation(next_observation)
        next_observation = stacker.add_frame(preprocessed_next_observation)

        # Store the transition in the DQN's memory
        dqn.remember(observation, action, reward, next_observation, terminated)

        # Perform replay after warmup
        if len(dqn.memory) >= training_params['warmup_steps']:
            beta = min(1.0, training_params['beta_start'] + episode * (1.0 - training_params['beta_start']) /
                       training_params['beta_frames'])
            loss = dqn.replay(batch_size, beta=beta)
            if loss is not None:
                # Update the rolling average loss
                if rolling_avg_loss is None:
                    rolling_avg_loss = loss.item()  # Initialize on first loss
                else:
                    rolling_avg_loss = alpha * loss.item() + (1 - alpha) * rolling_avg_loss

        # Log the action every 100 steps
        if steps % 100 == 0:
            print(f"Episode {episode + 1}, Step {steps}, Action: {action}")

        total_reward += reward
        done = terminated or truncated
        observation = next_observation
        steps += 1

    # Soft update the target network periodically
    if episode % target_update_freq == 0:
        tau = 0.005  # Soft update parameter
        for target_param, param in zip(dqn_target.parameters(), dqn.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # Track and display rewards
    episode_rewards.append(total_reward)
    line.set_xdata(range(1, len(episode_rewards) + 1))
    line.set_ydata(episode_rewards)
    ax.set_ylim(0, max(episode_rewards) + 100)
    fig.canvas.draw()
    fig.canvas.flush_events()
    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward} - Rolling Average Loss: {rolling_avg_loss if rolling_avg_loss is not None else 'N/A'}")

    # Manually force garbage collection to free up memory
    gc.collect()

    # If you're using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save the model
    if (episode + 1) % model_update_frequency == 0:
        torch.save(dqn.state_dict(), model_save_path)
        print(f"Model saved after episode {episode + 1}")

env.close()
plt.ioff()
plt.show()
