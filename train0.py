import yaml
import gymnasium as gym
from ale_py import ALEInterface
from dqncnn import FrameStacker, DQNCNN, preprocess_observation, ReplayBuffer
import torch
import os
import matplotlib.pyplot as plt
import random
import gc
import numpy as np

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract DQN hyperparameters and environment settings from YAML
dqn_params = config['dqn']
training_params = config['training']
env_params = config['env']

# initialize hyperparameters
gamma = dqn_params['gamma']
learning_rate = dqn_params['learning_rate']
epsilon_start = dqn_params['epsilon_start']
epsilon_end = dqn_params['epsilon_end']
target_update_freq = dqn_params['target_update']
batch_size = dqn_params['batch_size']
memory_capacity = dqn_params['memory_capacity']
num_episodes = dqn_params['num_episodes']
max_steps_per_episode = dqn_params['max_steps_per_episode']
model_update_freq = dqn_params['model_update']
warmup = training_params['warmup_steps']

# Initialize environment
env = gym.make(env_params['name'], render_mode="rgb_array" if env_params['render'] else None)
print("Action Space:", env.action_space)
print("Number of Actions:", env.action_space.n)

# Initialize DQN and target network with prioritized memory and noisy nets
input_dim = (4, 42, 42)
output_dim = env.action_space.n
dqn = DQNCNN(input_dim, output_dim)
#target_network = copy.deepcopy(dqn)  # Initialize target network as a copy of the main network
optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)  # Learning rate scheduling

# Load a saved model if it exists
model_save_path = 'Models/dqncnn_model.pth'
if os.path.exists(model_save_path):
    dqn.load_state_dict(torch.load(model_save_path, weights_only=True))
    print(f"Loaded saved model from {model_save_path}")
else:
    print("No saved model found, starting from scratch.")

# Initialize memory and frame stacker
replay_buffer = ReplayBuffer(memory_capacity)
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

# Define the device, use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for episode in range(num_episodes):
    # Environment reset
    observation, info = env.reset()
    if observation is None or observation.size == 0:
        print("Invalid initial observation.")
        continue

    # Preprocess and initialize the frame stacker
    preprocessed_observation = preprocess_observation(observation)
    stacked_observation = stacker.reset(preprocessed_observation)

    total_reward = 0
    done = False
    steps = 0

    # Initialize rolling average for loss with a smoothing factor (e.g., alpha)
    rolling_avg_loss = None
    alpha = 0.1  # Weight for rolling average (higher alpha means faster adaptation)

    # Track the number of lives
    previous_lives = info.get('lives', 3)  # Assume the game starts with 3 lives

    while not done and steps < max_steps_per_episode:
        # Select an action using the policy (epsilon-greedy)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(stacked_observation).unsqueeze(0).to(device)  # Add batch dimension
            action = dqn.select_action(state_tensor)

        # Print episode and action taken every 50 steps
        if steps % 100 == 0:
            print(f"Episode {episode + 1}, Step {steps}, Action: {action}, Epsilon: {dqn.epsilon}")

        # Execute action in the environment
        next_observation, reward, terminated, truncated, info = env.step(action)

        # If the observation is valid, preprocess and add to the frame stacker
        if next_observation is None or next_observation.size == 0:
            print("Invalid next observation.")
            break

        # Preprocess the observation and add to the frame stack
        preprocessed_next_observation = preprocess_observation(next_observation)
        next_stacked_observation = stacker.add_frame(preprocessed_next_observation)

        # Check if a life was lost and apply -1 penalty
        current_lives = info.get('lives', previous_lives)
        if current_lives < previous_lives:
            reward -= 1
        previous_lives = current_lives

        # Clip reward to stabilize training
        reward = np.clip(reward, -1.0, 1.0)

        # Store the transition using the stacked frames
        replay_buffer.push(stacked_observation, action, reward, next_stacked_observation, terminated or truncated)

        # Update the current stacked observation
        stacked_observation = next_stacked_observation

        # Perform replay after warmup
        if len(replay_buffer) >= training_params['warmup_steps']:

            # Sample a batch from the replay buffer
            minibatch = replay_buffer.sample(batch_size)

            # Convert to tensors, ensuring consistency of dimensions
            try:
                state_batch = torch.FloatTensor(minibatch[0]).to(device)  # Already stacked in the replay buffer
                action_batch = torch.LongTensor(minibatch[1]).to(device)
                reward_batch = torch.FloatTensor(minibatch[2]).to(device)
                next_state_batch = torch.FloatTensor(minibatch[3]).to(device)  # Already stacked in the replay buffer
                done_batch = torch.FloatTensor(minibatch[4]).to(device)
            except ValueError as e:
                # Handle shape inconsistency errors
                print("Error in minibatch data shapes:", e)
                continue  # Skip the current replay step if data shapes are inconsistent

            # Compute the target Q-values (Double DQN approach)
            with torch.no_grad():
                next_actions = dqn.forward(next_state_batch).argmax(1)
                next_q_values = dqn.target_network(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze()
                target_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

            # Compute the current Q-values
            current_q_values = dqn.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

            # Compute loss and optimize the model
            loss = torch.nn.SmoothL1Loss()(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn.parameters(),
                                           max_norm=10)  # Clip gradients to prevent them from becoming too large
            optimizer.step()
            scheduler.step()  # Update learning rate only after optimizer step

            # Update the rolling average loss
            if rolling_avg_loss is None:
                rolling_avg_loss = loss.item()  # Initialize on first loss
            else:
                rolling_avg_loss = alpha * loss.item() + (1 - alpha) * rolling_avg_loss

        total_reward += reward
        done = terminated or truncated
        observation = next_observation
        steps += 1
        global_step += 1

    # Update target network periodically
    if (episode + 1) % target_update_freq == 0:
        dqn.update_target_network(tau=0.005)

    # Only decay epsilon after warm-up phase is complete
    if len(replay_buffer) >= training_params['warmup_steps']:
        dqn.decay_epsilon()

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
    print(
        f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward} - Rolling Average Loss: {rolling_avg_loss if rolling_avg_loss is not None else 'N/A'}")

    if (episode + 1) % model_update_freq == 0:
        torch.save(dqn.state_dict(), model_save_path)
        print(f"Model saved after episode {episode + 1}")

    # Collect garbage and free up memory
    gc.collect()
    torch.cuda.empty_cache()

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
