import torch
import gymnasium as gym
import numpy as np
from ale_py import ALEInterface

from dqncnn import DQNCNN, preprocess_observation, FrameStacker

# Configuration parameters
MODEL_PATH = 'Models/dqncnn_model.pth'
ENV_NAME = 'ALE/MsPacman-v5'
NUM_TEST_EPISODES = 5
RENDER_MODE = 'human'

# Load the environment
env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
print("Action Space:", env.action_space)
print("Number of Actions:", env.action_space.n)

# Define the device, use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DQN model
input_dim = (4, 42, 42)
output_dim = env.action_space.n
dqn = DQNCNN(input_dim, output_dim)

# Load the trained model
if torch.cuda.is_available():
    dqn.load_state_dict(torch.load(MODEL_PATH))
else:
    dqn.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
print(f"Loaded model from {MODEL_PATH}")

# Set model to evaluation mode
dqn.eval()

dqn.epsilon = 0.05  # Set a small epsilon for near-greedy testing

# Initialize FrameStacker for stacking frames
stacker = FrameStacker(4)

# Run test episodes
for episode in range(NUM_TEST_EPISODES):
    observation, info = env.reset()
    preprocessed_observation = preprocess_observation(observation)
    stacked_observation = stacker.reset(preprocessed_observation)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Convert the stacked frames to a tensor and select an action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(stacked_observation).unsqueeze(0).to(device)
            action = dqn.select_action(state_tensor)

        # Perform the action in the environment
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Preprocess and update the stacked frames
        preprocessed_next_observation = preprocess_observation(next_observation)
        stacked_observation = stacker.add_frame(preprocessed_next_observation)

        # Update reward and step count
        total_reward += reward
        done = terminated or truncated
        steps += 1

    print(f"Episode {episode + 1}/{NUM_TEST_EPISODES} - Total Reward: {total_reward}, Steps: {steps}")

# Close the environment
env.close()
