import torch
import gymnasium as gym
import numpy as np
from NNnSPDDDQN import DQNCNN, FrameStacker, preprocess_observation
from ale_py import ALEInterface

# Configuration parameters
MODEL_PATH = 'Models/nndqn_model.pth'
ENV_NAME = 'ALE/MsPacman-v5'
NUM_TEST_EPISODES = 5
RENDER_MODE = 'human'  # Set to 'human' for rendering; None for faster testing

# Initialize environment and device
env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Action Space:", env.action_space)

# Initialize the DQN model and load the pre-trained weights
input_shape = (4, 84, 84)  # Preprocessed observation shape
num_actions = env.action_space.n
dqn = DQNCNN(input_shape, num_actions, noisy=True, n_step=1).to(device)  # Ensure this matches training setup

# Load the trained model
if torch.cuda.is_available():
    dqn.load_state_dict(torch.load(MODEL_PATH))
else:
    dqn.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
dqn.eval()  # Set model to evaluation mode
print(f"Loaded model from {MODEL_PATH}")

# Initialize frame stacker for stacking frames
frame_stacker = FrameStacker(num_frames=4)

# Run test episodes
for episode in range(NUM_TEST_EPISODES):
    observation, _ = env.reset()
    preprocessed_observation = preprocess_observation(observation)
    stacked_observation = frame_stacker.reset(preprocessed_observation)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Convert the stacked frames to a tensor and select an action with noise disabled
        state_tensor = torch.FloatTensor(stacked_observation).unsqueeze(0).to(device)
        with torch.no_grad():
            action = dqn.select_action(state_tensor, disable_noise=True)

        # Perform the action in the environment
        next_observation, reward, terminated, truncated, _ = env.step(action)



        # Preprocess and update the stacked frames
        preprocessed_next_observation = preprocess_observation(next_observation)
        stacked_observation = frame_stacker.add_frame(preprocessed_next_observation)

        # Update total reward and step count
        total_reward += reward
        done = terminated or truncated
        steps += 1

    print(f"Episode {episode + 1}/{NUM_TEST_EPISODES} - Total Reward: {total_reward}, Steps: {steps}")


# Close the environment
env.close()
