import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import random
from collections import deque
import copy

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store a transition in the replay buffer; state and next_state are stacked frames
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the replay buffer
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Use np.stack to ensure consistent shapes when stacking frames
        return (
            np.stack(states),  # Stack all states to maintain consistent shape
            np.array(actions),
            np.array(rewards),
            np.stack(next_states),  # Stack all next states to maintain consistent shape
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)



class FrameStacker:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = []

    def reset(self, first_frame):
        # Initialize stack with the first frame repeated
        self.frames = [first_frame] * self.num_frames
        return np.concatenate(self.frames, axis=0)  # Concatenate along the channel axis

    def add_frame(self, new_frame):
        if len(self.frames) >= self.num_frames:
            self.frames.pop(0)  # Remove the oldest frame
        self.frames.append(new_frame)
        return np.concatenate(self.frames, axis=0)  # Concatenate along the channel axis

def preprocess_observation(observation):
    # Convert the observation to grayscale
    gray_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    # Resize the observation to 42x42
    resized_observation = cv2.resize(gray_observation, (42, 42))

    # Normalize the pixel values to [0, 1]
    normalized_observation = resized_observation / 255.0

    # Add a channel dimension (1, 42, 42) to match the expected input shape
    return np.expand_dims(normalized_observation, axis=0)



class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCNN, self).__init__()
        # Define the main network
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)  # Reduced from 32 to 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # Reduced from 64 to 32 channels
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # Reduced from 64 to 32 channels

        # Calculate the size after convolutions
        convw = (input_shape[1] - 8) // 4 + 1
        convw = (convw - 4) // 2 + 1
        convw = (convw - 3) // 1 + 1
        convh = (input_shape[2] - 8) // 4 + 1
        convh = (convh - 4) // 2 + 1
        convh = (convh - 3) // 1 + 1
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 128)  # Input size is calculated dynamically
        self.fc2 = nn.Linear(128, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Replay buffer
        self.memory = ReplayBuffer(2000)  # Reduced from 10000 to 2000
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Initialize the target network as a copy of the main network
        self.target_network = copy.deepcopy(self)
        self.target_network.to(self.device)  # Ensure target network is moved to the correct device

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def remember(self, state, action, reward, next_state, done):
        # Clip the reward to prevent extreme values
        clipped_reward = np.clip(reward, -1.0, 1.0)
        # Store transition in replay buffer
        self.memory.push(state, action, clipped_reward, next_state, done)

    def select_action(self, state):
        # Convert state to a tensor
        state_tensor = torch.FloatTensor(state).to(self.device)

        # Check if the state tensor needs a batch dimension
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension, shape becomes (1, 4, 42, 42)

        if random.random() < self.epsilon:
            return random.randrange(self.fc2.out_features)  # Random action
        else:
            with torch.no_grad():
                q_values = self.forward(state_tensor)  # Forward pass
            return torch.argmax(q_values).item()  # Return the action with the highest Q-value

    def replay(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return

        minibatch = self.memory.sample(batch_size)

        # Using np.stack to ensure all states have the same shape
        state_batch = torch.FloatTensor(np.stack([s[0] for s in minibatch])).to(self.device)
        action_batch = torch.LongTensor(np.array([s[1] for s in minibatch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([s[2] for s in minibatch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.stack([s[3] for s in minibatch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([s[4] for s in minibatch])).to(self.device)

        with torch.no_grad():
            # Main network to select the best action for the next state
            next_actions = self.forward(next_state_batch).argmax(1)
            # Target network to evaluate the value of the selected actions
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

        current_q_values = self.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    def update_target_network(self, tau=1.0):
        # Soft update for the target network
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
